import re
import numpy as np
import pandas as pd
from typing import Iterator
from pathlib import Path

import pytesseract as pt
import nltk # pip install nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')
from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder
trigram_measures = nltk.collocations.TrigramAssocMeasures()
bigram_measures = nltk.collocations.BigramAssocMeasures()

# if you want to filter out entities of different categories... leaving in for now as people stray from the path
# from nltk.tag.stanford import NERTagger

import wikipedia #!pip install wikipedia
import datetime
from datetime import datetime as dt
import timeit
from random import sample

import spacy
# python -m spacy download en_core_web_lg
import en_core_web_lg

# pip install memoization
from memoization import cached 

from video import Video

# import timeit

def log(msg, type:str='generic'):
    with open('log.txt', mode='a+') as file:
        file.write(f'\n{"-"*50}\n{datetime.datetime.now()} ~~ type:{type} ~~ msg:{msg}')
        
def remove_stop_words(words):
    # TODO: get proper dictionaries
    stops = set(stopwords.words("english")) 
    return ' '.join([w for w in words.split(' ') if not w in stops])

def has_flagged_word(words, threshold=2):
    # TODO: get proper dictionaries and combine at init
    extensions = {'.pptx', '.ppt', '.odp', '.jpg', '.png', '.svg', '.doc', '.docx', 'pptx', 'ppt', 'odp', 'jpg', 'png', 'svg', 'doc', 'docx'}
    interface = {'file', 'home', 'search', 'play', 'bookmarks', 'bookmark', 'save', 'load', 'powerpoint', 'slides', 'slide', 'slideshow', 'insert', 'record', 'view', 'transitions', 'animations', 'endnote', 'share', 'help', 'endnote'}
    extensions.update(interface)

    for w in words.split(' '):
        if w.lower() in extensions:
            threshold -= 1
            if threshold <= 0:
                return True
    return False

def has_flagged_symbol(words):
    symbols = {'©', '0xC2 0xA9', 'U+0040', '@', '0x40'}
    for sym in symbols:
        if re.search(sym, words):
            return True  
    return False

# expects string field not lists
def remove_non_alpha(document):
    regex = re.compile(r'[^a-zA-Z]')
    return regex.sub(' ', document)
        

def get_text_from_frames(vid_path:str, start:float, end:float=None):
    vid = Video(vid_path)    
    end = vid.duration if end is None else end
    frames_iter = vid.get_frames(start, end, fpm=2)

    # path to tesseract.exe
    pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    
    def __get_text(frames_iter:Iterator):
        text = []
        for f in frames_iter:
            print(np.shape(f[1]))
            # Note that without getting times it would be just f not a tuple
            t = pt.image_to_string(f[1])
            # t = pt.image_to_string(f[1], config='--psm 4')
            text.append(t)
        return text
        
    text = __get_text(frames_iter)
    return text

def get_best_filtered(filtered:list[(str, float)], n_best:int=1)->list[str]:
    return sorted(filtered, key=lambda x: x[1], reverse=n_best > 0)[:np.abs(n_best)]

# pass a negative value for n_best to get worst
def filter_by_similarity(topics:set[str], n_best:int=None, page:bool=False, thresh_low:float=0.98, thresh_high:float=1.0, verbose:bool=True)->set[str]:
    nlp = spacy.load("en_core_web_lg")
    topics_list = [t for t in topics]
    topic_corpus = get_wiki_content(topics, page)
    # comparing the new topics against the existing set
    # searched_corpus = ' '.join(get_wiki_content([t for t in searched]))
    corpus_vec = nlp(' '.join(topic_corpus))
    # corpus_vec = nlp(' '.join(get_wiki_content([t for t in topics])))
    passed = []
    failed = []

    # TODO: compare vectors instead of reprocessing every time
    for i, doc in enumerate(nlp.pipe(topic_corpus)):
        similarity = doc.similarity(corpus_vec)
        if similarity >= thresh_low and similarity <= thresh_high:
            passed.append((topics_list[i], similarity))
        else:
            failed.append((topics_list[i], similarity))
    if verbose:
        for f in failed:
            print(f'\n{"-"*50}\nfailed threshold: {f[0]} score = {f[1]}')
        for p in passed:
            print(f'\n{"-"*50}\npassed threshold: {p[0]} score = {p[1]}')
    
    if n_best is not None:
        passed = get_best_filtered(passed, n_best)
        
    passed = set(remove_non_alpha(p[0]) for p in passed)
    return passed

def get_collocations(corpus:list, n_collocs:int=3)->list[(str,str)]:
    if type(corpus) == str:
        corpus = corpus.split(' ')
    tri_finder = TrigramCollocationFinder.from_words(corpus)
    t = tri_finder.nbest(trigram_measures.likelihood_ratio, n_collocs)

    bi_finder = BigramCollocationFinder.from_words(corpus)
    b = bi_finder.nbest(bigram_measures.likelihood_ratio, n_collocs)

    # TODO: stem results and remove duplicates
    t
    # TODO: add more effective document similarity comparison/annealing

    return b

def remove_words_by_length(doc:str, min:int=3):
    return ' '.join([w for w in doc.split(' ') if len(w) >= min])

# Ultimately what all of this is meant to accomplish is to approximate what has been 
# - mentioned most on the slides, with a weight towards headers.  Since headers aren't
# - always topical we can't rely on them alone. 
# Further slides which are left on screen for longer will be over sampled: which is 
# - what we want to identify the most likely body of text
# TODO: add a little image processing or rely on tesseract api to identify large text
# - for more precise header identification
def get_topics_from_ocr_text(corpus:list)-> dict:
    corpus_list = [re.sub(r'\n+', '\n', d).split('\n') for d in corpus if len(d)>0]

    # removelines with words specific to user interfaces, powerpoint, slides, 
    corpus_list = [[line for line in doc if not has_flagged_word(remove_non_alpha(line), 1) and not has_flagged_symbol(line)] for doc in corpus_list]
    
    # find n-grams/collocations
    all_slides = remove_words_by_length(remove_stop_words(remove_non_alpha('.'.join([' '.join(d) for d in corpus_list])))).lower()
    collocs = get_collocations(all_slides.split(' '), n_collocs=40)
    
    # compare against sentences from the top of the image (favoring topic headings and sub headings)
    topic_count_dict = {}
    for c in collocs: 
        for d in [d[0:5] for d in corpus_list]:
            if re.search(' '.join(c), ' '.join(d).lower()):
                if c in topic_count_dict:
                    topic_count_dict[c]['count'] += 1
                else:
                    topic_count_dict[c] = {'count': 1}
                    # topic_count_dict[c]['count'] = 1
    return topic_count_dict

def get_topics_over_mean(topics:dict)->dict:
    total = 0
    for key in topics:
        total += topics[key]['count']
    mean = total / len(topics)

    t50 = {}
    for key in topics:
        if topics[key]['count'] >= mean:
            t50[key] = topics[key]
    return t50

# convert wikipedia responses to sentences 
@cached
def wiki_to_sentences(article:str)->list:
    # remove anything between brackets to deal with latex
    a = re.sub(r"\{.*\}", "", article).lower() 

    # remove extra white space, non alpha characters after using punctuation to tokenize into sentences
    sentences = [re.sub('\s+', " ",remove_non_alpha(s)).strip() for s in nltk.sent_tokenize(a)]

    # remove sequences of single characters
    single_char_count = 0
    word_count = 0
    new_sentences = []
    for sentence in sentences:
        new_s = ""
        for w in sentence.split(' '):
            if len(w) == 1:
                single_char_count += 1
                if single_char_count >= 3:
                    continue
            elif len(w) == 0:
                continue
            else:
                single_char_count = 0
                word_count += 1
                new_s = ' '.join([new_s, w])

        if(word_count > 3):
            new_sentences.append(new_s.strip())
        word_count = 0
    return new_sentences

def sample_output(path, n_samples):
    with open(path, mode='r') as input:
        text = input.readlines()
        return sample(text, n_samples)

@cached
def clean_wiki_suggestions(sugg:list[str])->list[str]:
    banned = ['list of']
    def is_banned(s):
        for b in banned:
            if re.search(b, s):
                return True
        return False
    
    return [re.sub('\s+',' ', remove_non_alpha(s).lower().strip()) for s in sugg if not is_banned(s)]
    
@cached
def get_wiki_suggestions(search_terms:set, n_wiki_sugg:int=5, verbose:bool=True):
    if verbose:
        print(f'pre expanded search set {search_terms} for wikipedia')
    suggestions = set()

    for term in search_terms:
        response = wikipedia.search(term, results=n_wiki_sugg)
        suggestions.update(s for s in clean_wiki_suggestions(response))
 
    if verbose:
        print(f'\n{"-"*50}Suggestions are \n{suggestions}\n{"-"*50}')
    return suggestions - search_terms

def get_topics_from_summaries(summaries:list[str], n_collocs:int=3):
    # find most significant n-grams
    corpus = remove_words_by_length(remove_stop_words(' '.join(summaries).strip()), 4).lower()
    collocs = [remove_non_alpha(' '.join(c)).lower() for c in get_collocations(corpus.split(' '), n_collocs=n_collocs)]
    # collocs = filter_by_similarity(collocs, searched, summaries)
    topics = set(collocs)
    return topics

def get_wiki_content(search_terms:set, page:bool=False, as_list:bool=False)->list[str]:
    summaries = []
    for term in search_terms:
        article = get_wiki_article(term, page)
        if len(article) > 0:
            if as_list:
                summaries.append(article)
            else:
                summaries.append(' '.join(article))
    return summaries

# take a few ideas and blow them up into a corpus like a kid who loves dinosaurs
# 2: n_wiki_sugg 3, tangents 100
# 3: n_wiki_sugg: 10, tangents 300
# 4: n_wiki_sugg: 5, tangents 500
def sample_corpus(topics:set[str], n_samples:int, corpus:list[list[str]]=None, page:bool=False)->list[str]:
    if corpus is None:
        corpus = get_wiki_content(topics, page, as_list=True)
    if len(corpus) > 0:
        corpus = sum(corpus, [])
        n_samples = n_samples if n_samples <= len(corpus) else len(corpus)
    return sample(corpus, n_samples)

def ancillary_dictionary_dilation(topics:dict, n_topics:int=10, n_wiki_sugg:int=3, max_tangents:int=300, verbose=True):
    # TODO: move this logic somewhere else
    n_t = n_topics if n_topics <= len(topics) else len(topics)
    s = sorted(topics, key=lambda x: topics[x]['count'], reverse=True)[:n_t]
    search_terms = set([' '.join(terms).lower() for terms in s])

    wiki_sugg = get_wiki_suggestions(search_terms)
    # use doc2vec to start the wiki crawl with the apparently most relevant terms
    wiki_sugg = filter_by_similarity(wiki_sugg)
    original_wiki_sugg = wiki_sugg
    all_searched = set()
    # if we've run out of material expand to using pages instead of summaries
    use_pages = False
    # number of times we've run out of new topics
    n_dead_ends = 1
    max_dead_ends = 5
    original_n_wiki_sugg = 1
    n_wiki_sugg = original_n_wiki_sugg
    original_colloc_coef = 3
    colloc_coef = original_colloc_coef
    ramble_log = []
    
    # we want a big number of search terms in the beginning, 
    # scaling also to the number of articles returned so it doesn't peter out
    # as i increases the results coefficient increases so that non subject matter data is 
    # included but less likely to overwhelm the core subject matter
    max_tangents += 1

    for i in range(1, max_tangents):
        n_collocs = int(colloc_coef*np.log(i+1))

        # keep us anchored to the video by adding the original list of topics
        wiki_sugg = wiki_sugg.union(original_wiki_sugg)
        
        summaries =  get_wiki_content(wiki_sugg, use_pages)
        
        new_topics = get_topics_from_summaries(summaries, n_collocs)
        colloc_sugg = new_topics - all_searched

        wiki_sugg = get_wiki_suggestions(search_terms=colloc_sugg, n_wiki_sugg=n_wiki_sugg)
        
        new_topics = wiki_sugg - all_searched

        if verbose:
            print(f'\n{"-"*50} New topics \n{new_topics}\n{"-"*50}')
            
        # if len(search_terms - prev_search_terms) == 0:
        if len(new_topics) > 0: 
            all_searched.update(new_topics)
            use_pages = False
            n_dead_ends = 0
            n_wiki_sugg = original_n_wiki_sugg
            colloc_coef = original_colloc_coef
        else:
            # we're stuck!  
            print(f'\n{"-"*50} We\'re stuck!  \n n_wiki_sugg = {n_wiki_sugg} n_dead_ends = {n_dead_ends} \n, n_collocs: {n_collocs}\n{"-"*50}, colloc_coef: {colloc_coef}')
            n_dead_ends += 1
            n_wiki_sugg += 1
            colloc_coef += 1

            if n_dead_ends >= max_dead_ends:
                use_pages = True
                print('using pages')
            elif n_dead_ends == max_dead_ends + 1:
                wiki_sugg = all_searched
                print("using all searched with pages")
            elif n_dead_ends > max_dead_ends + 1:
                # let's try a sample of the current corpus
                print("taking a sample of the corpus")
                sample = [' '.join(sample_corpus(all_searched, 1000)) for i in range(3)]
                wiki_sugg = get_topics_from_summaries(sample, 10)
            elif n_dead_ends > max_dead_ends + 2:
                break

        if verbose:
            ramble_log.append((n_dead_ends, n_collocs, use_pages, colloc_coef, colloc_sugg, new_topics))

    if verbose:
        log_df = pd.DataFrame({'ramble_log': ramble_log})
        log_df.to_pickle(f'./ramble_log_{dt.now().strftime("%Y%m%d_%H%M")}.pkl')

    return all_searched    

def sanitize_filename(filename):
    return re.sub(" ", "_", remove_non_alpha(filename).strip())    

def have_wiki_article(search_term:str, page:bool=True)->bool:
    w_type = "article" if page else "summarie"
    localpath = f'./{w_type}s/{sanitize_filename(search_term)}.txt'
    return Path(localpath).is_file()


def get_wiki_article(search_term:str, page:bool=True, verbose:bool=False)->list[str]:
    w_type = "article" if page else "summarie"
    localpath = f'./{w_type}s/{sanitize_filename(search_term)}.txt'

    if verbose:
        print(f'Attempting to retrieve article for {search_term}')
    if Path(localpath).is_file():
        with open(localpath, mode='r', encoding='utf-8') as file:
            if verbose:
                print(f'Loaded {w_type} for {search_term}')
            return [re.sub('\n', ' ', s) for s in file.readlines() if len(s) > 2]
    else:
        try:
            if page:
                article = wikipedia.page(search_term).content
            else:
                article = wikipedia.summary(search_term)
            article = wiki_to_sentences(article)
            with open(localpath, mode='w', encoding='utf-8') as file:
                if verbose:
                    print(f'Saved {w_type} for {search_term}')
                for sentence in article:
                    file.write(f'\n{sentence}')
            return article
        except:
            return []


# takes a video path (such as myvid.mp4), extracts text from the video, and builds a scorer of single line 
# - sentences it thinks are related to the video 
def build_scorer(vid_path:str):
    wikipedia.set_rate_limiting(rate_limit=True, min_wait=datetime.timedelta(0, 0, 100000))
    outpath = f'{vid_path}_sentences.txt'
    
    try:
        text_df = pd.read_pickle(f'./img_to_txt_{vid_path}.pkl')
    except:
        text = get_text_from_frames(vid_path, 0)
        text_df = pd.DataFrame({'text': text})
        text_df.to_pickle(f'./img_to_txt_{vid_path}.pkl')

    ocr_topics = get_topics_from_ocr_text(text_df['text'])
    ocr_topics = get_topics_over_mean(ocr_topics)
    # ocr_topics = filter_by_similarity([remove_non_alpha(' '.join(t)).lower() for t in ocr_topics])
    search_terms = ancillary_dictionary_dilation(ocr_topics, verbose=True)
    
    log(vid_path + ': ' + ', '.join(search_terms), 'search_terms')
    with open(outpath, mode='a+', encoding='utf-8') as file:
        for term in search_terms:
            article = get_wiki_article(term)
            for sentence in article:
                file.write(f'\n{sentence}')

    print(f'sample output is {sample_output(outpath, 10)}')

# if __name__ == "__main__":
#     start_time = timeit.default_timer()
#     build_scorer('1_PCA.mp4')
#     print(f'time is {timeit.default_timer() - start_time}')
