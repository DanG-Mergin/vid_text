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
from random import sample

# TODO: clean this circular mess up!
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

# def get_tags(corpus):
    
# limiter allows us to use annealing and avoid oversampling
def get_collocations(corpus:list, n:int=40, collocation_results_coef:float=0.01)->list[(str,str)]:
    # n = int(collocation_results_coef*len(corpus))
    if type(corpus) == str:
        corpus = corpus.split(' ')

    # tri_finder = TrigramCollocationFinder.from_words(corpus)
    # tri_finder.apply_freq_filter(3)
    # t = tri_finder.nbest(trigram_measures.pmi, 20)

    bi_finder = BigramCollocationFinder.from_words(corpus)
    # scored = bi_finder.score_ngrams(bigram_measures.raw_freq)
    # scored = sorted(bigram for bigram, score in scored)

    # tagged = nltk.pos_tag(corpus)
    # bi_finder.apply_freq_filter(2)
    # b = bi_finder.nbest(bigram_measures.pmi, 20)

    # tri_finder = TrigramCollocationFinder.from_words(corpus)
    # tri_finder.apply_freq_filter(3)
    # t = tri_finder.nbest(trigram_measures.likelihood_ratio, 20)

    # bi_finder = BigramCollocationFinder.from_words(corpus)
    # bi_finder.apply_freq_filter(2)
    b = bi_finder.nbest(bigram_measures.likelihood_ratio, n)

    # t3 = tri_finder.nbest(trigram_measures.poisson_stirling, 20)
    # b3 = bi_finder.nbest(bigram_measures.poisson_stirling, 20)

    # TODO: stem results and remove duplicates

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
def get_topics(corpus:list)-> dict:
    corpus_list = [re.sub(r'\n+', '\n', d).split('\n') for d in corpus if len(d)>0]

    # removelines with words specific to user interfaces, powerpoint, slides, 
    corpus_list = [[line for line in doc if not has_flagged_word(remove_non_alpha(line), 1) and not has_flagged_symbol(line)] for doc in corpus_list]
    
    # find n-grams/collocations
    all_slides = remove_words_by_length(remove_stop_words(remove_non_alpha('.'.join([' '.join(d) for d in corpus_list])))).lower()
    collocs = get_collocations(all_slides.split(' '))

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

# n_topics is how many of the top counted topics to use
# n_search is how many of those topics to expand search on
# n_expand is how much that search should expand to related items
def get_wiki_articles(topics:dict, n_topics:int=10, n_search:int=10, n_expand=3)->Iterator:
    # throttling so we aren't blocked
    wikipedia.set_rate_limiting(rate_limit=True, min_wait=datetime.timedelta(0, 0, 1000000))
    n_t = n_topics if n_topics <= len(topics) else len(topics.keys)
    s = sorted(topics, key=lambda x: topics[x]['count'], reverse=True)[:n_t]
    search_terms = [' '.join(terms).lower() for terms in s]

    # create a set of related ideas according to wikipedia
    # wikipedia.search('matrix dimensions', results = 5, suggestion = True) 
    # yields ['Confusion matrix', 'Rotation matrix', 'Matrix norm', 'Covariance matrix', 'Sparse matrix'] 
    search_set = set(search_terms)
    print(f'pre expanded search set {search_set} for wikipedia')
    for term in search_terms[0:n_search]:
        search_set.update([x.lower() for x in wikipedia.search(term, results=n_expand)])
    
    # make sure we have at least one result for the terms we arent expanding search on
    for term in search_terms[n_search:]:
        search_set.update([x.lower() for x in wikipedia.search(term, results=1)])

    print(f'Expanded search set {search_set} for wikipedia')
    for t in search_set:
        try:
            response = wikipedia.page(t)
            # yield response.content
            yield response
        except:
            print(f'no wiki for {t}')
            # yield ''
            yield {'content': '', 'summary':''}

# convert wikipedia responses to sentences 
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

# def explode_wiki_search(wikis, n_expansions:int=10):
#     search_terms = []
#     corpus = ""
#     # extract summaries
#     for w in wikis:
#         if type(w) == wikipedia.wikipedia.WikipediaPage:
#             corpus = corpus + ' ' + ' '.join(wiki_to_sentences(w.summary))

#     # find most significant n-grams
#     corpus = remove_words_by_length(remove_stop_words(corpus.strip()), 4).lower()
#     collocs = get_collocations(corpus.split(' '))
#     # use as search terms
#     print('hi')
#     # repeat until we have sufficient results

def get_wiki_suggestions(search_terms:set, n_expand=10, verbose:bool=False):
    # don't search the same thing multiple times
    # search_set = search_set - searched_set
    if verbose:
        print(f'pre expanded search set {search_terms} for wikipedia')
    suggestions = set()

    for term in search_terms:
        result = wikipedia.search(term, results=n_expand)
        suggestions.update(x.lower() for x in result)
    
    return suggestions - search_terms

def get_topics_from_summaries(summaries:list[str], collocation_results_coef):
    # find most significant n-grams
    corpus = remove_words_by_length(remove_stop_words(' '.join(summaries).strip()), 4).lower()
    collocs = [' '.join(c).lower() for c in get_collocations(corpus.split(' '))]
    topics = set(collocs)
    return topics

def get_wiki_content(search_terms:set, page:bool=False):
    summaries = []
    for term in search_terms:
        article = get_wiki_article(term, page)
        if len(article) > 0:
            summaries.append(' '.join(article))
        # if use_pages:
        #     summaries.append(wikipedia.page(term).content)
        # else:
        #     summaries.append(wikipedia.summary(term))
    return summaries

# take a few ideas and blow them up into a corpus like a kid who loves dinosaurs
def ramble_excitedly(topics:dict, n_topics:int=10, n_expand:int=3, max_tangents:int=100, verbose=False):
    # TODO: move this logic somewhere else
    n_t = n_topics if n_topics <= len(topics) else len(topics.keys)
    s = sorted(topics, key=lambda x: topics[x]['count'], reverse=True)[:n_t]
    search_terms = set([' '.join(terms).lower() for terms in s])

    search_terms = get_wiki_suggestions(search_terms)
    searched = set()
    # if we've run out of material expand to using pages instead of summaries
    use_pages = False
    min_search_terms = 3
    # collocation_results_coef = 0.01
    # number of times we've run out of new topics
    n_dead_ends = 0

    # we want a big number of search terms in the beginning, 
    # scaling also to the number of articles returned so it doesn't peter out
    # as i increases the results coefficient decreases
    # TODO: it would be better to use a true calculation of relevance than these assumptions
    max_tangents += 1
    for i in range(1, max_tangents):
        n_collocs = int(max_tangents / i*1.5) if int(max_tangents / i*1.5) > min_search_terms else min_search_terms

        summaries = get_wiki_content(search_terms, use_pages)

        searched.update(search_terms)
        new_topics = get_topics_from_summaries(summaries, n_collocs)
        nlp_search_terms = new_topics - searched
        search_terms = get_wiki_suggestions(search_terms=nlp_search_terms, n_expand=n_expand)
        if verbose:
            print(f'Expanded search set {search_terms} for wikipedia')
        if len(search_terms) < min_search_terms:
            use_pages = True
            if verbose:
                print(f'Using pages for expanded wikipedia search at {i} iterations')
            if len(search_terms) == 0:
                # reprocess
                search_terms = searched
                # increase the number of results we want from wikipedia
                n_expand += 1
                n_dead_ends += 1
                # collocation_results_coef /= n_dead_ends
                if verbose:    
                    print(f'Ran out of steam at {i} iterations. Reprocessing with wikipedia suggestions at {n_expand}')
        # TODO: check for document similarity

    return search_terms    

def sanitize_filename(filename):
    return re.sub(" ", "_", remove_non_alpha(filename).strip())    
        
    # expand topics from summaries n-times
def get_wiki_article(search_term:str, page:bool=True, verbose:bool=False)->list[str]:
    localpath = f'./{"articles" if page else "summaries"}/{sanitize_filename(search_term)}.txt'
    # filename = sanitize_filename(search_term)
    if page:
        if verbose:
            print(f'Attempting to retrieve article for {search_term}')
        # path = Path(f'/articles/{filename}.txt')
        if Path(localpath).is_file():
            with open(localpath, mode='r', encoding='utf-8') as file:
                if verbose:
                    print(f'Loaded article for {search_term}')
                return file.read()
        else:
            try:
                article = wikipedia.page(search_term).content
                article = ' '.join(wiki_to_sentences(article))
                with open(localpath, mode='w', encoding='utf-8') as file:
                    if verbose:
                        print(f'Saved article for {search_term}')
                    file.write(article)
                return article
            except:
                return ''
    else:
        if verbose:
            print(f'Attempting to retrieve summary for {search_term}')
        # path = Path(f'/summaries/{filename}.txt')
        if Path(localpath).is_file():
            with open(localpath, mode='r', encoding='utf-8') as file:
                if verbose:
                    print(f'Loaded summary for {search_term}')
                return file.read()
        try:
            summary = wikipedia.summary(search_term)
            summary = ' '.join(wiki_to_sentences(summary))
            with open(localpath, mode='w', encoding='utf-8') as file:
                file.write(summary)
            if verbose:    
                print(f'Saved summary for {search_term}')
            return summary
        except:
            if verbose:
                print(f'Failed to retrieve summary for {search_term}')
            return ''

# def get_wiki_articles(search_terms)->Iterator:
#     for t in search_:
#         try:
#             response = wikipedia.page(t)
#             # yield response.content
#             yield response
#         except:
#             print(f'no wiki for {t}')
#             # yield ''
#             yield {'content': '', 'summary':''}

# takes a video path (such as myvid.mp4), extracts text from the video, and builds a scorer of single line 
# - sentences it thinks are related to the video 
def build_scorer(vid_path:str):
    wikipedia.set_rate_limiting(rate_limit=True, min_wait=datetime.timedelta(0, 0, 1000000))
    outpath = f'{vid_path}_sentences.txt'
    
    try:
        text_df = pd.read_pickle(f'./img_to_txt_{vid_path}.pkl')
    except:
        text = get_text_from_frames(vid_path, 0)
        text_df = pd.DataFrame({'text': text})
        text_df.to_pickle(f'./img_to_txt_{vid_path}.pkl')

    topics = get_topics(text_df['text'])
    # # t = get_topics_over_mean(t)
    # try:
    #     wikis = pd.read_pickle(f'./wiki_df.pkl')['wikis']
    # except:
    #     wikis = get_wiki_articles(t)
    #      # TODO: delete me
    #     wiki_df = pd.DataFrame({'wikis': wikis})
    #     wiki_df.to_pickle('wiki_df.pkl')
    
    # print(type(wikis))
    # explode_wiki_search(wikis)
    search_terms = ramble_excitedly(topics)
    log(vid_path + ': ' + ', '.join(search_terms), 'search_terms')
    with open(outpath, mode='a+', encoding='utf-8') as file:
        for term in search_terms:
            article = get_wiki_article(term)
            for sentence in article:
                file.write(f'\n{sentence}')

    # for a in wikis:
    #     if type(a) == wikipedia.wikipedia.WikipediaPage:
    #     # if len(a['content']) > 0: 
    #         with open(outpath, mode='a+', encoding='utf-8') as file:
    #             sentences = wiki_to_sentences(a.content) 
    #             for s in sentences:
    #                 file.write(f'\n{s}')
    #         # with open('categories.txt', mode='a+', encoding='utf-8') as file:
    #         #     # sentences = wiki_to_sentences(a.categories) 
    #         #     file.write(f'\n{"-"*50}\n\n')
    #         #     for s in a.categories:
    #         #         file.write(f'\n{s}')
    #         # with open('sections.txt', mode='a+', encoding='utf-8') as file:
    #         #     file.write(f'\n{"-"*50}\n\n')
    #         #     # sentences = wiki_to_sentences(a.sections) 
    #         #     for s in a.sections:
    #         #         file.write(f'\n{s}')
    #         # with open('summaries.txt', mode='a+', encoding='utf-8') as file:
    #         #     file.write(f'\n{"-"*50}\n\n')
    #         #     sentences = wiki_to_sentences(a.summary) 
    #         #     for s in sentences:
    #         #         file.write(f'\n{s}')


    print(f'sample output is {sample_output(outpath, 10)}')