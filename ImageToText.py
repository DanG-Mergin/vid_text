import re
import numpy as np
import pandas as pd
from typing import Iterator

import pytesseract as pt
import nltk # pip install nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')
from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder
trigram_measures = nltk.collocations.TrigramAssocMeasures()
bigram_measures = nltk.collocations.BigramAssocMeasures()

import wikipedia #!pip install wikipedia
import datetime

# TODO: clean this circular mess up!
from video import Video

# import timeit


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

def get_collocations(corpus):
    # tri_finder = TrigramCollocationFinder.from_words(corpus)
    # tri_finder.apply_freq_filter(3)
    # t = tri_finder.nbest(trigram_measures.pmi, 20)

    bi_finder = BigramCollocationFinder.from_words(corpus)
    # bi_finder.apply_freq_filter(2)
    # b = bi_finder.nbest(bigram_measures.pmi, 20)

    # tri_finder = TrigramCollocationFinder.from_words(corpus)
    # tri_finder.apply_freq_filter(3)
    # t = tri_finder.nbest(trigram_measures.likelihood_ratio, 20)

    # bi_finder = BigramCollocationFinder.from_words(corpus)
    # bi_finder.apply_freq_filter(2)
    b = bi_finder.nbest(bigram_measures.likelihood_ratio, 20)

    # t3 = tri_finder.nbest(trigram_measures.poisson_stirling, 20)
    # b3 = bi_finder.nbest(bigram_measures.poisson_stirling, 20)

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
            yield response.content
        except:
            print(f'no wiki for {t}')
            yield ''

# convert wikipedia responses to sentences 
def to_sentences(article:str)->list:
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

# takes a video path (such as myvid.mp4), extracts text from the video, and builds a scorer of single line 
# - sentences it thinks are related to the video 
def build_scorer(vid_path:str):
    try:
        text_df = pd.read_pickle(f'./img_to_txt_{vid_path}.pkl')
    except:
        text = get_text_from_frames(vid_path, 0)
        text_df = pd.DataFrame({'text': text})
        text_df.to_pickle(f'./img_to_txt_{vid_path}.pkl')

    t = get_topics(text_df['text'])
    wikis = get_wiki_articles(t)

    for a in wikis:
        if len(a) > 0: 
            with open(f'{vid_path}_sentences.txt', mode='a+', encoding='utf-8') as file:
                sentences = to_sentences(a) 
                for s in sentences:
                    file.write(f'\n{s}')