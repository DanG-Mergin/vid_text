import speech_recognition as sr 
import moviepy.editor as mp
from moviepy.video.tools.segmenting import findObjects
# import cv2 

import re
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# TODO: move to requirements file
# pip install SpeechRecognition moviepy

from pathlib import Path, PurePath

import numpy as np
import pandas as pd

import datetime
from typing import Iterator

# from ImageToText import TextRecognizer
import pytesseract as pt
import nltk # pip install nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')
from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder
trigram_measures = nltk.collocations.TrigramAssocMeasures()
bigram_measures = nltk.collocations.BigramAssocMeasures()

import wikipedia #!pip install wikipedia

import timeit

# import asyncio

# TODO: inherit from moviepy

# ingest video
class Video:
    path = ""
    name = "" # file name without extension
    clip_ext = "mp4"
    clip_dir = "clips"
    clip = None
    duration = 0 # TODO: inherit from moviepy
    default_fps = 24

    def __init__(self, path):
        self.path = path
        self.clip = mp.VideoFileClip(path)
        # TODO: inherit from moviepy
        self.duration = self.clip.duration
        self.__parse_path(path)

    def __parse_path(self, path):
        p = PurePath(path)
        self.name = p.stem

    def get_clip(self, path = path):
        # TODO: evaluate impact of keeping clip object in memory.. it isn't being used here
        # - if persisting then this should absorb get_subclip
        path = path if len(path) > 0 else self.path
        # TODO: test using "with video clip as clip here to free up resources"
        # https://zulko.github.io/moviepy/getting_started/efficient_moviepy.html 
        return mp.VideoFileClip(path)

    # TODO: add option for working directory
    def __get_subclip_path(self, start, end, ext):
        ext = ext if ext is not None else self.clip_ext 
        return Path(self.__get_subclip_dir_path(ext)).joinpath(f'{start}_{end}.{ext}')
    
    def __get_subclip_dir_path(self, subfold='_clips'):
        return Path(Path(__file__).parent).joinpath(self.clip_dir, f'{self.name}{subfold}')

    def __add_subclip(self, start, end):
        # TODO: move this outside of any loops if possible
        # create the subclips folder if it doesn't exist
        Path(f'{self.__get_subclip_dir_path()}').mkdir(parents=True, exist_ok=False)
        subclip = self.clip.subclip(start, end)
        subclip.write_videofile(f'{self.__get_subclip_path(start, end)}')

        return subclip

    def __add_audio_subclip(self, start, end, ext):
        subclip = mp.AudioFileClip(self.path).subclip(start, end)
        # ffmpeg params ATOW are for mono audio
        subclip.write_audiofile(f'{self.__get_subclip_path(start, end, ext)}', codec='pcm_s16le', ffmpeg_params=["-ac", "1"])

        return subclip

    def get_subclip(self, start, end, from_file:bool=False):
        # TODO: may be problematic on linux.. need to run in VM 
        if not from_file:
            return self.clip.subclip(start, end)

        else:
            subclip_path = self.__get_subclip_path(start, end)
            if subclip_path.is_file():
                print(f'-- Found {subclip_path}')
                return (mp.VideoFileClip(f'{subclip_path}'), subclip_path)  
            else: 
                print(f'-- Did not find {subclip_path}')
                self.__add_subclip(start, end)
                return subclip_path

            
            
            # return self.__add_subclip(start, end)

    def get_audio_subclip(self, start, end, ext="wav"):
        if end > self.duration:
            end = self.duration

        start = int(start)
        end = int(end)

        subclip_path = self.__get_subclip_path(start, end, ext)
        if not subclip_path.is_file():
            if not subclip_path.parent.exists():
                Path(f'{self.__get_subclip_dir_path(ext)}').mkdir(parents=True, exist_ok=False)
            print(f'-- Did not find {subclip_path}')
            self.__add_audio_subclip(start, end, ext)
            # return self.get_audio_subclip(start, end)
        else:
            print(f'-- Found {subclip_path}')
        return subclip_path

    # TODO: test cv2 to extract images from video.  Will be faster if subclips aren't needed
    # pass in the fpm you want, not the fpm of the video
    def get_frames(self, start, end, fpm=3)-> Iterator[tuple]: 
        return self.get_subclip(start, end).iter_frames(fps=fpm/60, with_times=True, dtype="uint8")

    # TODO: extend moviepy
    # @staticmethod
    # def save_frame(vid, start, end):
        

class Transcript:
    text = ""
    raw_t_path = "raw_transcript"
    audio_path = "vid_audio.wav"

    def __init__(self, vid_path):
        self.vid = Video(vid_path)


    # split audio into multiple files for api limits or threading
    def __split_audio(self, start, end, vid, clip_dur):
        durations = np.arange(start, int(end), clip_dur)

        paths = []
        for d in durations:
            paths.append(vid.get_audio_subclip(d, d + clip_dur))
            # TODO: add asyncio. Beware flask's gotchas as it isn't native async unless you use a specific version

        return paths

    def split_audio(self, start, end, vid, clip_dur=60):
        vid = vid if vid is not None else self.vid
        audio_paths = self.__split_audio(start, end, vid, clip_dur)
        return audio_paths

    def __transcribe_coqui(self, file_path):
        print('-'*50)
        print('undefined')

    def __transcribe_google(self, file_path):
        recognizer = sr.Recognizer()
        audio = sr.AudioFile(f'{file_path}')

        with audio as source:
            audio_file = recognizer.record(source)
            try:
                r = recognizer.recognize_google(audio_file)
            except:
                # TODO: this isn't remotely robust
                # need to track whitespace
                r = ""
            return r


    def transcribe(self, start, end, clip_dur=60, provider="google"):
        start = 0 if start is None else start
        end_ = self.vid.duration if end is None else end
        # vid = vid if vid is not None else self.vid

        t = ""
        transcripts = [] 
        dead_air = []
        audio_paths = self.__split_audio(start, end_, self.vid, clip_dur)

        for p in audio_paths:
            if provider == "google":
                res = self.__transcribe_google(p)
            elif provider == "coqui":
                res = self.__transcribe_coqui(p)
            if res == "":
                dead_air.append(p)

            transcripts.append(res)
            t = t + res

        return (t, transcripts, audio_paths, dead_air)

    # regex is a compiled re
    def save(self, path, text, regex:re=None):
        with open(f'{path}.txt', mode='w', encoding='utf-8') as file:
            if regex is not None:
                text = regex.sub('', text)
            file.write(text)
            print(f'saved transript file as {path}.txt')

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
    # vid.iter_frames gets image arrays
    # vid.get_frame returns np array at time t
    # save_frame saves a clip to image file at time t
    # write_images_sequence: writes clip to series of image files
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

def get_only_english(document):
    regex = re.compile(r'[^a-zA-Z]')
    return ' '.join(w for w in nltk.word_tokenize(regex.sub(' ', document)))

def get_hotwords(corpus):
    # remove non-alpha
    # cleaned = [remove_non_alpha(d) for d in corpus]
    # remove file extensions
    # cleaned = [remove_file_extensions(d) for d in cleaned]
    # remove words like powerpoint/slide

    cleaned = [get_only_english(d) for d in corpus if len(d) > 0]
    return cleaned

    # remove names???  
    # remove stop words as they should already be accounted for
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
def build_coqui_scorer(vid_path:str):
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

# manually run data ingest for coqui model training
# this will take a path to a video file and return transcribed 
# - files of the supplied/default clip length
def prep_coqui(vid_path, clip_dur:int=15):
    regex = re.compile(r'[^a-zA-Z\s]')
    transcript = Transcript(vid_path)
    transcript_tup = transcript.transcribe(0, end=None, clip_dur=clip_dur)
    transcript.save('1_PCA_total', transcript_tup[0])

    if len(transcript_tup[3]) > 0:
        with open('log.txt', mode='a+') as file:
            file.write(f'\n{"-"*50}\n{datetime.datetime.now()} ~~ failed remote transcription')
            for e in transcript_tup[3]:
                file.write(f'\n\t{datetime.datetime.now()} ~~ Response: {e} ~~ ')

    for i,p in enumerate(transcript_tup[2]):
        transcript.save(p, transcript_tup[1][i], regex)

    
# run after manual labeling for coqui training
def build_coqui_df(start, end, clip_dur, name):
    realpha = re.compile(r'[^a-zA-Z\s]')
    rewhtspc = re.compile(r'\s+')

    wav_filename = []
    wav_filesize = []
    transcript = []
    durations = np.arange(start, int(end), clip_dur)

    parent_path = Path(Path(__file__).parent).joinpath('clips', name)
    for d in durations:
        name = f'{d}_{d+clip_dur}.wav.txt'
        p = Path(parent_path).joinpath(name)
        # get video file size in bytes
        vid_p = Path(parent_path).joinpath(name.replace('.txt', ''))
        n_bytes = Path(vid_p).stat().st_size

        with open(p, mode='r') as input:
            text = input.read()
        with open(p, mode='w') as output:
            text = realpha.sub('', text).lower()
            text = rewhtspc.sub(' ', text)
            output.write(text)

        wav_filename.append(name)
        wav_filesize.append(n_bytes)
        transcript.append(text)
    
    df = pd.DataFrame({'wav_filename': wav_filename, 'wav_filesize': wav_filesize, 'transcript': transcript})
    print(df.head())

    df.to_csv(f'coqui_train_PCA_1_{start}_{end}', sep=',', encoding='utf-8')
    # there is one extra comma in the header... hacky fix

    with open(f'coqui_train_PCA_1_{start}_{end}', mode='r') as input:
        text = input.read()
    
    return df
        
        
# _________________________________________________________
# 
# to get a dataframe for coqui training
# prep_coqui('1_PCA.mp4')
# cq_df = build_coqui_df(0, 1380, 15, '1_PCAwav')
# 
# to get an array of frames from a video
# bleh = get_text_from_frames('1_PCA.mp4', 0, 120)
# pd.DataFrame({'bleh': bleh}).to_pickle('./img_to_txt.pkl')
# 
# To get sentences for scorer generation
# t = get_topics(bleh['bleh'])
# wikis = get_wiki_articles(t)
# 

if __name__ == "__main__":
    start_time = timeit.default_timer()
    # prep_coqui('1_PCA.mp4')
    # cq_df = build_coqui_df(0, 1380, 15, '1_PCAwav')
    
    # bleh = get_text_from_frames('1_PCA.mp4',0)
    # pd.DataFrame({'bleh': bleh}).to_pickle('./img_to_txt_PCA_full.pkl')

    # bleh = pd.read_pickle('./img_to_txt_PCA_full.pkl')
    # print(bleh['bleh'])
    # print(get_hotwords(bleh['bleh']))
    # print(get_topics(bleh['bleh']))
    # t = get_topics(bleh['bleh'])
    # wikis = get_wiki_articles(t)
    build_coqui_scorer('1_PCA.mp4')
    # print(bleh)

    # v = Video('1_PCA.mp4')
    # v.get_audio_subclip(0, 99999999999999)
    # v.get_clip()
    
    # t = Transcript(v.get_clip())
    # t = Transcript('section1.mp4')

    # t = Transcript('1_PCA.mp4')
    # t.save_transcript("raw_transcript")
    print(f'time is {timeit.default_timer() - start_time}')
