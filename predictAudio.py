# The primary goal is to predict words within segments of video given OCR data from slides
# Secondarily I would like to establish thresholds of certainty (that a word will be spoken)
#  - which can be used to inform the weights of a speech to text system I've been working on
# for some time.
#
# This module should at the end provide the following functionality:
# 1. Take as inputs:
#       a) Video frames in series with the following characteristics
#           i.      May or may not have text
#           ii.     Text may or may not be relevant (such as interface text, etc)
#           iii.    Text may appear in different locations between different slides
#           iv.     Text may be interrupted with frames of non text
#       b)  Audio transcript segmented by time
#
# 2. Provide as output:
#       a)  Some kind of score which indicates the general predictive capability of words on slides
#           as predictors of those words being spoken within a given period of time
#           i.      With potentially relevant factors:
#                   1)  Time on screen
#                   2)  Repetition within a given span of time
#                   3)  Location on screen
#                   4)  Ratio of slides to duration of video
#       b)

import re
import numpy as np
import pandas as pd
from typing import Iterator
from pathlib import Path

import pytesseract as pt
import nltk  # pip install nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
from nltk.collocations import TrigramCollocationFinder, BigramCollocationFinder

trigram_measures = nltk.collocations.TrigramAssocMeasures()
bigram_measures = nltk.collocations.BigramAssocMeasures()

import datetime
from datetime import datetime as dt
import timeit
from random import sample

import spacy

# python -m spacy download en_core_web_lg
import en_core_web_lg

# pip install memoization
from memoization import cached

from imageToText import (
    get_text_from_frames,
    has_flagged_word,
    remove_non_alpha,
    has_flagged_symbol,
    remove_words_by_length,
    remove_stop_words,
)


def clean_ocr_text(corpus: list[list[str]]) -> list[str]:
    # TODO: need to add time stamps and ensure that line order is being preserved
    corpus_list = [re.sub(r"\n+", "\n", d).split("\n") for d in corpus if len(d) > 0]

    # removelines with words specific to user interfaces, powerpoint, slides,
    corpus_list = [
        [
            remove_words_by_length(remove_non_alpha(line))
            for line in doc
            if not has_flagged_word(remove_non_alpha(line), 1)
            and not has_flagged_symbol(line)
        ]
        for doc in corpus_list
    ]

    corpus_list = [
        [line.strip().lower() for line in doc if len(line) > 0] for doc in corpus_list
    ]
    # corpus_list = [remove_words_by_length(remove_non_alpha(d)) for d in corpus_list]

    return corpus_list


def remove_non_dict(corpus: list[list[str]], words: list[str]) -> list[list[str]]:
    # creates list of documents, with lists of lines, with lists of words on each line
    corpus_list = [[word_tokenize(line) for line in doc] for doc in corpus]
    corpus_list = [
        [[w for w in line if w in words] for line in doc] for doc in corpus_list
    ]
    return corpus_list


def get_words_from_ocr_text(corpus: list, dict_path: str) -> dict:
    corpus_list = clean_ocr_text(corpus)
    with open(f"./{dict_path}", mode="r", encoding="utf-8") as input:
        words = input.read()

    # words = remove_non_alpha(words).strip()
    words = word_tokenize(remove_stop_words(words))

    corpus_list = remove_non_dict(corpus_list, words)
    # word_list = []
    return corpus_list
    # find n-grams/collocations
    # all_slides = remove_words_by_length(remove_stop_words(remove_non_alpha('.'.join([' '.join(d) for d in corpus_list])))).lower()
    # collocs = get_collocations(all_slides.split(' '), n_collocs=40)

    # # compare against sentences from the top of the image (favoring topic headings and sub headings)
    # topic_count_dict = {}
    # for c in collocs:
    #     for d in [d[0:5] for d in corpus_list]:
    #         if re.search(' '.join(c), ' '.join(d).lower()):
    #             if c in topic_count_dict:
    #                 topic_count_dict[c]['count'] += 1
    #             else:
    #                 topic_count_dict[c] = {'count': 1}
    #                 # topic_count_dict[c]['count'] = 1
    # return topic_count_dict


def get_hotwords(vid_path: str, dict_path: str):
    text = get_text_from_frames(vid_path, 0)
    try:
        text_df = pd.read_pickle(f"./img_to_txt_{vid_path}.pkl")
    except:
        text = get_text_from_frames(vid_path, 0)
        text_df = pd.DataFrame({"text": text})
        text_df.to_pickle(f"./img_to_txt_{vid_path}.pkl")

    # TODO: enhance tesseract word recognition using generated dictionary
    ocr_words = get_words_from_ocr_text(text_df["text"], dict_path)

    return ocr_words


if __name__ == "__main__":
    start_time = timeit.default_timer()
    get_hotwords(vid_path="1_PCA.mp4", dict_path="1_PCA.mp4_sentences.txt")
    print(f"time is {timeit.default_timer() - start_time}")
