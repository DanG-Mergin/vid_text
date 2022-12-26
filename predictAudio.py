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


def clean_ocr_text(corpus: list[(float, list[str])]) -> list[(float, list[str])]:
    corpus_list = [
        (frame[0], re.sub(r"\n+", "\n", frame[1]).split("\n"))
        for frame in corpus
        if len(frame[1]) > 0
    ]

    corpus_list = [
        (
            frame[0],
            [
                remove_words_by_length(remove_non_alpha(line))
                for line in frame[1]
                if not has_flagged_word(remove_non_alpha(line), 1)
                and not has_flagged_symbol(line)
            ],
        )
        for frame in corpus_list
    ]

    corpus_list = [
        (frame[0], [line.strip().lower() for line in frame[1] if len(line) > 0])
        for frame in corpus_list
    ]

    return corpus_list


def remove_non_dict(
    corpus: list[(float, list[str])], words: list[str]
) -> list[(float, list[str])]:
    # creates list of documents, with lists of lines, with lists of words on each line
    corpus_list = [
        (frame[0], [word_tokenize(line) for line in frame[1]]) for frame in corpus
    ]
    corpus_list = [
        (frame[0], [[w for w in line if w in words] for line in frame[1]])
        for frame in corpus_list
    ]
    return corpus_list


def get_words_from_ocr_text(
    corpus: list[(float, list[str])], dict_path: str
) -> list[(float, list[str])]:
    corpus_list = clean_ocr_text(corpus)
    try:
        with open(f"./{dict_path}", mode="r", encoding="utf-8") as input:
            words = input.read()
            assert (
                len(words) > 0
            ), f'\n{"-"*50}\nWARNING!\nA text file with all words generated for the video must be provided.\nNo hotwords will be used to improve transcription \n{"-"*50}\n'
            words = word_tokenize(remove_stop_words(words))
            corpus_list = remove_non_dict(corpus_list, words)
    except AssertionError as e:
        print(e)
        # TODO: log and look more into exceptions
        # TODO: consider exiting program
    return corpus_list


def compress_to_time_range(
    corpus: list[(float, list[str])]
) -> tuple[list[float], list[list[str]]]:
    # half_open ranges for binary search representing the start time of slides
    slide_start_times = []
    slide_text = []
    start = 0
    for i, frame in enumerate(corpus):
        if i + 1 < len(corpus):
            if frame[1] == corpus[i + 1][1]:
                continue
            else:
                slide_start_times.append(start)
                slide_text.append(frame[1])
                start = corpus[i + 1][0]
        else:
            slide_start_times.append(start)
            slide_text.append(frame[1])

    return (slide_start_times, slide_text)


def get_hotwords_by_time(
    start: float, end: float, slides_by_time: tuple[list[float], list[list[str]]]
):
    # note: last index returned will be outside of the range even if end = value at last index
    # - however I'm keeping them for now as the resolution is set at 15 seconds
    indices = np.searchsorted(slides_by_time[0], [start, end], side="left")
    slides = slides_by_time[1][indices[0] : indices[-1]]
    return (indices, slides)


def get_hotwords(vid_path: str, dict_path: str):
    try:
        text_df = pd.read_pickle(f"./img_to_txt_{vid_path}.pkl")
    except:
        text = get_text_from_frames(vid_path, 0, fpm=4)
        text_df = pd.DataFrame({"text": text})
        text_df.to_pickle(f"./img_to_txt_{vid_path}.pkl")

    # ocr_words = get_words_from_ocr_text(text_df["text"], dict_path)
    # slides_by_time = compress_to_time_range(ocr_words)
    # ocr_words = get_words_from_ocr_text(text_df["text"], dict_path)

    # TODO: move this process to the beginning for efficiency
    try:
        slides_by_time_df = pd.read_pickle(f"./slides_by_time_{vid_path}.pkl")
        slides_by_time = (slides_by_time_df["time"], slides_by_time_df["slides"])
    except:
        ocr_words = get_words_from_ocr_text(text_df["text"], dict_path)
        slides_by_time = compress_to_time_range(ocr_words)
        slides_by_time_df = pd.DataFrame(
            {"time": slides_by_time[0], "slides": slides_by_time[1]}
        )
        slides_by_time_df.to_pickle(f"./slides_by_time_{vid_path}.pkl")

    return slides_by_time


if __name__ == "__main__":
    start_time = timeit.default_timer()
    slides_by_time = get_hotwords(
        vid_path="1_PCA.mp4", dict_path="1_PCA.mp4_sentences.txt"
    )
    hw = get_hotwords_by_time(0, 160, slides_by_time)
    print(f"time is {timeit.default_timer() - start_time}")
