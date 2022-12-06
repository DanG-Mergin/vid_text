"""

- 1. Recieve request for words by time range (when word probabilities in STT are low)
--- a. batch or one off
--- b. word processing (elsewhere) will handle stop-words and stemming, etc

- 2. Select sampling method (for frames to look for text in)
--- a. processing the entire collection for text is infeasible
--- b. if first and last frame of time period are different will need to sample more and return specific times

- 3. Return words and probabilities (if accuracy is in question)  
--- a. separated by time period

{
    start, 
    end,
    frame-extracts: [frame-extract]
}
frame extract: 
{
    observed-start,
    observed-end: only if end is actually known,
    words: [word]
}
word: 
{

}
"""

"""
Note: additional processing using 
"""


from PIL import Image
# from pytesseract import pytesseract

import re

# TODO: move to requirements file
# sudo apt install tesseract-ocr
# pip install opencv-contrib-python pytesseract
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import pytesseract 
# import cv2

#Define path to tessaract.exe

#Define path to image
# path_to_image = 'images/sampletext1-ocr.png'
#Point tessaract_cmd to tessaract.exe

# pt.tesseract_cmd = path_to_tesseract
#Open image with PIL
# img = Image.open(path_to_image)
#Extract text from image

# text = .image_to_string(img)
# print(text)

class TextRecognizer(pytesseract):
    
    def __init__(self, path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract'):
        super.tesseract_cmd = path_to_tesseract

    # get_text_from_image
    

    