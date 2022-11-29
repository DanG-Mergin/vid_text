import os
import sys
import wave
import json

from vosk import Model, KaldiRecognizer, SetLogLevel
# !pip install vosk
import Word as custom_Word

SetLogLevel(0)

 # path to vosk model downloaded from
# https://alphacephei.com/vosk/models
model_path = "./vosk_models/vosk-model-en-us-0.42-gigaspeech"

if not os.path.exists(model_path):
    print(f"Please download the model from https://alphacephei.com/vosk/models and unpack as {model_path}")
    sys.exit()

print(f"Reading your vosk model '{model_path}'...")
model = Model(model_path)
print(f"'{model_path}' model was successfully read")

# Reading your vosk model '../models/vosk-model-en-us-0.21'...
# '../models/vosk-model-en-us-0.21' model was successfully read

# name of the audio file to recognize
# audio_filename = "../audio/speech_recognition_systems.wav"

audio_filename = "./vid_audio.wav"

# audio_filename = "./clips/testwav/0_60.wav"

# name of the text file to write recognized text
text_filename = "vosk_with_timestamps.txt"  

if not os.path.exists(audio_filename):
    print(f"File '{audio_filename}' doesn't exist")
    sys.exit()

print(f"Reading your file '{audio_filename}'...")
wf = wave.open(audio_filename, "rb")
print(f"'{audio_filename}' file was successfully read")  

# check if audio is mono wav
# if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
#     print("Audio file must be WAV format mono PCM.")
#     sys.exit()  

rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)  

results = []
# recognize speech using vosk model
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        part_result = json.loads(rec.Result())
        results.append(part_result)

part_result = json.loads(rec.FinalResult())
results.append(part_result) 

list_of_words = []
for sentence in results:
    if len(sentence) == 1:
        # sometimes there are bugs in recognition 
        # and it returns an empty dictionary
        # {'text': ''}
        continue
    for obj in sentence['result']:
        w = custom_Word.Word(obj)  # create custom Word object
        list_of_words.append(w)  # and add it to list  

    for word in list_of_words:
        print(word.to_string())  

text = ''
for r in results:
    text += r['text'] + ' '

print("\tVosk thinks you said:\n")
print(text)  

print(f"Saving text to '{text_filename}'...")
with open(text_filename, "w") as text_file:
    text_file.write(text)
print(f"Text successfully saved")  