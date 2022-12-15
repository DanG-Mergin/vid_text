import numpy as np
import speech_recognition as sr 
import re
import timeit
import datetime
from pathlib import Path
import pandas as pd

# local modules
from imageToText import build_scorer
from video import Video

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

# Run to get transcripts you can manually fix
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

    # with open(f'coqui_train_PCA_1_{start}_{end}', mode='r') as input:
    #     text = input.read()
    
    return df
        
        
# _________________________________________________________
# 
# to get a dataframe for coqui training/transcripts to fix
# prep_coqui('1_PCA.mp4')
# 
# after manually fixing transcripts
# cq_df = build_coqui_df(0, 1380, 15, '1_PCAwav')
# 
# then run build_scorer('<vid path>') for a scorer file

# if __name__ == "__main__":
#     start_time = timeit.default_timer()

#     build_scorer('1_PCA.mp4')

#     print(f'time is {timeit.default_timer() - start_time}')
