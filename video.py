import speech_recognition as sr 
import moviepy.editor as mp
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from pathlib import Path, PurePath, PurePosixPath
import numpy as np

# import asyncio

# TODO: inherit from moviepy

# ingest video
class Video:
    path = ""
    # clips_dir = "video_clips"
    # clips_dir = ""
    name = "" # file name without extension
    orig_extension = ""
    clip_ext = "mp4"
    clip_dir = "clips"
    clip = None
    duration = 0 # TODO: inherit from moviepy

    def __init__(self, path):
        self.path = path
        self.clip = mp.VideoFileClip(path)
        # TODO: inherit from moviepy
        self.duration = self.clip.duration
        self.__parse_path(path)

    def __parse_path(self, path):
        p = PurePath(path)
        self.name = p.stem
        self.orig_extension = p.suffix

    def get_clip(self, path = path):
        # TODO: evaluate impact of keeping clip object in memory.. it isn't being used here
        # - if persisting then this should absorb get_subclip
        path = path if len(path) > 0 else self.path
        return mp.VideoFileClip(path)

    # TODO: add option for working directory
    def __get_subclip_path(self, start, end, ext):
        ext = ext if ext is not None else self.clip_ext 
        return Path(self.__get_subclip_dir_path(ext)).joinpath(f'{start}_{end}.{ext}')
    
        # return Path(Path(__file__).parent).joinpath(self.clip_dir, f'{self.name}_clips', f'{start}_{end}.{self.clip_ext}')
    
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
        # Path(f'{self.__get_subclip_dir_path(ext)}').mkdir(parents=True, exist_ok=False)
        
        subclip = mp.AudioFileClip(self.path, nchannels=1).subclip(start, end)
        # ffmpeg params ATOW are for mono audio
        subclip.write_audiofile(f'{self.__get_subclip_path(start, end, "wav")}', codec='pcm_s32le', ffmpeg_params=["-ac", "1"])

        return subclip

    def get_subclip(self, start, end):
        # TODO: may be problematic on linux.. need to run in VM 
        subclip_path = self.__get_subclip_path(start, end)

        if subclip_path.is_file():
            print(f'-- Found {subclip_path}')
            return (mp.VideoFileClip(f'{subclip_path}'), subclip_path)  
        else: 
            print(f'-- Did not find {subclip_path}')
            self.__add_subclip(start, end)
            return self.get_subclip(start, end)
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

class Transcript:
    text = ""
    raw_t_path = "raw_transcript"
    audio_path = "vid_audio.wav",
    provider = ""
    
    def __init__(self, vid_path, provider="google"):
        self.vid = Video(vid_path)
        # self.text = self.transcribe(self.extract_audio(self.vid))
        self.provider = provider
        if provider == "google":

            # split audio up into files below the api threshold
            self.text = self.__split_audio(0, self.vid.duration)

    # split audio into multiple files for api limits or threading
    def __split_audio(self, start, end, clip_dur=60):
        # TODO: this is tightly coupled to the end > durations catch
        durations = np.arange(start, int(end), clip_dur)

        clip_tr = ""

        for d in durations:
            sub_clip_path = self.vid.get_audio_subclip(d, d + clip_dur)
            # TODO: add asyncio. Beware flask's gotchas as it isn't native async unless you use a specific version
            transcribed = self.transcribe(sub_clip_path)
            clip_tr = clip_tr + transcribed

        return clip_tr

    def transcribe(self, file_path, provider="google"):
        recognizer = sr.Recognizer()
        audio = sr.AudioFile(f'{file_path}')

        with audio as source:
            audio_file = recognizer.record(source)
            if self.provider == 'google':
                try: 
                    return recognizer.recognize_google(audio_file)
                except:
                    print('yeah')

    
    def save_transcript(self, path):
        with open(f'{path}.txt', mode='w', encoding='utf-8') as file:
            file.write(self.text)
            print(f'saved transript file as {path}.txt')

# extract video stills

# integrate timeline with video
#  - timeline to be generated elsewhere


if __name__ == "__main__":
    # v = Video('section1.mp4')
    # v.get_subclip(0, 60)
    # v.get_clip()
    
    # t = Transcript(v.get_clip())
    # t = Transcript('section1.mp4')
    t = Transcript('test.mp4')
    t.save_transcript("raw_transcript")
