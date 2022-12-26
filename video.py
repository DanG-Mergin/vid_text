import moviepy.editor as mp

# TODO: move to requirements file
# pip install SpeechRecognition moviepy

from pathlib import Path, PurePath
from typing import Iterator

# TODO: inherit from moviepy
# ingest video
class Video:
    path = ""
    name = ""  # file name without extension
    clip_ext = "mp4"
    clip_dir = "clips"
    clip = None
    duration = 0  # TODO: inherit from moviepy
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

    def get_clip(self, path=path):
        # TODO: evaluate impact of keeping clip object in memory.. it isn't being used here
        # - if persisting then this should absorb get_subclip
        path = path if len(path) > 0 else self.path
        # TODO: test using "with video clip as clip here to free up resources"
        # https://zulko.github.io/moviepy/getting_started/efficient_moviepy.html
        return mp.VideoFileClip(path)

    # TODO: add option for working directory
    def __get_subclip_path(self, start, end, ext):
        ext = ext if ext is not None else self.clip_ext
        return Path(self.__get_subclip_dir_path(ext)).joinpath(f"{start}_{end}.{ext}")

    def __get_subclip_dir_path(self, subfold="_clips"):
        return Path(Path(__file__).parent).joinpath(
            self.clip_dir, f"{self.name}{subfold}"
        )

    def __add_subclip(self, start, end):
        # TODO: move this outside of any loops if possible
        # create the subclips folder if it doesn't exist
        Path(f"{self.__get_subclip_dir_path()}").mkdir(parents=True, exist_ok=False)
        subclip = self.clip.subclip(start, end)
        subclip.write_videofile(f"{self.__get_subclip_path(start, end)}")

        return subclip

    def __add_audio_subclip(self, start, end, ext):
        subclip = mp.AudioFileClip(self.path).subclip(start, end)
        # ffmpeg params ATOW are for mono audio
        subclip.write_audiofile(
            f"{self.__get_subclip_path(start, end, ext)}",
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
        )

        return subclip

    def get_subclip(self, start, end, from_file: bool = False):
        # TODO: may be problematic on linux.. need to run in VM
        if not from_file:
            return self.clip.subclip(start, end)

        else:
            subclip_path = self.__get_subclip_path(start, end)
            if subclip_path.is_file():
                print(f"-- Found {subclip_path}")
                return (mp.VideoFileClip(f"{subclip_path}"), subclip_path)
            else:
                print(f"-- Did not find {subclip_path}")
                self.__add_subclip(start, end)
                return subclip_path

    def get_audio_subclip(self, start, end, ext="wav"):
        if end > self.duration:
            end = self.duration

        start = int(start)
        end = int(end)

        subclip_path = self.__get_subclip_path(start, end, ext)
        if not subclip_path.is_file():
            if not subclip_path.parent.exists():
                Path(f"{self.__get_subclip_dir_path(ext)}").mkdir(
                    parents=True, exist_ok=False
                )
            print(f"-- Did not find {subclip_path}")
            self.__add_audio_subclip(start, end, ext)
            # return self.get_audio_subclip(start, end)
        else:
            print(f"-- Found {subclip_path}")
        return subclip_path

    # TODO: test cv2 to extract images from video.  Will be faster if subclips aren't needed
    # pass in the fpm you want, not the fpm of the video
    def get_frames(self, start, end, fpm=3) -> Iterator[tuple]:
        return self.get_subclip(start, end).iter_frames(
            fps=fpm / 60, with_times=True, dtype="uint8"
        )
