from pytube.exceptions import RegexMatchError
from pytube import YouTube

from collections import defaultdict
import re


def extract_audio_from_video(name:str, url:str, sink_path:str, verbose:bool=False) -> tuple:
    """
    Extracts the audio from a video and saves it as an mp3 file
    :param name: name of the file
    :param url: url of the video
    :param sink_path: path where to save the mp3 file
    :param verbose: if True, prints the name and url of the video
    :return: success (True or False), logs (string)
    """
    log = f'Current file is: {name}, {url}\n'

    try:

        # get the video
        yt = YouTube(url)
        
        # extraction of the audio stream only
        files = yt.streams.filter(only_audio=True)

    except RegexMatchError:
        log += f"RegexMatchError\n"
        return False, log

    itag = None

    # get the itag of the mp3 stream
    for file in files:
        if file.mime_type == "audio/mp4":
            itag = file.itag
            break

    if not itag:
        log += "No MP3 audio found"
        return False, log

    # get the mp3 stream
    stream = yt.streams.get_by_itag(itag)
    size = stream.filesize
    if size==0:
        log += "MP3 audio not downloaded"
        return False, log
    else:
        stream.download(
            output_path=sink_path,
            filename=f'{name}.mp3'
        )
        log += "MP3 audio downloaded successfully"  
    return True, log


def extract_audio_from_playlist(playlist_path:str, sink_path:str, separator:str) -> dict:
    """
    Extracts the audio from all the videos in a playlist
    :param playlist_path: path of the playlist
    :param sink_path: path where to save the mp3 files
    :param separator: separator symbol used in the playlist
    :return: logs as a dictionary
    """

    logs_dict = defaultdict(list)

    with open(playlist_path, 'r', encoding="utf8") as playlist:
        lines = playlist.readlines()
        for line in lines[1:]:
            line = line.replace('\t', '')
            content = line.split(separator)
            name = content[3]

            # fix the name of the file to avoid errors with pytube
            regex = r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~’]"
            name = re.sub(regex, '_', name)
            name = name.replace(' ', '_').replace('__', '_')

            success, logs = extract_audio_from_video(name, content[0], sink_path)
            logs_dict[success].append(logs)

    return logs_dict
