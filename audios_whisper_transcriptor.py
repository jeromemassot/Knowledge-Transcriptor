from transformers.pipelines.audio_utils import ffmpeg_read
from transformers import pipeline
import torch
import json
import os


# initialize the model and processor
BATCH_SIZE = 8
FILE_LIMIT_MB = 40
YT_LENGTH_LIMIT_S = 5400  # limit to 1.5 hour audio files

# device should be GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def init_pipeline(model_name) -> pipeline:
    """
    Initialize the pipeline
    :param model_name: name of the model to be used
    :return: pipeline object
    """
    model_name = f'openai/{model_name}'
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=device,
    )
    return pipe


def transcribe(filepath, pipe, chunks_folder) -> str:
    """
    Transcribe the audio file
    :param filepath: path to the audio file
    :param pipe: pipeline object
    :param chunks_folder: folder where the chunks will be saved
    :return: message log
    """
    
    with open(filepath, "rb") as f:
        inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    chunks = pipe(
        inputs, batch_size=BATCH_SIZE, 
        generate_kwargs={"task": 'transcribe'},
        return_timestamps=True
    )["chunks"]

    output_path = os.path.join(chunks_folder, os.path.basename(filepath).split(".")[0] + ".jsonl")
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    return f"Transcription saved to {output_path}."


def segment(filepath, segments_folder, length:int=60) -> list:
    """
    Segment the chunks to get segments of the desired length
    :param filepath: path to the jsonl file
    :param segments_folder: folder where the segments will be saved
    :param length: length of the segments in seconds
    :return: list of segments
    """

    # read the chunks as list of individual json objects
    with open(filepath, "r") as f:
        chunks = [json.loads(line) for line in f.readlines()]

    # segment the chunks
    segments = []
    current_segment = {'text': '', 'timestamp': [0, 0]}
    current_segment_length = 0
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        chunk_length = current_chunk["timestamp"][1] - current_chunk["timestamp"][0]
        if current_segment_length + chunk_length > length:
            if len(current_segment) > 0:
                segments.append(current_segment)
                current_segment_length = chunk_length
                current_segment = current_chunk
            else:
                segments.append([current_chunk])
                current_segment_length = 0
                current_segment = {'text': '', 'timestamp': [0, 0]}
        else:
            current_segment['text'] += current_chunk['text']
            current_segment['timestamp'][1] = current_chunk['timestamp'][1]
            current_segment_length += chunk_length
        i += 1

    # last segment
    segments.append(current_segment)

    output_path = os.path.join(segments_folder, os.path.basename(filepath))
    with open(output_path, "w") as f:
        for segment in segments:
            f.write(json.dumps(segment) + "\n")

    return f"Segments saved to {output_path}."
