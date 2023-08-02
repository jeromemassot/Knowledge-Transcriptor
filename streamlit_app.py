from audios_whisper_transcriptor import init_pipeline, transcribe, segment
from videos_stream_retriever import extract_audio_from_playlist
import streamlit as st
import os


# add the ffmpeg binaries to the path
@st.cache_resource()
def update_path():
    ffmpeg_path = os.path.join("E:\\", "03- Knowledge Projects", "02- Knowledge Transcriptor", "resources", "bin")
    os.environ["PATH"] += os.pathsep + ffmpeg_path


update_path()

@st.cache_resource(show_spinner=False)
def _init_pipeline(model_name):
    """
    Initialize the pipeline
    :param model_name: name of the model to be used
    :return: pipeline object
    """
    return init_pipeline(model_name)


st.title("YouTube Playlist Semantic Search")

st.subheader("Upload YouTube videos playlist")

st.markdown("""
    The first step of the pipeline is to upload a list of videos from a YouTube playlist.
    The list must be a csv file with two columns: video name and url of the video.
    The file can contain additional columns that could be used as metadata for filtering.
""")

# upload playlist file
uploaded_object = st.file_uploader("Upload Playlist", type="csv")
if uploaded_object:
    input_path = os.path.join("./inputs", uploaded_object.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_object.getbuffer())
    st.success("File uploaded successfully and added to the playlists folder")

st.subheader("Extract audio streams from the playlist videos")

st.markdown("""
    The second step of the pipeline is to extract all the audio streams from
    the videos mentionned in the playlist. The audios files are saved in a
    collection of mp3 files.
""")

st.info("""
    The extraction of the audio streams can take a while depending 
    on the number of videos in the playlist
""")

# list of playlist files
playlist_files = os.listdir("./inputs")

# select the playlist file
play_list_filename = st.selectbox("Playlist", options=playlist_files)

# open the playlist file to check its content and the separator symbol
st.warning("""Please inspect the playlist in particular the separator symbol.""")
with st.expander("Inspect playlist schema"):
    with open(os.path.join("./inputs", play_list_filename), "r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [l.replace('\t', '') for l in lines[:2]]
        st.write(lines)

# specify the separator symbol
separator = st.selectbox("Separator symbol", options=[",", ";", "|"])

# collection name to be created or updated
collection_name = st.text_input("Collection name", value="my_collection")

# extract audio streams button
extract_streams_button = st.button("Extract audio from playlist")
collection_path = os.path.join("./mp3", collection_name)

if extract_streams_button and collection_name and play_list_filename:
    play_list_path = os.path.join("./inputs", play_list_filename)

    if not os.path.exists(collection_path):
        os.mkdir(collection_path)

    logs = extract_audio_from_playlist(play_list_path, collection_path, separator)

    with st.expander("Show logs"):
        st.write(logs)

st.subheader("Transcript the audio streams")

st.markdown("""
    The third step of the pipeline is to extract the transcripts from 
    the audio streams. OpenAI Whisper model is used for this operation.
    Depending the available resources, the operation can take a while.
""")

st.info("""
    The transcription of the audio streams can take a while depending 
    if the GPU is available or not.
""")

# list of mp3 collections
mp3_collections = os.listdir("./mp3")

# select the mp3 collection folder
mp3_collection_name = st.selectbox("Collection of audio files", options=mp3_collections)
mp3_collection_path = os.path.join("./mp3", mp3_collection_name)

# set the output chunks folder
chunks_folder =  os.path.join("./outputs/chunks", mp3_collection_name)
if not os.path.exists(chunks_folder):
    os.mkdir(chunks_folder)

# select the whisper model to be used
model_name = st.selectbox("Model", options=["whisper-tiny", "whisper-medium", "whisper-large"])

# transcript audio streams button
transcript_streams_button = st.button("Transcript selected collection")

if transcript_streams_button and mp3_collection_path:
    with st.spinner("Loading pipeline..."):
        pipe = _init_pipeline(model_name)

    # transcripting the audio streams and save the chunks as jsonl files
    transcription_log = []
    k = 0
    for audio_file in os.listdir(mp3_collection_path):
        audio_path = os.path.join(mp3_collection_path, audio_file)
        msg = transcribe(audio_path, pipe, chunks_folder)
        transcription_log.append(msg)
        if k > 3:
            break
        else:
            k += 1

    # add some fun
    st.balloons()

    st.write(f'Transcription completed for {len(transcription_log)} audio file(s).')

st.subheader("Create Segments from the transcription")

st.markdown("""
    The fourth step of the pipeline is to merge the transcripted pieces into
    segments of desired length.
""")

st.info("""
    You can choose the length of the segments in seconds. Remember that the segments
    should be long enough to contain a consistent piece of information, but not too
    long to avoid the loss of information. The default segment length is 60 seconds.
""")

# list of chunks collections
chunks_collections = os.listdir("./outputs/chunks")

# select the chunks collection folder
chunks_collection_name = st.selectbox("Collection", options=chunks_collections)
chunks_collection_path = os.path.join("./outputs/chunks", chunks_collection_name)

# set the output chunks folder
segments_folder =  os.path.join("./outputs/segments", chunks_collection_name)
if not os.path.exists(segments_folder):
    os.mkdir(segments_folder)


with st.form(key="segment_form"):
    # select the segment length
    segment_length = st.slider(
        "Segment length (seconds)", min_value=30, max_value=180, value=60, step=30
    )
     # segment the chunks button
    segment_chunks_button = st.form_submit_button("Segment selected transcriptions")

if segment_chunks_button and chunks_collection_path:
    segmentation_log = []
    for chunk_file in os.listdir(chunks_collection_path):
        chunk_path = os.path.join(chunks_collection_path, chunk_file)
        msg = segment(chunk_path, segments_folder, segment_length)
        segmentation_log.append(msg)
    
     # add some fun
    st.balloons()

    st.write(f'Segmentation completed for {len(segmentation_log)} chunked file(s).')
