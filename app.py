import streamlit as st
import tempfile
import os
import nltk
import requests
import json
import io
import time
import mimetypes
import logging
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page title and configuration
st.set_page_config(page_title="AI Meeting Summarizer", layout="wide")
st.title("AI Meeting Summarizer")

# Check if server is running
def is_server_running():
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Display server status in sidebar
with st.sidebar:
    st.subheader("Server Status")
    server_status = is_server_running()
    if server_status:
        st.success("✅ FastAPI Server Online")
    else:
        st.error("❌ FastAPI Server Offline")
        st.info("Please run 'python server.py' in a separate terminal")

# Ensure NLTK dependencies are downloaded
@st.cache_resource
def safe_convert_audio(audio_file):
    try:
        # Convert audio using pydub
        audio = AudioSegment.from_file(audio_file)
        # Convert to WAV format with standard parameters
        buffer = io.BytesIO()
        audio.export(buffer, format='wav', parameters=['-ac', '1', '-ar', '16000'])
        
        conversion_info = {
            "converted_size": len(buffer.getvalue()),
            "format": "wav",
            "channels": 1,
            "sample_rate": 16000
        }
        
        return buffer.getvalue(), conversion_info
    except Exception as e:
        return audio_file.getvalue(), {"error": str(e)}

def download_nltk_data():
    try:
        # Check if NLTK data is already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
    except Exception as e:
        st.warning(f"Could not download NLTK data: {str(e)}. Some features may be limited.")

# Download NLTK data
download_nltk_data()

# Initialize session state variables if they don't exist
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None

# Show sidebar with options
with st.sidebar:
    st.subheader("Settings")
    num_summary_points = st.slider("Number of summary points", 3, 10, 5)
    summary_style = st.radio("Summary Style", ["Bullets", "Paragraph"])
    
    # Debug options
    st.subheader("Debug Options")
    convert_audio = st.checkbox("Convert audio before sending", value=True)
    debug_mode = st.checkbox("Show debug information", value=False)

# File uploader
uploaded_file = st.file_uploader("Upload an audio/video file", type=["mp3", "wav", "m4a", "flac", "aac", "ogg", "mp4", "avi", "mkv"])

# Handle video files by extracting audio
if uploaded_file is not None and uploaded_file.type and 'video' in uploaded_file.type:
    try:
        st.info("Processing video file to extract audio...")
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_video:
            temp_video.write(uploaded_file.getvalue())
            temp_video_path = temp_video.name

        # Extract audio using pydub
        video_audio = AudioSegment.from_file(temp_video_path)
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            video_audio.export(temp_audio.name, format='wav')
            # Read the extracted audio
            with open(temp_audio.name, 'rb') as audio_file:
                uploaded_file = st.BytesIO(audio_file.read())
                uploaded_file.name = os.path.splitext(uploaded_file.name)[0] + '.wav'
                uploaded_file.type = 'audio/wav'

        # Clean up temp files
        os.unlink(temp_video_path)
        os.unlink(temp_audio.name)
        
    except Exception as e:
        st.error(f"Error extracting audio from video: {str(e)}")
        uploaded_file = None


# Display debug information if enabled
if debug_mode and uploaded_file is not None:
    st.sidebar.subheader("File Information")
    file_info = {
        "Filename": uploaded_file.name,
        "Size": f"{len(uploaded_file.getvalue())} bytes",
        "Type": uploaded_file.type if hasattr(uploaded_file, 'type') and uploaded_file.type else "Unknown"
    }
    
    # Guess MIME type
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    file_info["Detected MIME Type"] = mime_type or "Unknown"
    
    st.sidebar.json(file_info)

# Display audio player if file is uploaded
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    
    # Process button
    if st.button("Process Audio"):
        # First check if server is running
        if not is_server_running():
            st.error("Cannot process audio: FastAPI server is not running. Please start the server with 'python server.py' in a separate terminal.")
        else:
            st.session_state.processing = True
            st.session_state.error_message = None
            st.session_state.job_id = None
            st.session_state.debug_info = None
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            debug_text = st.empty()
            
            # Process audio via FastAPI with background processing
            try:
                # Set the status
                status_text.text("Preparing audio file...")
                progress_bar.progress(5)
                
                # Reset the file pointer to the beginning
                uploaded_file.seek(0)
                
                debug_info = {
                    "original_filename": uploaded_file.name,
                    "original_file_size": len(uploaded_file.getvalue())
                }
                
                file_bytes = uploaded_file.getvalue()
                mime_type = uploaded_file.type or f"audio/{uploaded_file.name.split('.')[-1]}"
                filename = uploaded_file.name
                
                # Convert audio if enabled
                if convert_audio:
                    status_text.text("Converting audio to optimize for transcription...")
                    progress_bar.progress(8)
                    
                    try:
                        file_bytes, conversion_info = safe_convert_audio(uploaded_file)
                        debug_info.update(conversion_info)
                        
                        if "error" in conversion_info:
                            status_text.text(f"Audio conversion warning: {conversion_info['error']}. Proceeding anyway...")
                        else:
                            status_text.text("Audio conversion successful")
                        
                        # Update MIME type and filename for the converted file
                        mime_type = "audio/wav" 
                        filename = os.path.splitext(uploaded_file.name)[0] + ".wav"
                        
                    except Exception as e:
                        status_text.text(f"Audio conversion failed: {str(e)}. Using original file...")
                        logger.error(f"Error converting audio: {str(e)}")
                        # Continue with original file if conversion fails
                        uploaded_file.seek(0)
                        file_bytes = uploaded_file.getvalue()
                        debug_info["conversion_error"] = str(e)
                
                # Update status  
                status_text.text("Sending file to processing server...")
                progress_bar.progress(10)
                
                if debug_mode:
                    debug_info["mime_type_sent"] = mime_type
                    debug_info["filename_sent"] = filename
                
                # Prepare the form data
                files = {
                    "file": (filename, io.BytesIO(file_bytes), mime_type)
                }
                
                data = {
                    "options": json.dumps({
                        "num_summary_points": num_summary_points,
                        "summary_style": summary_style,
                        "debug": debug_mode
                    })
                }
                
                # Send the request to the FastAPI server
                response = requests.post(
                    "http://localhost:8000/process-audio/",
                    files=files,
                    data=data,
                    timeout=60  # 1 minute timeout for initial upload
                )
                
                if response.status_code == 200:
                    # Get the job ID
                    job_data = response.json()
                    job_id = job_data.get("job_id")
                    st.session_state.job_id = job_id
                    
                    if debug_mode:
                        debug_info["job_id"] = job_id
                        st.session_state.debug_info = debug_info
                        debug_text.json(debug_info)
                    
                    # Poll for job status
                    max_wait_time = 20 * 60  # 20 minute maximum wait time
                    start_time = time.time()
                    
                    while time.time() - start_time < max_wait_time:
                        # Get job status
                        status_response = requests.get(f"http://localhost:8000/job-status/{job_id}", timeout=10)
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            
                            # Update debug info if available
                            if debug_mode and "debug_info" in status_data:
                                debug_info.update(status_data["debug_info"])
                                st.session_state.debug_info = debug_info
                                debug_text.json(debug_info)
                            
                            if status_data.get("status") == "complete":
                                # Process complete, get results
                                results = status_data.get("results", {})
                                st.session_state.transcript = results.get("transcript")
                                st.session_state.summary = results.get("summary")
                                st.session_state.sentiment = results.get("sentiment")
                                
                                progress_bar.progress(100)
                                status_text.text("Processing complete!")
                                
                                # Force a rerun to update the UI
                                st.session_state.processing = False
                                st.session_state.job_id = None
                                st.rerun()
                                break
                                
                            elif status_data.get("status") == "error":
                                # Process error
                                error_message = status_data.get("error", "Unknown error")
                                st.session_state.error_message = error_message
                                st.session_state.processing = False
                                st.session_state.job_id = None
                                
                                if debug_mode and "debug_info" in status_data:
                                    debug_info.update(status_data["debug_info"])
                                    st.session_state.debug_info = debug_info
                                
                                st.error(f"Error processing audio: {error_message}")
                                break
                                
                            else:
                                # Process is still running, update progress
                                progress = status_data.get("progress", 0)
                                message = status_data.get("message", "Processing...")
                                
                                progress_bar.progress(progress)
                                status_text.text(message)
                                
                                # Sleep before next poll
                                time.sleep(2)
                        else:
                            st.error(f"Error checking job status: {status_response.status_code} - {status_response.text}")
                            st.session_state.processing = False
                            break
                    
                    # Check if we timed out
                    if time.time() - start_time >= max_wait_time:
                        st.error("Processing timed out after maximum wait time")
                        st.session_state.processing = False
                        
                else:
                    st.error(f"Error from server: {response.status_code} - {response.text}")
                    st.session_state.processing = False
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the processing server. Make sure the FastAPI server is running on port 8000.")
                st.session_state.processing = False
            except requests.exceptions.ReadTimeout:
                st.error("The initial request to the server timed out. This might be due to a very large file or server overload.")
                st.session_state.processing = False
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                st.session_state.processing = False

# Check for ongoing processing on page load/refresh
if st.session_state.processing and st.session_state.job_id:
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Get job status
            status_response = requests.get(f"http://localhost:8000/job-status/{st.session_state.job_id}", timeout=10)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                if status_data.get("status") == "complete":
                    # Process complete, get results
                    results = status_data.get("results", {})
                    st.session_state.transcript = results.get("transcript")
                    st.session_state.summary = results.get("summary")
                    st.session_state.sentiment = results.get("sentiment")
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Update session state
                    st.session_state.processing = False
                    st.session_state.job_id = None
                    st.rerun()
                    
                elif status_data.get("status") == "error":
                    # Process error
                    error_message = status_data.get("error", "Unknown error")
                    st.session_state.error_message = error_message
                    st.session_state.processing = False
                    st.session_state.job_id = None
                    st.error(f"Error processing audio: {error_message}")
                    
                else:
                    # Process is still running, update progress
                    progress = status_data.get("progress", 0)
                    message = status_data.get("message", "Processing...")
                    
                    progress_bar.progress(progress)
                    status_text.text(message)
                    
                    # Add a cancel button
                    if st.button("Cancel Processing"):
                        st.session_state.processing = False
                        st.session_state.job_id = None
                        st.rerun()
        except Exception as e:
            # Error checking status
            st.session_state.processing = False
            st.session_state.job_id = None
            st.error(f"Failed to check processing status: {str(e)}")

# Display error message if there is one
if st.session_state.error_message:
    st.error(f"Error processing audio: {st.session_state.error_message}")

# Display debug information if available
if debug_mode and st.session_state.debug_info:
    with st.expander("Debug Information", expanded=False):
        st.json(st.session_state.debug_info)

# Display results if available
if st.session_state.transcript is not None:
    with st.expander("Transcript", expanded=True):
        st.write(st.session_state.transcript)
    
    # Search functionality
    st.subheader("Search Meeting Content")
    search_query = st.text_input("Search the meeting transcript:")
    if search_query:
        if not is_server_running():
            st.error("Cannot search: FastAPI server is not running")
        else:
            try:
                # Call the search endpoint
                search_response = requests.post(
                    "http://localhost:8000/search/",
                    json={"query": search_query}
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json().get("results", [])
                    st.write(f"Found {len(search_results)} relevant segments:")
                    for result in search_results:
                        st.markdown(f"- *{result['sentence']}* (Relevance: {result['score']:.2f})")
                else:
                    st.error(f"Error searching transcript: {search_response.status_code} - {search_response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the search server. Make sure the FastAPI server is running.")
    
    if st.session_state.summary is not None:
        st.subheader("Meeting Summary")
        st.write(st.session_state.summary)
    
    if st.session_state.sentiment is not None:
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Sentiment", st.session_state.sentiment["Overall Sentiment"])
            if "Compound Score" in st.session_state.sentiment:
                st.metric("Compound Score", f"{st.session_state.sentiment['Compound Score']:.2f}")
        
        with col2:
            st.metric("Positive", f"{float(st.session_state.sentiment['Positive'].strip('%')):.2f}")
            st.metric("Negative", f"{float(st.session_state.sentiment['Negative'].strip('%')):.2f}")
            st.metric("Neutral", f"{float(st.session_state.sentiment['Neutral'].strip('%')):.2f}")
            st.metric("Positive Sentences", st.session_state.sentiment["Stats"]["Positive Sentences"])
            st.metric("Negative Sentences", st.session_state.sentiment["Stats"]["Negative Sentences"])
            st.metric("Neutral Sentences", st.session_state.sentiment["Stats"]["Neutral Sentences"])
            st.metric("Average Positive Score", f"{st.session_state.sentiment['Stats']['Average Scores']['Positive']:.2f}")
            st.metric("Average Negative Score", f"{st.session_state.sentiment['Stats']['Average Scores']['Negative']:.2f}")
            st.metric("Average Neutral Score", f"{st.session_state.sentiment['Stats']['Average Scores']['Neutral']:.2f}")
            st.metric("Total Sentences", st.session_state.sentiment["Stats"]["Total Sentences"])

# Add a refresh button
if st.session_state.transcript is not None:
    if st.button("Start Over"):
        # Reset all session state variables
        st.session_state.transcript = None
        st.session_state.summary = None
        st.session_state.sentiment = None
        st.session_state.processing = False
        st.session_state.job_id = None
        st.session_state.error_message = None
        st.session_state.debug_info = None
        st.rerun()

# Add footer
st.markdown("---")
st.markdown("AI Meeting Summarizer | Using AssemblyAI and NLTK for Audio Processing")