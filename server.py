import time
from fastapi import FastAPI, UploadFile, File, Form, Depends, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import json
import asyncio
import time
from typing import Optional, Dict, Any, Callable
from contextlib import asynccontextmanager
from transcription import transcribe_audio, check_audio_file, convert_audio_to_wav
# Import other modules as needed
try:
    from summarization import summarize_text
    from sentiment import analyze_sentiment
    from search import index_transcript, search_meeting
except ImportError:
    # Create placeholder functions for development/testing
    def summarize_text(text, num_points=5):
        return f"• Summary with {num_points} points would go here.\n• This is a placeholder."
        
    def analyze_sentiment(text):
        return {"Overall Sentiment": "Neutral", "Compound Score": 0.0, "Positive": 0.0, "Negative": 0.0, "Neutral": 1.0}
        
    def index_transcript(text):
        pass
        
    def search_meeting(query):
        return [(f"This is a placeholder result for: {query}", 0.95)]

# Status tracking dictionary
job_status: Dict[str, Dict[str, Any]] = {}

# Define lifespan context manager to replace @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    cleanup_task = asyncio.create_task(cleanup_old_jobs())
    yield
    # Shutdown logic
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryOptions(BaseModel):
    num_summary_points: int = 5
    summary_style: str = "Bullets"

class SearchQuery(BaseModel):
    query: str

# Helper function to parse options from form data
async def get_options(options: Optional[str] = Form(None)):
    if not options:
        return SummaryOptions()
    try:
        options_dict = json.loads(options)
        return SummaryOptions(**options_dict)
    except:
        return SummaryOptions()

def update_status(job_id: str, progress: int, message: str):
    """Update the status of a job"""
    if job_id in job_status:
        job_status[job_id]["progress"] = progress
        job_status[job_id]["message"] = message
        job_status[job_id]["updated_at"] = time.time()
        # Print for debugging
        print(f"Job {job_id}: {progress}% - {message}")

async def process_audio_task(temp_audio_path: str, job_id: str, options: SummaryOptions):
    """Background task to process audio file"""
    try:
        # Create status update callback
        def status_callback(progress: int, message: str):
            update_status(job_id, progress, message)
        
        # First, check audio file format
        update_status(job_id, 5, "Checking audio file format...")
        audio_info = await run_in_threadpool(check_audio_file, temp_audio_path)
        
        # Log debug information
        print(f"Audio file info: {audio_info}")
        
        # Report audio file info
        if audio_info.get("error"):
            update_status(job_id, 10, f"Audio needs conversion: {audio_info.get('error')}")
        else:
            duration = audio_info.get("duration", 0)
            sample_rate = audio_info.get("sample_rate", 0)
            channels = audio_info.get("channels", 0)
            update_status(job_id, 10, f"Audio file: {duration:.2f}s, {sample_rate}Hz, {channels} channel(s)")
            
        # Convert to WAV before processing if needed
        wav_path = temp_audio_path
        try:
            update_status(job_id, 15, "Converting audio to proper format...")
            wav_path = await run_in_threadpool(convert_audio_to_wav, temp_audio_path)
            update_status(job_id, 20, "Audio converted successfully")
        except Exception as e:
            update_status(job_id, 0, f"Audio conversion failed: {str(e)}")
            raise ValueError(f"Audio conversion failed: {str(e)}")
        
        # Process the audio file with status updates
        # Run CPU-intensive tasks in a threadpool
        update_status(job_id, 25, "Starting transcription...")
        transcript = await run_in_threadpool(
            transcribe_audio, 
            wav_path, 
            status_callback
        )
        
        # Update status
        update_status(job_id, 80, "Transcription complete. Generating summary...")
        
        # Index the transcript for searching
        await run_in_threadpool(index_transcript, transcript)
        
        # Generate summary based on options
        summary = await run_in_threadpool(
            summarize_text,
            transcript, 
            options.num_summary_points
        )
        
        # Convert to paragraph style if selected
        if options.summary_style == "Paragraph":
            paragraph = summary.replace("• ", "")
            paragraph = paragraph.replace("\n", " ")
            summary = paragraph
        
        # Update status
        update_status(job_id, 90, "Analyzing sentiment...")
        
        # Analyze sentiment
        sentiment = await run_in_threadpool(analyze_sentiment, transcript)
        
        # Set complete results
        update_status(job_id, 100, "Processing complete")
        job_status[job_id]["complete"] = True
        job_status[job_id]["results"] = {
            "transcript": transcript,
            "summary": summary,
            "sentiment": sentiment
        }
        
    except Exception as e:
        # Update status with error
        error_message = str(e)
        print(f"Job {job_id} error: {error_message}")
        update_status(job_id, 0, f"Error: {error_message}")
        job_status[job_id]["error"] = error_message
    finally:
        # Clean up the temp files
        try:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if wav_path != temp_audio_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            print(f"Failed to clean up temp file: {str(e)}")

@app.post("/process-audio/")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: SummaryOptions = Depends(get_options)
):
    # Generate a unique job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Print debug info
    print(f"Processing file: {file.filename}")
    print(f"Content type: {file.content_type}")
    
    # First basic validation - check the content type
    if file.content_type and not (file.content_type.startswith('audio/') or file.content_type.startswith('video/')):
        print(f"Warning: Content-Type '{file.content_type}' doesn't appear to be audio")
        # We'll continue but log the warning
    
    # Read file content
    file_content = await file.read()
    file_size = len(file_content)
    print(f"File size: {file_size / 1024:.2f} KB")
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty (0 bytes)")
    
    # Determine appropriate file extension
    extension = None
    if file.filename:
        # Extract extension from filename if available
        parts = file.filename.split('.')
        if len(parts) > 1:
            extension = parts[-1].lower()
    
    # Fallback to content-type if no extension or unrecognized
    if not extension or extension not in ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg', 'mp4', 'wma']:
        # Extract extension from content type if possible
        if file.content_type and '/' in file.content_type:
            mime_subtype = file.content_type.split('/')[-1]
            if mime_subtype in ['mpeg', 'mp3']:
                extension = 'mp3'
            elif mime_subtype in ['wav', 'x-wav', 'wave']:
                extension = 'wav'
            elif mime_subtype in ['aac']:
                extension = 'aac'
            elif mime_subtype in ['ogg', 'vorbis']:
                extension = 'ogg'
            elif mime_subtype in ['flac']:
                extension = 'flac'
            else:
                # Default extension as fallback
                extension = 'mp3'
        else:
            # Last resort fallback
            extension = 'mp3'
    
    print(f"Using file extension: .{extension}")
    
    # Save the uploaded file to a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_audio:
        temp_audio.write(file_content)
        temp_audio_path = temp_audio.name
    
    print(f"Saved temporary file to: {temp_audio_path}")
    
    # Reset file for future reading
    await file.seek(0)
    
    # Try to detect file type using more methods
    try:
        import magic
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(temp_audio_path)
        print(f"File MIME type detected by python-magic: {file_type}")
        
        if not file_type.startswith('audio/') and not file_type.startswith('video/'):
            print(f"Warning: File doesn't appear to be audio according to python-magic. MIME: {file_type}")
    except ImportError:
        print("python-magic not installed, cannot detect MIME type")
    
    # Try using the validation function
    try:
        from file_validation import validate_audio_file
        is_valid, message = validate_audio_file(temp_audio_path)
        if not is_valid:
            print(f"Audio validation failed: {message}")
            # We could raise an exception here, but let's try to process anyway
            # and let the conversion/transcription handle errors
    except ImportError:
        print("Audio validation module not available")
    
    # Initialize job status
    job_status[job_id] = {
        "progress": 0,
        "message": "Job started",
        "created_at": time.time(),
        "updated_at": time.time(),
        "complete": False,
        "error": None,
        "results": None
    }
    
    # Start the processing in a background task
    background_tasks.add_task(process_audio_task, temp_audio_path, job_id, options)
    
    # Return the job ID immediately
    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get the job status
    status = job_status[job_id]
    
    # If the job is complete or has an error, return results
    if status["error"]:
        return {
            "status": "error",
            "message": status["message"],
            "error": status["error"]
        }
    elif status["complete"]:
        return {
            "status": "complete",
            "results": status["results"]
        }
    else:
        # Return progress information
        return {
            "status": "processing",
            "progress": status["progress"],
            "message": status["message"]
        }

@app.post("/search/")
async def search_transcript(search_request: SearchQuery):
    results = await run_in_threadpool(search_meeting, search_request.query)
    return {"results": [{"sentence": sentence, "score": score} for sentence, score in results]}

# Add a simple health check endpoint for testing
@app.get("/health")
def health_check():
    return {"status": "OK"}

# Cleanup job function moved outside of the startup event
async def cleanup_old_jobs():
    while True:
        await asyncio.sleep(3600)  # Run every hour
        current_time = time.time()
        keys_to_remove = []
        
        for job_id, status in job_status.items():
            # Remove jobs older than 24 hours
            if current_time - status["created_at"] > 86400:
                keys_to_remove.append(job_id)
        
        for job_id in keys_to_remove:
            del job_status[job_id]
            
        print(f"Cleaned up {len(keys_to_remove)} old jobs")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)