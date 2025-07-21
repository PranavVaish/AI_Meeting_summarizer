import os
import tempfile
import mimetypes
import platform

def validate_audio_file(file_path):
    """
    Comprehensive audio file validation that checks:
    1. If the file exists
    2. If the file has content
    3. If the file appears to be an audio file based on extension and content
    
    Returns a tuple: (is_valid, message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    # Check if file has content
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "File is empty (0 bytes)"
    
    # Get file extension and check if it's a known audio type
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    # List of common audio extensions
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.mp4', '.wma']
    
    if extension not in audio_extensions:
        return False, f"File extension '{extension}' does not appear to be a supported audio format"
    
    # Try to identify the file type using several methods
    mime_type = None
    
    # Method 1: Use python-magic if available (most accurate)
    try:
        import magic
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(file_path)
        
        if not mime_type.startswith('audio/') and not mime_type.startswith('video/'):
            return False, f"File does not appear to be audio. MIME type: {mime_type}"
    except ImportError:
        # Method 2: Use mimetypes library as fallback
        mime_type = mimetypes.guess_type(file_path)[0]
        
        if mime_type and not mime_type.startswith('audio/') and not mime_type.startswith('video/'):
            return False, f"File does not appear to be audio based on extension. Guessed MIME type: {mime_type}"
    
    # Method 3: Try to use FFmpeg/FFprobe if available for deep inspection
    try:
        import subprocess
        # Check if ffprobe is available
        try:
            # On Windows, we need shell=True to find executables in PATH
            use_shell = platform.system() == "Windows"
            subprocess.run(["ffprobe", "-version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          shell=use_shell)
            has_ffprobe = True
        except (FileNotFoundError, subprocess.SubprocessError):
            has_ffprobe = False
            
        if has_ffprobe:
            # Use ffprobe to get stream information
            cmd = ["ffprobe", "-v", "error", "-show_entries", 
                   "stream=codec_type", "-of", "default=noprint_wrappers=1", file_path]
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   shell=use_shell,
                                   text=True)
            
            # Check if it contains an audio stream
            if "codec_type=audio" not in result.stdout:
                return False, "File does not contain valid audio streams according to FFprobe"
    except (ImportError, Exception) as e:
        # If ffprobe check fails, continue to next method
        pass
    
    # Method 4: Try to read with librosa as a final validation
    try:
        import librosa
        try:
            duration = librosa.get_duration(path=file_path)
            if duration <= 0:
                return False, "File could not be read as audio by librosa (zero duration)"
        except Exception as e:
            return False, f"File could not be read as audio by librosa: {str(e)}"
    except ImportError:
        # If librosa isn't available, we've done all we can with other methods
        pass
        
    # If we've made it here, the file appears to be valid audio
    return True, "File appears to be a valid audio file"