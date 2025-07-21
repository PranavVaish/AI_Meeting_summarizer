import os
import sys
import magic
import time

def check_audio_file(filepath):
    """
    Comprehensive diagnostic tool for audio files
    
    Args:
        filepath: Path to the audio file to check
    """
    results = {
        "file_exists": False,
        "file_size": 0,
        "mime_type": None,
        "file_extension": None,
        "readable": False,
        "issues": []
    }
    
    print(f"Checking file: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        results["issues"].append("File does not exist")
        return results
    
    results["file_exists"] = True
    
    # Get file size
    try:
        size_bytes = os.path.getsize(filepath)
        results["file_size"] = size_bytes
        print(f"File size: {size_bytes} bytes ({size_bytes/1024/1024:.2f} MB)")
        
        if size_bytes == 0:
            results["issues"].append("File is empty (0 bytes)")
    except Exception as e:
        results["issues"].append(f"Error getting file size: {str(e)}")
    
    # Get file extension
    try:
        filename, extension = os.path.splitext(filepath)
        results["file_extension"] = extension.lower()
        print(f"File extension: {extension}")
        
        if extension.lower() not in ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.mp4', '.mpeg', '.webm']:
            results["issues"].append(f"File extension '{extension}' may not be a common audio format")
    except Exception as e:
        results["issues"].append(f"Error checking file extension: {str(e)}")
    
    # Get MIME type using python-magic
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(filepath)
        results["mime_type"] = mime_type
        print(f"MIME type: {mime_type}")
        
        if not mime_type.startswith(('audio/', 'video/')):
            results["issues"].append(f"MIME type '{mime_type}' does not appear to be audio or video")
            
            # Special case for 'application/octet-stream'
            if mime_type == 'application/octet-stream':
                results["issues"].append("File detected as generic binary data, not recognizable as audio")
    except Exception as e:
        results["issues"].append(f"Error checking MIME type: {str(e)}")
    
    # Try opening the file to check if readable
    try:
        with open(filepath, 'rb') as f:
            # Read a small chunk to verify it's accessible
            chunk = f.read(1024)
            results["readable"] = True
    except Exception as e:
        results["issues"].append(f"File not readable: {str(e)}")
    
    # Try to check file header
    try:
        with open(filepath, 'rb') as f:
            header = f.read(16).hex()
            print(f"File header (hex): {header}")
            
            # Check for common audio format headers
            headers = {
                "4944330": "MP3 with ID3v2 tag",
                "fffb": "MP3 frame sync",
                "fff3": "MP3 frame sync",
                "fff2": "MP3 frame sync",
                "fffe": "MP3 frame sync",
                "52494646": "WAV/RIFF format",
                "4f676753": "Ogg format",
                "664c6143": "FLAC format",
                "6674797069": "MP4/M4A container"
            }
            
            format_detected = False
            for sig, format_name in headers.items():
                if header.startswith(sig):
                    print(f"Detected format: {format_name}")
                    format_detected = True
                    break
            
            if not format_detected:
                results["issues"].append("File header doesn't match common audio formats")
    except Exception as e:
        results["issues"].append(f"Error checking file header: {str(e)}")
    
    # Try to get file creation/modification times
    try:
        creation_time = os.path.getctime(filepath)
        modify_time = os.path.getmtime(filepath)
        
        print(f"File created: {time.ctime(creation_time)}")
        print(f"File modified: {time.ctime(modify_time)}")
    except Exception as e:
        results["issues"].append(f"Error getting file timestamps: {str(e)}")
    
    # Try using ffprobe if available to get audio file info
    try:
        import subprocess
        cmd = ["ffprobe", "-hide_banner", "-i", filepath, "-show_format", "-show_streams", "-v", "error"]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode == 0:
            print("\nFFprobe detected media information:")
            ffprobe_output = process.stdout.decode()
            print(ffprobe_output)
            
            if "Stream #0:0: Audio" in ffprobe_output or "Stream #0:0(und): Audio" in ffprobe_output:
                print("FFprobe confirms this is a valid audio file")
            else:
                results["issues"].append("FFprobe didn't detect an audio stream in this file")
        else:
            error = process.stderr.decode()
            print(f"FFprobe error: {error}")
            results["issues"].append(f"FFprobe couldn't process this file: {error}")
    except FileNotFoundError:
        print("FFprobe not installed or not in PATH - cannot check media info")
    except Exception as e:
        results["issues"].append(f"Error running FFprobe: {str(e)}")
    
    # Summary
    print("\nIssues found:")
    if results["issues"]:
        for issue in results["issues"]:
            print(f" - {issue}")
    else:
        print(" - No issues detected")
    
    # Recommendations
    print("\nRecommendations:")
    if results["issues"]:
        print(" - Try converting the file to WAV format using a tool like Audacity or FFmpeg")
        print(" - Make sure the file has audio content and isn't corrupt")
        print(" - Check if the file is password protected or encrypted")
        print(" - Try a sample command: ffmpeg -i your_file -acodec pcm_s16le -ar 44100 -ac 1 output.wav")
    else:
        print(" - File appears to be a valid audio file")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_audio.py <audio_file_path>")
        sys.exit(1)
    
    check_audio_file(sys.argv[1])