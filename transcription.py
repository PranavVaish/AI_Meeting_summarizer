import os
import sys
import time
import assemblyai as aai
from typing import Callable, Optional
import librosa
import soundfile as sf
import tempfile
import numpy
from check_audio import check_audio_file

def reduce_noise(audio_path: str, output_path: str = None) -> str:
    """
    Reduces noise in an audio file using librosa and saves the output.
    If output_path is not provided, a temporary file will be created.
    """
    if output_path is None:
        # Create a temporary file with the .wav extension if no output path specified
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = temp_file.name
        temp_file.close()
    
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    
    # Compute spectrogram
    S_full = numpy.abs(librosa.stft(audio_data))
    
    # Compute noise power on first few frames
    noise_power = numpy.mean(numpy.abs(S_full[:, :10]) ** 2, axis=1)
    noise_power = noise_power[:, numpy.newaxis]
    
    # Perform noise reduction
    S_clean = S_full ** 2 - noise_power
    S_clean = numpy.maximum(S_clean, 0.0)
    S_clean = numpy.sqrt(S_clean) * numpy.exp(1.j * numpy.angle(S_full))
    
    # Inverse STFT
    reduced_noise = librosa.istft(S_clean)
    
    # Save the processed audio
    sf.write(output_path, reduced_noise, sample_rate)
    
    return output_path

def convert_audio_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Converts an audio file to properly formatted WAV file that AssemblyAI can process.
    Uses FFmpeg if available for more reliable conversion, falls back to librosa.
    Returns the path to the converted file.
    """
    import os
    import tempfile
    import platform
    import subprocess
    
    if output_path is None:
        # Create a temporary file with the .wav extension if no output path specified
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = temp_file.name
        temp_file.close()
    
    # Check if the source file exists and has content
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    if os.path.getsize(input_path) == 0:
        raise ValueError("Audio file is empty (0 bytes)")
    
    # First, try using FFmpeg for conversion (most reliable)
    try:
        # Check if ffmpeg is available
        use_shell = platform.system() == "Windows"  # Need shell=True on Windows
        try:
            subprocess.run(["ffmpeg", "-version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          shell=use_shell)
            has_ffmpeg = True
        except (FileNotFoundError, subprocess.SubprocessError):
            has_ffmpeg = False
        
        if has_ffmpeg:
            print(f"Using FFmpeg to convert {input_path} to {output_path}")
            
            # FFmpeg command: convert to 16-bit PCM WAV at 44.1kHz
            cmd = [
                "ffmpeg", 
                "-y",  # Overwrite output file if it exists
                "-i", input_path,  # Input file
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "44100",  # 44.1kHz sample rate
                "-ac", "1",  # Mono
                output_path  # Output file
            ]
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=use_shell
            )
            
            # Check if conversion was successful
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"FFmpeg conversion successful: {output_path}")
                return output_path
            else:
                print(f"FFmpeg conversion failed: {result.stderr.decode()}")
                # Fall back to librosa if FFmpeg fails
        else:
            print("FFmpeg not found, using librosa for conversion")
    except Exception as e:
        print(f"FFmpeg attempt failed: {str(e)}, falling back to librosa")
    
    # Fallback: Use librosa for conversion
    try:
        print(f"Using librosa to convert {input_path}")
        
        # Load audio with librosa (handles many formats)
        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=True)
        
        # Check for valid audio data
        if len(audio_data) == 0:
            raise ValueError("No audio data found in file")
            
        # Ensure the sample rate is supported by AssemblyAI (they support 8kHz - 48kHz)
        if sample_rate < 8000:
            # Resample to 16kHz if too low
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Write out as WAV with specific parameters AssemblyAI expects
        sf.write(
            output_path, 
            audio_data, 
            sample_rate, 
            format='WAV',
            subtype='PCM_16'  # 16-bit PCM is widely compatible
        )
        
        # Verify the file was created successfully
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ValueError("Failed to create valid WAV file")
            
        print(f"Librosa conversion successful: {output_path}")
        return output_path
        
    except Exception as e:
        # Clean up temp file if there was an error
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise ValueError(f"Audio conversion failed: {str(e)}")

def transcribe_audio(audio_path: str, status_callback: Optional[Callable] = None):
    """
    Transcribe audio with AssemblyAI using the modern SDK with progress reporting
    
    Args:
        audio_path: Path to audio file
        status_callback: Optional callback function to report progress (0-100)
    """
    # Report initial status
    if status_callback:
        status_callback(5, "Preparing audio for transcription...")
        
    # First, validate the file exists and has content
    if not os.path.exists(audio_path):
        if status_callback:
            status_callback(0, f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    if os.path.getsize(audio_path) == 0:
        if status_callback:
            status_callback(0, "Audio file is empty (0 bytes)")
        raise ValueError("Audio file is empty (0 bytes)")
    
    # Get diagnostic information about the audio file
    if status_callback:
        status_callback(8, "Checking audio file format...")
    
    audio_info = check_audio_file(audio_path)
    if audio_info["error"]:
        if status_callback:
            status_callback(10, f"Converting audio to WAV format (original had issues: {audio_info['error']})")
    else:
        if status_callback:
            status_callback(10, f"Audio file looks valid: {audio_info['duration']:.2f}s, {audio_info['sample_rate']}Hz, {audio_info['channels']} channel(s)")
    
    # Always convert the audio to a properly formatted WAV file
    try:
        wav_audio_path = convert_audio_to_wav(audio_path)
        if status_callback:
            status_callback(15, "Audio converted to WAV format successfully")
    except Exception as e:
        if status_callback:
            status_callback(0, f"Audio conversion failed: {str(e)}")
        raise ValueError(f"Failed to convert audio format: {str(e)}")

    # Verify the WAV file
    wav_info = check_audio_file(wav_audio_path)
    if wav_info["error"]:
        if status_callback:
            status_callback(0, f"WAV conversion produced invalid file: {wav_info['error']}")
        raise ValueError(f"WAV conversion produced invalid file: {wav_info['error']}")
    
    # Load API key from environment variable
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    # If not found, use a default API key as fallback
    if not api_key:
        api_key = "your api key"  # Your API key from the example
    
    # Configure the AssemblyAI client
    aai.settings.api_key = api_key
    
    # Create a transcriber with the best speech model
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        language_detection=True
    )
    
    if status_callback:
        status_callback(30, "Starting transcription with AssemblyAI SDK...")
    
    try:
        # Create transcriber with config
        transcriber = aai.Transcriber(config=config)
        
        if status_callback:
            status_callback(40, "Submitting audio for transcription...")
        
        # Directly transcribe the file (simplified approach)
        transcript = transcriber.transcribe(wav_audio_path)
        
        # Check for completion status
        if transcript.status == aai.TranscriptStatus.error:
            error_message = f"Transcription failed: {getattr(transcript, 'error', 'Unknown error')}"
            if status_callback:
                status_callback(0, error_message)
            raise Exception(error_message)
        
        if status_callback:
            status_callback(100, "Transcription completed successfully!")
        
        # Clean up temporary file
        try:
            if os.path.exists(wav_audio_path) and wav_audio_path != audio_path:
                os.remove(wav_audio_path)
        except:
            pass  # Ignore cleanup errors
        
        return transcript.text
        
    except Exception as e:
        if status_callback:
            status_callback(0, f"Transcription error: {str(e)}")
        raise e
    
if __name__ == "__main__":
            import sys
            
            if len(sys.argv) < 2:
                print("Usage: python transcription.py <audio_file_path>")
                sys.exit(1)
                
            audio_file = sys.argv[1]
            
            def print_status(progress, message):
                print(f"Progress: {progress}% - {message}")
            
            try:
                # Print diagnostic information about the file
                print(f"Attempting to transcribe file: {audio_file}")
                if os.path.exists(audio_file):
                    print(f"File size: {os.path.getsize(audio_file) / 1024:.2f} KB")
                    
                    # Check audio file format
                    print("Audio file diagnostics:")
                    audio_info = check_audio_file(audio_file)
                    for key, value in audio_info.items():
                        print(f"  - {key}: {value}")
                    
                    # Try to detect file type
                    try:
                        import magic
                        mime = magic.Magic(mime=True)
                        file_type = mime.from_file(audio_file)
                        print(f"File MIME type: {file_type}")
                    except ImportError:
                        print("python-magic not installed, cannot detect MIME type")
                else:
                    print(f"File does not exist: {audio_file}")
                    sys.exit(1)
                
                # Apply noise reduction
                print("\nApplying noise reduction...")
                denoised_audio = reduce_noise(audio_file)
                print("Noise reduction completed")
                
                # Transcribe the denoised audio
                result = transcribe_audio(denoised_audio, print_status)
                print("\nTranscription result:")
                print(result)
                
                # Save transcript to file
                transcript_file = audio_file + "_transcript.txt"
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"\nTranscript saved to: {transcript_file}")
                
                # Clean up temporary denoised file
                if os.path.exists(denoised_audio):
                    os.remove(denoised_audio)
                
            except Exception as e:
                print(f"\nError: {str(e)}")