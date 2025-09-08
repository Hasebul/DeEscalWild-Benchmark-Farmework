import os
import numpy as np
import sounddevice as sd
import wave
import simpleaudio as sa
import re
from datetime import datetime
import time
import textwrap
import logging
from pathlib import Path
import csv
import random
import sys  # sys import for path handling
import whisper
from ollama import Client as OllamaClient # Using Client from ollama
from google.cloud import texttospeech # Added for Google TTS

# ============================
# LOGGING SETUP (Basic Example)
# ============================
# Configure basic logging to file and console
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Adjust level as needed (e.g., DEBUG for more detail)

# Console Handler (logs to console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# Prevent adding handler multiple times if script is re-run in some environments
if not logger.handlers:
    logger.addHandler(console_handler)

# ============================
# Check and set FFmpeg path for Whisper
# ============================
FFMPEG_PATH = None
possible_ffmpeg_paths = [
    r"C:\FFmpeg\bin\ffmpeg.exe",  # Standard FFmpeg installation
    r"C:\FFmpeg\ffmpeg.exe",       # Direct in FFmpeg folder
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin", "ffmpeg.exe")
]

for path in possible_ffmpeg_paths:
    if os.path.exists(path):
        FFMPEG_PATH = path
        os.environ["FFMPEG_BINARY"] = path  # Set environment variable
        os.environ["PATH"] = f"{os.path.dirname(path)};{os.environ['PATH']}"  # Add to PATH
        logger.info(f"Found FFmpeg at: {FFMPEG_PATH}")
        break

if not FFMPEG_PATH:
    logger.warning("FFmpeg not found in standard locations. Whisper transcription may not work properly.")
else:
    # Inform whisper about the FFmpeg location
    whisper.audio.FFMPEG_BINARY = FFMPEG_PATH

# ============================
# CONFIGURATION
# ============================
class Config:
    def __init__(self):
        self.FS = 16000  # Sample Rate (Hertz)
        self.INPUT_DEVICE = None # Default audio input device. Specify index if needed (e.g., 1).
        self.SILENCE_THRESHOLD = 0.025 # Amplitude threshold to detect silence.
        self.SILENCE_DURATION = 1.5 # Seconds of silence to stop recording.
        # --- IMPORTANT: Set path for session data ---
        self.SESSION_DATA_DIR = Path("./session_data") # Base directory for logs and audio.

        # TTS Defaults (pitch/rate might not be used by local TTS if not supported)
        self.BASE_PITCH = -0.1 # Base pitch adjustment for TTS.
        self.BASE_RATE = 0.9 # Base speaking rate for TTS (1.0 is normal).

        # Ensure session data directory exists
        self.SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# LOCAL MODEL CLIENTS
# ============================
class WhisperSTTClient:
    def __init__(self, model_path="base"): # Using standard Whisper model names e.g. "base", "small", "medium", "large"
        logger.info(f"Loading Whisper model: {model_path}")
        try:
            self.model = whisper.load_model(model_path)
            logger.info(f"Whisper model '{model_path}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_path}': {e}", exc_info=True)
            logger.error("Ensure Whisper is installed and model files are accessible if providing a custom path.")
            raise

    def transcribe(self, audio_path):
        """
        Transcribes the audio file using the local Whisper model.

        Args:
            audio_path (pathlib.Path or str): Path to the audio file.

        Returns:
            str: The transcribed text.
        """
        try:
            # Ensure audio_path is an absolute path string for Whisper
            resolved_path = Path(audio_path).resolve()
            abs_audio_path_str = str(resolved_path)
            
            if not resolved_path.is_file():
                logger.error(f"Audio file not found at resolved path: {abs_audio_path_str}")
                return ""
            
            logger.debug(f"Transcribing audio file: {abs_audio_path_str}")
            
            # Try Method 1: Load audio with whisper.load_audio then transcribe
            try:
                logger.debug("Using Method 1: whisper's built-in audio loader")
                audio = whisper.load_audio(abs_audio_path_str)
                result = self.model.transcribe(audio)
                return result["text"].strip()
            except Exception as e1:
                logger.warning(f"Method 1 transcription failed: {e1}")
                
                # Try Method 2: Direct file path transcribe as fallback
                try:
                    logger.debug("Using Method 2: direct file path transcription")
                    result = self.model.transcribe(abs_audio_path_str)
                    return result["text"].strip()
                except Exception as e2:
                    logger.error(f"Method 2 transcription also failed: {e2}")
                    return ""
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            return ""

class Gemma3Client: # Using Ollama to run gemma3 model
    def __init__(self, model_name="gemma3:4b"): # Using gemma3:4b model via Ollama
        logger.info(f"Initializing Ollama client for model: {model_name}")
        try:
            self.client = OllamaClient() # Default host, or specify host='http://localhost:11434'
            self.model_name = model_name
            # Check if model is available by trying to show its info
            self.client.show(self.model_name)
            logger.info(f"Ollama client initialized. Using model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client or access model '{model_name}': {e}", exc_info=True)
            logger.error("Ensure Ollama server is running and the model is pulled (e.g., 'ollama pull gemma3:4b').")
            raise

    def generate(self, prompt):
        """
        Generates text using the configured Ollama model.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The generated text response.
        """
        try:
            # Using the 'generate' endpoint for single prompt completion
            response = self.client.generate(model=self.model_name, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Ollama LLM generation error: {e}", exc_info=True)
            return f"[LLM Error: {e}]"

class GoogleTTSClient:
    """Handles speech synthesis using Google Cloud Text-to-Speech."""
    def __init__(self, config):
        self.config = config
        self.language_code = "en-US"  # Default language code
        self.voice_name = None  # Let API choose voice based on language and gender

        # HARDCODED API KEY - Consider security implications
        api_key = "AIzaSyCzUSIKHGh6TdffCBT8QRWjc5Bko98BCL8" # Replace with your actual API key
        
        try:
            logger.info("Attempting to initialize Google TextToSpeechClient with hardcoded API key.")
            client_options = {'api_key': api_key}
            self.client = texttospeech.TextToSpeechClient(client_options=client_options)
            logger.info("Google TextToSpeechClient initialized successfully using hardcoded API key.")
        except Exception as e:
            logger.error(f"Failed to initialize Google TextToSpeechClient with hardcoded API key: {e}", exc_info=True)
            logger.error("Ensure the hardcoded API key is valid and the Google Cloud Text-to-Speech API is enabled for it.")
            raise

    def synthesize(self, text, out_wav_path):
        """
        Synthesizes speech from text and saves it to a WAV file.

        Args:
            text (str): The text to synthesize.
            out_wav_path (pathlib.Path or str): The path to save the output WAV file.

        Returns:
            float: Duration of the synthesized audio in seconds. Returns 0.0 on error.
        """
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Voice selection - use a default voice name if none specified
            voice_name = self.voice_name or "en-US-Studio-Q"
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=voice_name
            )
            
            # Audio configuration specifies output format and properties.
            # Use the same sample rate as the config FS for consistency
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=self.config.BASE_RATE, # From Config class
                pitch=self.config.BASE_PITCH,       # From Config class
                sample_rate_hertz=self.config.FS  # Use config FS instead of hardcoded value
            )

            # Create the request object as done in the reference code
            request = texttospeech.SynthesizeSpeechRequest(
                input=synthesis_input, 
                voice=voice_params, 
                audio_config=audio_config
            )

            logger.info(f"Synthesizing speech with Google TTS: '{text[:50]}...' (Rate: {self.config.BASE_RATE}, Pitch: {self.config.BASE_PITCH})")
            
            # Use the request object format as in reference code
            response = self.client.synthesize_speech(request=request)

            # Ensure the directory exists
            Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the audio to a file
            with open(str(out_wav_path), "wb") as out:
                out.write(response.audio_content)
            logger.info(f"Audio content written to file: {out_wav_path}")

            # Calculate duration using wave module
            duration = 0.0
            try:
                with wave.open(str(out_wav_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        duration = frames / float(rate)
                    else:
                        logger.warning(f"Could not determine sample rate from WAV header for {out_wav_path}. Duration set to 0.")
            except wave.Error as e:
                logger.error(f"Could not read duration from synthesized WAV file {out_wav_path}: {e}")
                # Estimate duration if file reading failed but we have content
                if len(response.audio_content) > 0:
                    bytes_per_sample = 2  # For LINEAR16
                    num_samples = len(response.audio_content) / bytes_per_sample
                    duration = num_samples / self.config.FS if self.config.FS > 0 else 0
                    logger.warning(f"Could not read duration from file, estimated as {duration:.2f}s")
            except Exception as e:
                logger.error(f"Unexpected error reading duration from {out_wav_path}: {e}")
            
            return duration

        except Exception as e:
            logger.error(f"Google TTS synthesis error: {e}", exc_info=True)
            return 0.0

# ============================
# AUDIO HANDLING
# ============================
class AudioHandler:
    """Handles audio recording, playback, transcription, and synthesis using local/cloud models."""
    def __init__(self, config, stt_client: WhisperSTTClient, tts_client: GoogleTTSClient): # Updated tts_client type hint
        self.config = config
        self.stt_client = stt_client
        self.tts_client = tts_client # Now expects an instance of GoogleTTSClient
        logger.info("AudioHandler initialized with WhisperSTTClient and GoogleTTSClient.")

    def get_audio_duration(self, filename):
        """Calculates the duration of a WAV file in seconds."""
        try:
            filepath_str = str(filename)
            if not Path(filepath_str).is_file():
                logger.error(f"Wave file not found for duration calculation: {filepath_str}")
                return 0.0

            with wave.open(filepath_str, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate == 0:
                    logger.error(f"Wave file has zero framerate: {filepath_str}")
                    return 0.0
                duration = frames / float(rate)
                return duration
        except wave.Error as e:
            logger.error(f"Could not read wave file duration {filename}: {e}")
            return 0.0
        except FileNotFoundError:
            logger.error(f"Wave file not found (FileNotFoundError): {filename}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error getting audio duration for {filename}: {e}")
            return 0.0


    def record_audio(self, filename):
        """
        Records audio from the input device until silence is detected.
        Saves the recording to the specified filename (full path).
        
        Returns:
            tuple: (success_flag, start_time, end_time, actual_duration)
        """
        frames = []
        chunk_size = int(self.config.FS * 0.1)  # 100ms chunks
        max_chunks_for_silence = int(self.config.SILENCE_DURATION / 0.1)
        silence_chunk_counter = 0
        recording_started = False 
        start_time = time.perf_counter()
        end_time = start_time  # Initial value, will be updated

        logger.info(f"[Recording audio to {filename.name}... Speak now.]")

        try:
            with sd.InputStream(
                device=self.config.INPUT_DEVICE,
                samplerate=self.config.FS,
                channels=1,
                dtype='float32',
                blocksize=chunk_size
            ) as stream:
                recording_started = True
                while True:
                    chunk, overflowed = stream.read(chunk_size)
                    if overflowed:
                        logger.warning("Audio input overflowed during recording.")
                    
                    frames.append(chunk)
                    amplitude = np.abs(chunk).mean()
                    
                    if amplitude < self.config.SILENCE_THRESHOLD:
                        silence_chunk_counter += 1
                        if silence_chunk_counter >= max_chunks_for_silence:
                            logger.info(f"Silence detected for >= {self.config.SILENCE_DURATION}s. Stopping recording.")
                            break
                    else:
                        silence_chunk_counter = 0
            
            # Update end_time for normal completion
            end_time = time.perf_counter()

        except sd.PortAudioError as pae:
            logger.error(f"PortAudioError during recording: {pae}")
            if "Invalid device" in str(pae):
                logger.error("Please check the configured INPUT_DEVICE index.")
            return False, start_time, time.perf_counter(), 0.0
        except Exception as e:
            logger.error(f"Generic recording error: {str(e)}", exc_info=True)
            current_time_on_error = time.perf_counter()
            effective_end_time = current_time_on_error if recording_started else start_time
            return False, start_time, effective_end_time, 0.0

        if not frames:
            logger.warning("No audio frames recorded.")
            return False, start_time, end_time, 0.0

        audio_data = np.concatenate(frames, axis=0)
        
        # Trim trailing silence if detected
        if silence_chunk_counter >= max_chunks_for_silence:
            num_silent_samples = silence_chunk_counter * chunk_size
            if len(audio_data) > num_silent_samples:
                audio_data = audio_data[:-num_silent_samples]
                logger.debug(f"Trimmed approx {num_silent_samples / self.config.FS:.2f}s of trailing silence.")

        actual_duration = len(audio_data) / self.config.FS if self.config.FS > 0 else 0
        min_rec_duration = 0.1  # Minimum valid recording duration
        
        if actual_duration < min_rec_duration:
            logger.warning(f"Recorded audio duration ({actual_duration:.2f}s) is less than minimum {min_rec_duration}s. Discarding.")
            return False, start_time, end_time, actual_duration

        # Convert to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.FS)
                wf.writeframes(audio_int16.tobytes())
            
            logger.info(f"Audio successfully recorded and saved to {filename.name} ({actual_duration:.2f}s)")
            return True, start_time, end_time, actual_duration
        except Exception as e:
            logger.error(f"Error saving audio file {filename}: {str(e)}")
            return False, start_time, end_time, actual_duration


    def play_audio(self, filename):
        """Plays the specified audio file using simpleaudio."""
        start_time = 0.0
        end_time = 0.0
        duration = self.get_audio_duration(filename)
        logger.info(f"[Playing audio: {filename.name} ({duration:.2f}s)]")

        if duration <= 0.0:
             logger.warning(f"Skipping playback of zero or invalid duration file: {filename}")
             return False, start_time, end_time, duration

        try:
            filepath_str = str(filename)
            if not Path(filepath_str).is_file():
                 logger.error(f"Audio file not found for playback: {filepath_str}")
                 return False, start_time, end_time, duration

            wave_obj = sa.WaveObject.from_wave_file(filepath_str)
            start_time = time.perf_counter()
            play_obj = wave_obj.play()
            play_obj.wait_done()
            end_time = time.perf_counter()
            logger.info(f"Finished playing {filename.name}")
            return True, start_time, end_time, duration
        except FileNotFoundError:
             logger.error(f"Audio file not found (simpleaudio): {filename}")
             return False, start_time, time.perf_counter(), duration
        except Exception as e:
            end_time = time.perf_counter()
            logger.error(f"Audio playback error for {filename}: {str(e)}", exc_info=True)
            return False, start_time, end_time, duration


    def transcribe_audio(self, filename, stt_client: WhisperSTTClient):
        """
        Transcribes the audio file using the local WhisperSTTClient.

        Args:
            filename (pathlib.Path): The full path to the audio file to transcribe.
            stt_client (WhisperSTTClient): The initialized Whisper STT client.

        Returns:
            tuple: (transcript, start_time, end_time)
                   transcript (str): The transcribed text, or "" on error.
                   start_time (float): Timestamp when transcription started.
                   end_time (float): Timestamp when transcription ended.
        """
        logger.info(f"Transcribing audio file with local Whisper: {filename.name}...")
        transcript = ""
        start_time = 0.0
        end_time = 0.0

        try:
            filepath_str = str(filename)
            if not Path(filepath_str).is_file():
                logger.error(f"Audio file not found for transcription: {filepath_str}")
                return "", start_time, end_time
            
            # Check if file is empty before passing to Whisper
            if Path(filepath_str).stat().st_size == 0:
                logger.warning(f"Audio file {filename.name} is empty. Skipping transcription.")
                return "", time.perf_counter(), time.perf_counter()

            start_time = time.perf_counter()
            transcript = stt_client.transcribe(filename) # Use the passed client instance
            end_time = time.perf_counter()

            if transcript:
                logger.info(f"Local Whisper transcription successful ({end_time - start_time:.2f}s): '{transcript}'")
            else:
                logger.warning("Local Whisper transcription returned no results or failed.")
            return transcript, start_time, end_time

        except Exception as e:
            end_time = time.perf_counter() if start_time > 0 else time.perf_counter()
            logger.error(f"Local Whisper transcription error for {filename}: {str(e)}", exc_info=True)
            return "", start_time, end_time
    def synthesize_speech(self, text, filename, tts_client: GoogleTTSClient):
        """
        Synthesizes speech from text using Google Cloud TTS.

        Args:
            text (str): The text to synthesize.
            filename (pathlib.Path): The full path to save the synthesized audio file.
            tts_client (GoogleTTSClient): The initialized Google TTS client.

        Returns:
            tuple: (success_flag, start_time, end_time, duration)
                   success_flag (bool): True if synthesis and saving were successful.
                   start_time (float): Timestamp when synthesis API call started.
                   end_time (float): Timestamp when synthesis API call ended.
                   duration (float): Duration of the synthesized audio in seconds.
        """
        logger.info(f"Synthesizing speech with Google TTS to {filename.name}...")
        start_time = 0.0
        end_time = 0.0
        duration = 0.0

        if not text:
            logger.warning("Skipping Google TTS for empty text.")
            return False, start_time, end_time, duration

        try:
            start_time = time.perf_counter()
            # Ensure the directory exists
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            duration = tts_client.synthesize(text, filename) # Use the passed client instance
            end_time = time.perf_counter()

            if duration > 0:
                logger.info(f"Google TTS synthesis successful ({end_time - start_time:.2f}s). Saved to {filename.name} ({duration:.2f}s)")
                return True, start_time, end_time, duration
            else:
                logger.error(f"Google TTS synthesis failed or produced zero duration audio for {filename.name}.")
                return False, start_time, end_time, duration

        except Exception as e:
            end_time = time.perf_counter() if start_time > 0 else time.perf_counter()
            logger.error(f"Google TTS synthesis error for {filename}: {str(e)}", exc_info=True)
            return False, start_time, end_time, duration


# ============================
# CONTEXT RETRIEVAL (Simple Keyword-Based)
# ============================
class ContextRetriever:
    """Handles retrieval of relevant context snippets using keyword matching."""
    def __init__(self):
        logger.info("Initializing Simple ContextRetriever...")
        self.context_list_store = []
        logger.info("Simple ContextRetriever initialized.")

    def initialize_index(self, scenario):
        """Initializes the context store for the given scenario's RAG context."""
        self.context_list_store = []
        rag_context_list = scenario.get('rag_context', [])

        if not isinstance(rag_context_list, list) or not rag_context_list:
            logger.warning("No valid RAG context list provided for scenario. RAG will be inactive.")
            return

        rag_context_list = [item for item in rag_context_list if isinstance(item, str) and item.strip()]
        if not rag_context_list:
             logger.warning("RAG context list is empty after filtering. RAG will be inactive.")
             return

        logger.info(f"Initializing simple context retriever with {len(rag_context_list)} context snippets...")
        try:
            start_time = time.perf_counter()
            self.context_list_store = rag_context_list
            end_time = time.perf_counter()
            logger.info(f"Context store initialized successfully. Total time: {end_time - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error initializing context store: {str(e)}", exc_info=True)
            self.context_list_store = []

    def retrieve_context(self, query, k=3):
        """Retrieves the top k relevant context snippets for a given query based on keyword matching."""
        start_time = time.perf_counter()
        retrieved_snippets = []
        source_context_list = self.context_list_store

        if not source_context_list:
            end_time = time.perf_counter()
            logger.debug(f"Context store not available, returning empty list.")
            return [], start_time, end_time

        if not query:
            logger.debug("Empty query for context retrieval, returning no snippets.")
            end_time = time.perf_counter()
            return [], start_time, end_time

        try:
            # Simple relevance scoring based on word overlap
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            if not query_words:
                # Return random selection if query has no words
                import random
                retrieved_snippets = random.sample(source_context_list, min(k, len(source_context_list)))
                logger.debug(f"No words in query, returning random selection.")
            else:
                # Score contexts based on word overlap
                scored_contexts = []
                for context in source_context_list:
                    context_words = set(re.findall(r'\b\w+\b', context.lower()))
                    overlap = len(query_words.intersection(context_words))
                    scored_contexts.append((overlap, context))
                
                # Sort by score (descending) and take top k
                scored_contexts.sort(reverse=True)
                retrieved_snippets = [ctx for _, ctx in scored_contexts[:k]]
                logger.debug(f"Retrieved {len(retrieved_snippets)} context snippets based on keyword matching.")
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            # Return some random snippets as fallback
            import random
            retrieved_snippets = random.sample(source_context_list, min(k, len(source_context_list)))

        end_time = time.perf_counter()
        logger.debug(f"Context retrieval total time: {end_time - start_time:.3f}s")
        return retrieved_snippets, start_time, end_time

# ============================
# EMOTION CONTROL (Simplified)
# ============================
class EmotionController:
    """Manages the simulated emotional state and corresponding TTS parameters."""
    def __init__(self, config):
        self.config = config
        self.progress_score = 0.0
        self.current_emotion = "Neutral" # Possible values: Neutral, Calm, Defiance
        self.base_pitch = config.BASE_PITCH # May not be used by local TTS
        self.base_rate = config.BASE_RATE   # May not be used by local TTS
        logger.info("EmotionController initialized (Simplified Model).")

    def analyze_conversation_flow(self, officer_input, scenario, llm_model: Gemma3Client):
        """Evaluates de-escalation, empathy, risk (1-5 each) using local LLM."""
        start_time = time.perf_counter()
        end_time = start_time
        composite_score = 0.0
        scenario_background = scenario.get('character_profile', scenario.get('detailed_context', 'No background provided.'))
        max_bg_len = 500
        if len(scenario_background) > max_bg_len:
             scenario_background = scenario_background[:max_bg_len] + "..."

        prompt = textwrap.dedent(f"""
            Analyze the following police officer's statement based on the scenario context.
            Evaluate ONLY these three factors, each on a scale of 1 (very negative) to 5 (very positive):
            1. De-escalation Effectiveness (1=Likely Escalates, 5=Very Calming)
            2. Empathy Shown (1=Dismissive, 5=Highly Empathetic)
            3. Risk Score (1=High Risk of Escalation, 5=Low Risk of Escalation)

            Scenario Context: {scenario_background}
            Officer's statement: "{officer_input}"

            Provide ONLY the three numerical scores separated by commas (e.g., 4,3,5).
            Do NOT include explanations, labels, or any other text.
            """)

        logger.debug(f"Analyzing conversation flow for: '{officer_input}'")
        try:
            llm_call_start_time = time.perf_counter()
            response_text = llm_model.generate(prompt) # Use local LLM
            end_time = time.perf_counter()
            logger.debug(f"LLM flow analysis raw response: '{response_text}' ({(end_time - llm_call_start_time):.2f}s)")

            numbers = re.findall(r'\b[1-5]\b', response_text)
            if len(numbers) == 3:
                factors_num = list(map(int, numbers))
                deescalation, empathy, risk_score = factors_num
                base_flow_score = (deescalation + empathy + risk_score) - 9
                logger.debug(f"Parsed flow factors: De-esc={deescalation}, Empathy={empathy}, RiskScore={risk_score}. Base score={base_flow_score}")
            else:
                logger.warning(f"Could not parse 3 scores (1-5) from flow analysis response: '{response_text}'. Found: {numbers}")
                base_flow_score = 0

            lower_input = officer_input.lower()
            success_phrases = [p for p in scenario.get('success_phrases', []) if isinstance(p, str)]
            failure_phrases = [p for p in scenario.get('failure_phrases', []) if isinstance(p, str)]
            success_score = sum(1 for p in success_phrases if re.search(r'\b' + re.escape(p.lower()) + r'\b', lower_input)) * 0.75
            failure_score = sum(1 for p in failure_phrases if re.search(r'\b' + re.escape(p.lower()) + r'\b', lower_input)) * (-1.0)
            logger.debug(f"Keyword scores: Success={success_score}, Failure={failure_score}")
            composite_score = (base_flow_score * 0.5) + success_score + failure_score
        except Exception as e:
            end_time = time.perf_counter()
            logger.error(f"Conversation flow analysis LLM error: {str(e)}", exc_info=True)
            composite_score = 0

        logger.info(f"Conversation flow analysis complete. Composite score: {composite_score:.2f}")
        return composite_score, start_time, end_time

    def update_emotion_state(self, officer_input, scenario, llm_model: Gemma3Client):
        """Updates progress score and emotion label based on flow analysis."""
        flow_score, start_time, end_time = self.analyze_conversation_flow(officer_input, scenario, llm_model)
        self.progress_score += flow_score
        success_threshold = scenario.get('progress_threshold', 5)
        failure_threshold = scenario.get('failure_threshold', -3)

        previous_emotion = self.current_emotion
        if self.progress_score >= success_threshold:
            self.current_emotion = "Calm"
        elif self.progress_score <= failure_threshold:
            self.current_emotion = "Defiance"
        else:
            self.current_emotion = "Neutral"

        if previous_emotion != self.current_emotion:
             logger.info(f"Emotion state changed from {previous_emotion} to {self.current_emotion}")
        logger.info(f"Updated emotion state: Score={self.progress_score:.2f}, Emotion={self.current_emotion}")
        return self.current_emotion, start_time, end_time
    
    def calculate_tts_parameters(self):
        """
        Calculates TTS speaking rate and pitch based on emotion.
        
        Returns:
            dict: TTS parameters (pitch, rate). Note: GoogleTTSClient uses
                  config values directly, this method is kept for potential future use.
        """
        speaking_rate = self.base_rate
        pitch = self.base_pitch

        if self.current_emotion == "Calm":
            speaking_rate = max(0.75, self.base_rate - 0.1)
            pitch = max(-4.0, self.base_pitch - 1.0)
        elif self.current_emotion == "Defiance":
            speaking_rate = min(1.75, self.base_rate + 0.2)
            pitch = min(4.0, self.base_pitch + 2.0)

        speaking_rate = max(0.75, min(speaking_rate, 1.75))
        pitch = max(-4.0, min(pitch, 4.0))
        params = {"speaking_rate": round(speaking_rate, 2), "pitch": round(pitch, 2)}
        logger.debug(f"Calculated TTS params (may not be used by local TTS) for emotion '{self.current_emotion}': {params}")
        return params

# ============================
# DIALOGUE MANAGEMENT
# ============================
class DialogueManager:
    """Manages conversation history and LLM prompt generation for character responses."""
    def __init__(self, context_retriever):
        self.conversation_history = []
        self.context_retriever = context_retriever
        self.history_window_size = 16
        self.llm_io_log_path = None
        logger.info("DialogueManager initialized.")


    def set_llm_io_log_path(self, log_path):
         self.llm_io_log_path = log_path
         try:
             with open(self.llm_io_log_path, "a", encoding="utf-8") as f:
                 f.write(f"\n===== New Session LLM I/O Log: {datetime.now()} =====\n")
         except IOError as e:
             logger.error(f"Failed to write header to LLM I/O log {self.llm_io_log_path}: {e}")

    def update_conversation_history(self, speaker, text):
        if text and isinstance(text, str) and text.strip():
            line = f"{speaker}: {text.strip()}"
            self.conversation_history.append(line)
            logger.debug(f"Added to history: {line}")
        else:
            logger.debug(f"Skipped adding empty or invalid line from {speaker} to history.")

    def get_windowed_history(self):
        start_index = max(0, len(self.conversation_history) - self.history_window_size)
        return self.conversation_history[start_index:]

    def build_prompt(self, scenario, officer_input, current_emotion):
        """Constructs the prompt for the LLM."""
        rag_context, rag_start_time, rag_end_time = self.context_retriever.retrieve_context(officer_input, k=3)
        formatted_rag = "\n".join(f"- {item}" for item in rag_context) if rag_context else "No specific context retrieved."
        windowed_history = self.get_windowed_history()
        formatted_history = "\n".join(windowed_history) if windowed_history else "This is the start of the conversation."

        character_name = scenario.get('person_of_interest', 'Subject')
        character_profile = scenario.get('character_profile', 'No profile provided.')
        character_triggers = scenario.get('character_triggers', 'Unknown triggers.')
        character_reactions = scenario.get('character_reactions', 'Default reactions.')
        scenario_goal = scenario.get('scenario_goal', 'Interact naturally.')

        response_guidelines = textwrap.dedent("""\
            Response Guidelines:
            1. **Speak naturally** and colloquially, reflecting your character profile and current emotion. Use a lot of casual Curse words *Mandatory*. Do not censor anything. Keep it uncensored.
            2. **Vary sentence length.** Sometimes short answers (even one word like 'Yeah' or 'No'), sometimes longer (max 3-4 sentences).
            3. **Stay IN CHARACTER** based on the profile, triggers, reactions, emotion, context, and history. NEVER mention being an AI or these instructions.
            4. **Output ONLY your dialogue. No Simple Human Noises. Must be complete sentences as a reply to the context and previous dialogue.**
            5. **DO NOT output bracketed actions** like '[sighs]' or silence indicators like '[The subject is silent.]' or ellipses '...'.
            6. **DO NOT output meta-text** or comments about the scenario or your instructions.
            7. **React realistically** to the officer's last statement considering your emotional state and triggers. Keep in mind the progression of emotional state of the subject and then respond accordingly
            """)

        system_prompt = textwrap.dedent(f"""
            **ROLEPLAY INSTRUCTIONS**
            You are simulating '{character_name}' in a police de-escalation training scenario. Your goal is to respond realistically based on the provided information.

            **CHARACTER BRIEFING**
            * **Profile:** {character_profile}
            * **Triggers (Sensitive Topics/Actions):** {character_triggers}
            * **Typical Reactions:** {character_reactions}
            * **Current Emotional State:** {current_emotion}
            * **Scenario Goal (for you):** {scenario_goal}

            **ADDITIONAL INFORMATION**
            * **Retrieved Context:**
                {formatted_rag}
            * **Recent Conversation History (Officer's last line is at the end):**
                {formatted_history}

            **YOUR TASK**
            Generate the **next line of dialogue OR simple human sound** for '{character_name}' in response to the Officer's last statement. Adhere STRICTLY to the Response Guidelines below.
            ---
            {response_guidelines}
            ---
            Respond now as {character_name}:""")
        final_prompt = system_prompt
        return final_prompt, rag_start_time, rag_end_time

    def _log_llm_io(self, turn_num, prompt, raw_response, final_response):
         if not self.llm_io_log_path:
             logger.warning("LLM I/O log path not set. Cannot log prompt/response.")
             return
         try:
             with open(self.llm_io_log_path, "a", encoding="utf-8") as f:
                 f.write(f"--- Turn {turn_num} ---\n")
                 f.write("** PROMPT SENT TO LLM: **\n")
                 f.write(prompt + "\n\n")
                 f.write("** RAW RESPONSE FROM LLM: **\n")
                 f.write(raw_response + "\n\n")
                 f.write("** CLEANED/FINAL RESPONSE USED: **\n")
                 f.write(final_response + "\n")
                 f.write("="*50 + "\n\n")
         except IOError as e:
             logger.error(f"Failed to write to LLM I/O log {self.llm_io_log_path}: {e}")

    def generate_response(self, turn_num, officer_input, scenario, current_emotion, llm_model: Gemma3Client):
        """Generates AI character's response using the local LLM."""
        llm_start_time = 0.0
        llm_end_time = 0.0
        raw_text = "[LLM Response Not Captured]"
        final_response_text = "[ERROR: Could not generate response]"

        try:
            prompt_text, rag_start_time, rag_end_time = self.build_prompt(scenario, officer_input, current_emotion)
            logger.debug(f"Generated LLM prompt length: {len(prompt_text)} characters")
        except Exception as build_prompt_error:
             logger.error(f"Error building prompt: {build_prompt_error}", exc_info=True)
             self._log_llm_io(turn_num, "[Error Building Prompt]", raw_text, final_response_text)
             return final_response_text, 0.0, 0.0, 0.0, 0.0

        logger.info("Generating response with local LLM...")
        try:
            llm_start_time = time.perf_counter()
            raw_text = llm_model.generate(prompt_text) # Use local LLM
            llm_end_time = time.perf_counter()
            logger.debug(f"LLM Raw Response ({llm_end_time - llm_start_time:.2f}s):\n{raw_text}")

            clean_text = raw_text
            character_name = scenario.get('person_of_interest', 'Subject')
            pattern = re.compile(rf"^\s*{re.escape(character_name)}\s*[:\-]\s*", re.IGNORECASE)
            clean_text = pattern.sub("", clean_text).strip()
            clean_text = re.sub(r'\*[^*]*\*', '', clean_text)
            clean_text = re.sub(r'\([^)]*\)', '', clean_text)
            clean_text = re.sub(r'\[[^\]]*\]', '', clean_text)
            clean_text = re.sub(r'\{[^}]*\}', '', clean_text)
            clean_text = re.sub(r'<[^>]*>', '', clean_text)

            refusal_patterns = [
                r"i cannot fulfill this request", r"i am unable to", r"i cannot provide",
                r"as an ai", r"my instructions", r"response guidelines"
            ]
            for pattern_text in refusal_patterns: # Renamed pattern to pattern_text to avoid conflict
                if re.search(pattern_text, clean_text, re.IGNORECASE):
                    logger.warning(f"LLM response contained meta-comment/refusal pattern: '{pattern_text}'. Original: '{raw_text}'")
                    clean_text = "Um..."
                    break
            clean_text = ' '.join(clean_text.split()).strip()
            final_response_text = clean_text if clean_text else "Uh..."
            logger.info(f"LLM Cleaned Response: '{final_response_text}'")
        except Exception as e:
            if llm_start_time > 0:
                 llm_end_time = time.perf_counter()
            logger.error(f"LLM response generation/processing error: {str(e)}", exc_info=True)
            final_response_text = "[ERROR: Technical difficulty generating response]"
            self._log_llm_io(turn_num, prompt_text, f"[ERROR: {e}]", final_response_text)
            return final_response_text, llm_start_time, llm_end_time, rag_start_time, rag_end_time

        self._log_llm_io(turn_num, prompt_text, raw_text, final_response_text)
        return final_response_text, llm_start_time, llm_end_time, rag_start_time, rag_end_time

# ============================
# SESSION MANAGEMENT
# ============================
class SessionManager:
    """Orchestrates the de-escalation training session with local models."""
    def __init__(self, config: Config, stt_client: WhisperSTTClient, tts_client: GoogleTTSClient, llm_model: Gemma3Client):
        self.config = config
        self.stt_client = stt_client
        self.tts_client = tts_client
        self.llm_model = llm_model

        self.audio_handler = AudioHandler(config, stt_client, tts_client)
        self.context_retriever = ContextRetriever()
        self.emotion_controller = EmotionController(config)
        self.dialogue_manager = DialogueManager(self.context_retriever)

        self.session_id = None
        self.session_dir = None
        self.turn_counter = 0
        self.latency_log = []
        self.current_scenario = None
        logger.info("SessionManager initialized with local clients.")

    def _create_session(self, scenario_choice_num, scenario_data):
        self.current_scenario = scenario_data
        scenario_name_safe = re.sub(r'[^\w\-]+', '_', scenario_data.get('name', f'scenario_{scenario_choice_num}'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{scenario_name_safe}_{timestamp}"
        self.session_dir = self.config.SESSION_DATA_DIR / self.session_id
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create session directory {self.session_dir}: {e}")
             raise

        self.turn_counter = 0
        self.latency_log = []
        self.emotion_controller.progress_score = 0.0
        self.emotion_controller.current_emotion = "Neutral"
        self.dialogue_manager.conversation_history.clear()
        llm_log_path = self.session_dir / f"{self.session_id}_llm_io.log"
        self.dialogue_manager.set_llm_io_log_path(llm_log_path)
        self.context_retriever.initialize_index(scenario_data)
        logger.info(f"Created new session: {self.session_id}")
        logger.info(f"Session data directory: {self.session_dir}")

    def select_scenario(self, scenarios):
        print("\n" + "="*20 + " Scenario Selection " + "="*20)
        for num in sorted(scenarios.keys()):
            details = scenarios[num]
            scenario_name = details.get('name', f"Unknown Scenario {num}")
            print(f"{num}. {scenario_name}")
        print("="*62)
        while True:
            try:
                choice_str = input("Choose scenario number: ")
                choice = int(choice_str)
                if choice in scenarios:
                    self._create_session(choice, scenarios[choice])
                    return scenarios[choice]
                else:
                    print(f"Invalid choice '{choice}'. Please enter a number from the list.")
            except ValueError:
                print(f"Invalid input '{choice_str}'. Please enter a number.")
            except Exception as e:
                 logger.error(f"Error during scenario selection: {e}", exc_info=True)
                 print("An unexpected error occurred during selection.")

    def play_scenario_setup(self):
        if not self.current_scenario or not self.session_dir:
             logger.error("Cannot play setup, session not properly initialized.")
             return
        setup_text = self.current_scenario.get('setup', '')
        if not setup_text:
            logger.warning("No setup text found for this scenario.")
            return

        setup_audio_file = self.session_dir / "audio_setup.wav"
        self.emotion_controller.progress_score = 0.0
        self.emotion_controller.current_emotion = "Neutral"
        # TTS params from emotion controller are not directly used by GoogleTTSClient.synthesize
        # self.emotion_controller.calculate_tts_parameters() # Call if it has other side effects

        logger.info("Synthesizing scenario setup with local TTS...")
        tts_success, _, _, tts_duration = self.audio_handler.synthesize_speech(
            setup_text, setup_audio_file, self.tts_client
        )

        if tts_success and tts_duration > 0:
            print("\n--- Scenario Setup ---")
            print(textwrap.fill(setup_text, width=80))
            print("----------------------")
            play_success, _, _, _ = self.audio_handler.play_audio(setup_audio_file)
            if not play_success:
                 logger.error("Failed to play scenario setup audio after synthesis.")
        else:
            logger.error(f"Failed to synthesize setup audio (Success: {tts_success}, Duration: {tts_duration}). Displaying text only.")
            print("\n--- Scenario Setup (Text Only) ---")
            print(textwrap.fill(setup_text, width=80))
            print("------------------------------------")
        try:
             input("\nPress ENTER when ready to begin the interaction...")
        except EOFError:
             logger.warning("EOF detected while waiting for user readiness. Attempting to proceed.")

    def _log_turn_latency(self, turn_data):
        expected_keys = [
            'turn', 'rec_start', 'rec_end', 'rec_duration',
            'stt_start', 'stt_end', 'stt_duration',
            'emo_analyze_start', 'emo_analyze_end', 'emo_analyze_duration',
            'current_emotion', 'progress_score',
            'rag_start', 'rag_end', 'rag_duration',
            'llm_start', 'llm_end', 'llm_duration',
            'tts_start', 'tts_end', 'tts_duration', 'tts_audio_duration',
            'play_start', 'play_end',
            'total_response_latency', 'processing_latency'
        ]
        log_entry = {'turn': self.turn_counter}
        for key in expected_keys:
             default_val = 0.0 if 'time' in key or 'duration' in key or 'score' in key else ""
             if key == 'progress_score':
                 val = turn_data.get(key, 0.0)
                 log_entry[key] = val if isinstance(val, (int, float)) else 0.0
             else:
                 log_entry[key] = turn_data.get(key, default_val)
        self.latency_log.append(log_entry)
        try:
            rec_dur_str = f"{log_entry['rec_duration']:.2f}" if isinstance(log_entry['rec_duration'], float) else str(log_entry['rec_duration'])
            stt_dur_str = f"{log_entry['stt_duration']:.2f}" if isinstance(log_entry['stt_duration'], float) else str(log_entry['stt_duration'])
            emo_dur_str = f"{log_entry['emo_analyze_duration']:.2f}" if isinstance(log_entry['emo_analyze_duration'], float) else str(log_entry['emo_analyze_duration'])
            score_str = f"{log_entry['progress_score']:.2f}" if isinstance(log_entry['progress_score'], (int, float)) else str(log_entry['progress_score'])
            rag_dur_str = f"{log_entry['rag_duration']:.2f}" if isinstance(log_entry['rag_duration'], float) else str(log_entry['rag_duration'])
            llm_dur_str = f"{log_entry['llm_duration']:.2f}" if isinstance(log_entry['llm_duration'], float) else str(log_entry['llm_duration'])
            tts_dur_str = f"{log_entry['tts_duration']:.2f}" if isinstance(log_entry['tts_duration'], float) else str(log_entry['tts_duration'])
            tts_audio_dur_str = f"{log_entry['tts_audio_duration']:.2f}" if isinstance(log_entry['tts_audio_duration'], float) else str(log_entry['tts_audio_duration'])
            total_resp_str = f"{log_entry['total_response_latency']:.2f}" if isinstance(log_entry['total_response_latency'], float) else str(log_entry['total_response_latency'])

            logger.info(
                 f"Turn {int(log_entry['turn'])} Latency (s): "
                 f"Rec={rec_dur_str}, STT={stt_dur_str}, Emo={emo_dur_str} ({log_entry['current_emotion']}, {score_str}), "
                 f"RAG={rag_dur_str}, LLM={llm_dur_str}, TTS={tts_dur_str} (Audio={tts_audio_dur_str}), TotalResp={total_resp_str}"
            )
        except (TypeError, ValueError, KeyError) as fmt_err:
             logger.error(f"Error formatting latency log message for turn {log_entry.get('turn', 'N/A')}: {fmt_err}. Raw data: {log_entry}")

    def main_loop(self):
        if not self.current_scenario or not self.session_dir or not self.dialogue_manager:
             logger.error("Session not properly initialized. Cannot start main loop.")
             return

        scenario_name = self.current_scenario.get('name', 'Scenario')
        character_name = self.current_scenario.get('person_of_interest', 'Subject')
        logger.info(f"Starting main loop for scenario: {scenario_name}")
        print("\n" + "="*20 + f" Scenario: {scenario_name} " + "="*20)
        print(f"You are speaking with: {character_name}")
        print("Say 'STOP SCENARIO' to end the session.")
        print("="*(42 + len(scenario_name)))

        self.turn_counter = 0
        while True:
            current_turn_attempt = self.turn_counter + 1
            logger.info(f"--- Turn {current_turn_attempt} ---")
            turn_latency = {'turn': current_turn_attempt}

            input_audio_file = self.session_dir / f"turn_{current_turn_attempt:03d}_input.wav"
            rec_success, rec_start, rec_end, rec_duration = self.audio_handler.record_audio(input_audio_file)
            turn_latency['rec_start'], turn_latency['rec_end'], turn_latency['rec_duration'] = rec_start, rec_end, rec_duration

            min_meaningful_duration = 0.2
            if not rec_success or rec_duration < min_meaningful_duration:
                print("[Recording failed or too short. Please try speaking again.]" if not rec_success else "[Recording very short. Speak clearly.]")
                turn_latency['current_emotion'] = self.emotion_controller.current_emotion
                turn_latency['progress_score'] = self.emotion_controller.progress_score
                time.sleep(1)
                continue

            transcript, stt_start, stt_end = self.audio_handler.transcribe_audio(input_audio_file, self.stt_client)
            turn_latency['stt_start'], turn_latency['stt_end'] = stt_start, stt_end
            turn_latency['stt_duration'] = stt_end - stt_start if stt_start > 0 else 0.0

            if not transcript:
                print("[No speech detected or transcription failed. Try again or say 'STOP SCENARIO'.]")
                turn_latency['current_emotion'] = self.emotion_controller.current_emotion
                turn_latency['progress_score'] = self.emotion_controller.progress_score
                continue

            self.turn_counter = current_turn_attempt
            turn_latency['turn'] = self.turn_counter

            self.dialogue_manager.update_conversation_history("Officer", transcript)
            print(f"\nTurn {self.turn_counter} - Officer ({rec_duration:.2f}s):")
            print(textwrap.fill(transcript, width=70, initial_indent="  ", subsequent_indent="  "))

            if "stop scenario" in transcript.lower():
                logger.info("'Stop scenario' command detected. Ending session.")
                print("\n[Stopping Scenario...]")
                turn_latency['current_emotion'] = self.emotion_controller.current_emotion
                turn_latency['progress_score'] = self.emotion_controller.progress_score
                self._log_turn_latency(turn_latency)
                break

            current_emotion, emo_analyze_start, emo_analyze_end = self.emotion_controller.update_emotion_state(
                transcript, self.current_scenario, self.llm_model
            )
            turn_latency['emo_analyze_start'], turn_latency['emo_analyze_end'] = emo_analyze_start, emo_analyze_end
            turn_latency['emo_analyze_duration'] = emo_analyze_end - emo_analyze_start if emo_analyze_start > 0 else 0.0
            turn_latency['current_emotion'], turn_latency['progress_score'] = current_emotion, self.emotion_controller.progress_score
            print(f"  [AI Emotion: {current_emotion} | Progress: {self.emotion_controller.progress_score:.2f}]")

            response_text, llm_start, llm_end, rag_start, rag_end = self.dialogue_manager.generate_response(
                self.turn_counter, transcript, self.current_scenario, current_emotion, self.llm_model
            )
            turn_latency['rag_start'], turn_latency['rag_end'] = rag_start, rag_end
            turn_latency['rag_duration'] = rag_end - rag_start if rag_start > 0 else 0.0
            turn_latency['llm_start'], turn_latency['llm_end'] = llm_start, llm_end
            turn_latency['llm_duration'] = llm_end - llm_start if llm_start > 0 else 0.0

            self.dialogue_manager.update_conversation_history(character_name, response_text)
            print(f"\nTurn {self.turn_counter} - {character_name}:")
            print(textwrap.fill(response_text, width=70, initial_indent="  ", subsequent_indent="  "))

            output_audio_file = self.session_dir / f"turn_{self.turn_counter:03d}_output.wav"
            # TTS params from emotion controller are not directly used by current GoogleTTSClient
            # self.emotion_controller.calculate_tts_parameters() # Call if it has other side effects
            tts_success, tts_start, tts_end, tts_audio_duration = self.audio_handler.synthesize_speech(
                response_text, output_audio_file, self.tts_client
            )
            turn_latency['tts_start'], turn_latency['tts_end'] = tts_start, tts_end
            turn_latency['tts_duration'] = tts_end - tts_start if tts_start > 0 else 0.0
            turn_latency['tts_audio_duration'] = tts_audio_duration

            if tts_success and tts_audio_duration > 0:
                 play_success, play_start, play_end, _ = self.audio_handler.play_audio(output_audio_file)
                 turn_latency['play_start'], turn_latency['play_end'] = play_start, play_end
            else:
                 logger.warning(f"Skipping playback for turn {self.turn_counter} due to TTS failure or zero duration.")
                 turn_latency['play_start'], turn_latency['play_end'] = 0.0, 0.0

            if rec_end > 0 and turn_latency.get('play_start', 0.0) > 0:
                 turn_latency['total_response_latency'] = turn_latency['play_start'] - rec_end
                 turn_latency['processing_latency'] = turn_latency['play_start'] - rec_end
            else:
                 turn_latency['total_response_latency'], turn_latency['processing_latency'] = -1.0, -1.0
            self._log_turn_latency(turn_latency)

            success_threshold = self.current_scenario.get('progress_threshold', 5)
            failure_threshold = self.current_scenario.get('failure_threshold', -3)
            end_message, end_audio_file, should_break = None, None, False

            if self.emotion_controller.progress_score >= success_threshold:
                end_message, end_audio_file = "Situation appears stabilized. Good progress.", self.session_dir / "audio_success.wav"
                logger.info(f"Success threshold ({success_threshold}) reached.")
                should_break = True
            elif self.emotion_controller.progress_score <= failure_threshold:
                end_message, end_audio_file = "Situation has escalated significantly!", self.session_dir / "audio_failure.wav"
                logger.warning(f"Failure threshold ({failure_threshold}) reached.")
                should_break = True

            if should_break and end_message and end_audio_file:
                print(f"\n[{end_message}]")
                tts_success_end, _, _, dur_end = self.audio_handler.synthesize_speech(
                    end_message, end_audio_file, self.tts_client
                )
                if tts_success_end and dur_end > 0:
                    self.audio_handler.play_audio(end_audio_file)
                break
        logger.info(f"Main loop finished for session {self.session_id}.")
        self.save_session()

    def analyze_performance(self):
        if not self.current_scenario or not self.dialogue_manager:
            logger.error("Cannot analyze performance, session or dialogue manager not initialized.")
            return "Error: Session or dialogue manager not initialized."
        if not self.dialogue_manager.conversation_history:
             logger.warning("Cannot analyze performance, conversation history is empty.")
             return "No conversation recorded to analyze."

        logger.info("Generating performance analysis with local LLM...")
        full_history = "\n".join(self.dialogue_manager.conversation_history)
        scenario_name = self.current_scenario.get('name', 'this scenario')
        character_profile = self.current_scenario.get('character_profile', 'Not specified.')
        scenario_goal = self.current_scenario.get('scenario_goal', 'Not specified.')

        prompt_text = textwrap.dedent(f"""
            **Task:** Analyze the police officer's performance in the following de-escalation training scenario simulation transcript.

            **Scenario Context:**
            * Name: {scenario_name}
            * Goal: {scenario_goal}
            * Subject Profile: {character_profile}

            **Conversation Transcript:**
            --- START TRANSCRIPT ---
            {full_history}
            --- END TRANSCRIPT ---

            **Analysis Instructions:**
            Provide concise feedback (around 100-120 words maximum) for the trainee officer. Focus on:
            1.  **Communication Effectiveness:** Clarity, tone, active listening demonstrated.
            2.  **Empathy:** Was the subject's perspective acknowledged appropriately?
            3.  **De-escalation Techniques:** What specific techniques were used effectively or missed? Consider techniques relevant to *this* scenario.
            4.  **Overall Outcome:** Briefly comment on whether the interaction moved towards successful de-escalation or escalated.

            **Output Format:**
            * Write the entire feedback as a single block of plain text (a paragraph or multiple connected paragraphs).
            * **DO NOT use markdown formatting** like bolding, italics, or bullet points.
            * Start with a brief overall summary, then mention strengths and areas for improvement.
            * Use clear, constructive language suitable for training. Avoid jargon.
            * **Strictly adhere to the word count limit (approx 100-120 words).**
            """)
        try:
            start_time = time.perf_counter()
            analysis_text = self.llm_model.generate(prompt_text) # Use local LLM
            end_time = time.perf_counter()
            analysis_text = analysis_text.replace("**", "").replace("*", "") # Basic markdown removal
            logger.info(f"Performance analysis generated ({end_time - start_time:.2f}s).")
            return analysis_text
        except Exception as e:
            logger.error(f"Performance analysis LLM error: {str(e)}", exc_info=True)
            return f"[Error generating performance analysis: {str(e)}]"

    def save_session(self):
        if not self.current_scenario or not self.session_dir or not self.dialogue_manager:
             logger.error("Cannot save session, critical components missing.")
             return
        logger.info(f"Saving session data for {self.session_id}...")
        try:
            transcript_file = self.session_dir / f"{self.session_id}_transcript.txt"
            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(f"==== Scenario: {self.current_scenario.get('name', 'Unknown')} ====\n")
                    f.write(f"==== Session ID: {self.session_id} ====\n\n")
                    if self.dialogue_manager.conversation_history:
                        f.write("\n".join(self.dialogue_manager.conversation_history))
                    else:
                        f.write("[No conversation recorded]")
                logger.info(f"Transcript saved: {transcript_file.name}")
            except IOError as e:
                 logger.error(f"Failed to save transcript file {transcript_file}: {e}")

            analysis_text = "[Analysis skipped: No conversation history]"
            if self.dialogue_manager.conversation_history:
                 analysis_text = self.analyze_performance()
            else:
                 logger.warning("Skipping performance analysis due to empty history.")
            analysis_file = self.session_dir / f"{self.session_id}_analysis.txt"
            try:
                with open(analysis_file, "w", encoding="utf-8") as f:
                    f.write(f"==== Performance Analysis for Session: {self.session_id} ====\n\n")
                    f.write(analysis_text)
                logger.info(f"Analysis saved: {analysis_file.name}")
            except IOError as e:
                 logger.error(f"Failed to save analysis file {analysis_file}: {e}")

            latency_file = self.session_dir / f"{self.session_id}_latency.csv"
            if self.latency_log:
                try:
                    expected_keys = [
                         'turn', 'rec_start', 'rec_end', 'rec_duration',
                         'stt_start', 'stt_end', 'stt_duration',
                         'emo_analyze_start', 'emo_analyze_end', 'emo_analyze_duration',
                         'current_emotion', 'progress_score',
                         'rag_start', 'rag_end', 'rag_duration',
                         'llm_start', 'llm_end', 'llm_duration',
                         'tts_start', 'tts_end', 'tts_duration', 'tts_audio_duration',
                         'play_start', 'play_end',
                         'total_response_latency', 'processing_latency'
                    ]
                    first_valid_entry = next((item for item in self.latency_log if item), None)
                    if first_valid_entry:
                        actual_headers = list(first_valid_entry.keys())
                        headers = [k for k in expected_keys if k in actual_headers]
                        extra_keys = [k for k in actual_headers if k not in expected_keys]
                        headers.extend(extra_keys)
                        if extra_keys: logger.warning(f"Found unexpected keys in latency data: {extra_keys}")
                        with open(latency_file, "w", newline='', encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                            writer.writeheader()
                            valid_rows = [row for row in self.latency_log if isinstance(row, dict)]
                            writer.writerows(valid_rows)
                        logger.info(f"Latency log saved: {latency_file.name}")
                    else: logger.warning("Latency log contains no valid entries, skipping save.")
                except (IOError, IndexError, KeyError) as e:
                     logger.error(f"Failed to save latency log {latency_file}: {e}", exc_info=True)
            else: logger.warning("Latency log is empty, skipping save.")

            play_analysis = True
            should_play = (play_analysis and analysis_text and not analysis_text.startswith("[Error") and
                           not analysis_text.startswith("[Analysis skipped") and not analysis_text.startswith("[Analysis blocked"))
            if should_play:
                analysis_audio_file = self.session_dir / "audio_analysis.wav"
                max_analysis_len = 1000
                analysis_tts_input = "Performance analysis summary: " + analysis_text[:max_analysis_len]
                if len(analysis_text) > max_analysis_len: analysis_tts_input += "... Analysis truncated for audio playback."
                
                tts_success_analysis, _, _, dur_analysis = self.audio_handler.synthesize_speech(
                    analysis_tts_input, analysis_audio_file, self.tts_client
                )
                if tts_success_analysis and dur_analysis > 0:
                    print("\n--- Performance Analysis Summary ---")
                    print(textwrap.fill(analysis_text, width=80))
                    print("------------------------------------")
                    logger.info("Playing performance analysis audio...")
                    self.audio_handler.play_audio(analysis_audio_file)
                else:
                    logger.error("Failed to synthesize or play performance analysis. Displaying text only.")
                    if not (tts_success_analysis and dur_analysis > 0):
                         print("\n--- Performance Analysis Summary (Text Only) ---")
                         print(textwrap.fill(analysis_text, width=80))
                         print("----------------------------------------------")
        except Exception as e:
            logger.error(f"Unexpected error during session saving: {str(e)}", exc_info=True)

# ============================
# MAIN EXECUTION BLOCK
# ============================
if __name__ == "__main__":
    print("\n" + "="*20 + " Police De-escalation Trainer (Local Models) " + "="*20)

    # --- Initialize Configuration ---
    config = Config()
    logger.info("Configuration loaded.")    # --- Initialize Local AI Clients ---
    logger.info("Initializing AI clients...")
    try:
        stt_client = WhisperSTTClient(model_path="base") # Using "base" model
        ollama_client = Gemma3Client() # Uses default gemma3:4b
        
        # Initialize Google TTS client
        try:
            tts_client = GoogleTTSClient(config)
            logger.info("Google TTS client initialized successfully.")
        except Exception as tts_e:
            logger.error(f"Failed to initialize Google TTS client: {tts_e}")
            logger.error("Ensure Google Cloud credentials are properly configured.")
            sys.exit(1)        
        audio_handler = AudioHandler(config, stt_client, tts_client)
        context_retriever = ContextRetriever() # Simple keyword-based
        emotion_controller = EmotionController(config)
        dialogue_manager = DialogueManager(context_retriever) # Pass retriever
        
        logger.info("AI clients initialized successfully.")
    except Exception as client_error:
        logger.error(f"FATAL: Failed to initialize local AI clients: {client_error}", exc_info=True)
        logger.error("Please check model paths, installations (Whisper, Ollama, Coqui TTS), and ensure Ollama server is running.")
        sys.exit(1) # Exit if core components fail

    # --- Initialize Session Manager ---
    logger.info("Initializing SessionManager...")
    try:
        # Ensure ollama_client (which is our llm_model) is passed to SessionManager
        session_mgr = SessionManager(config, stt_client, tts_client, ollama_client)
        logger.info("SessionManager initialized successfully.")
    except Exception as sm_error:
        logger.error(f"FATAL: Failed to initialize SessionManager: {sm_error}", exc_info=True)
        sys.exit(1)

    # --- Main Application Loop ---
    SCENARIOS = {
        1: {
            "name": "Mike's Bad Day - Suicide-by-Cop",
            "setup": textwrap.dedent("""\
                Units respond to River Bottoms Pub, reference 10-56 (Suicide Attempt/Threat). Caller, the bartender, advises
                an intoxicated male subject, Mike, stated he has a gun in his coat pocket and wants to die.
                Caller is only other person inside. Subject reportedly upset over wife leaving him. Proceed with caution.
            """),
            "character_profile": textwrap.dedent("""\
                Mike, 42, construction worker. Recently lost job, wife left 3 weeks ago. Depressed, drinking heavily.
                States he has a gun and wants to die ('suicide by cop' risk). Can be talkative/cooperative
                if officer shows empathy and avoids triggers. Cusses almost every sentence. Likes cars and action movies. Becomes angry/hostile
                if communication is poor, authoritative, or focuses repeatedly on wife/job loss.
                Currently inside the 'River Bottoms' pub. Intoxicated.
            """),
            "character_triggers": "Directly asking about/dwelling on wife or job loss, loud/authoritative commands, dismissive language ('calm down', 'it's not worth it'), focusing on the gun prematurely.",
            "character_reactions": "Initial reluctance/defiance ('leave me alone'). Can escalate to anger, threats (hostage, shooting self/officers) if poorly handled. Can become cooperative/compliant if rapport is built through active listening and empathy.",
            "scenario_goal": "Build rapport, ensure safety (hands away from pocket), and arrange for mental health evaluation (mental APP). Avoid escalating to violence. Manage the bartender's safety.",
            "rag_context": [
                "Mike has a history of depression but no prior violent incidents.",
                "The bartender is the only other person in the bar unless told to leave.",
                "Mike mentioned the gun is in his right front coat pocket.",
                "Mike hasn't explicitly threatened the bartender yet, but might if pressured.",
                "The pub has a pool table, bar, jukebox, dartboard.",
                "Showing empathy about his 'tough time' without dwelling on specifics is often effective.",
                "Avoid telling him 'it's not worth it' or 'you'll find another job/wife'.",
                "He might respond positively to distraction topics like cars or movies once initial contact is made."
            ],
            "voice_name": "en-US-Studio-Q", # Kept for reference, configurable in GoogleTTSClient
            "progress_threshold": 5, "failure_threshold": -4,
            "success_phrases": ["hands visible", "stay safe", "professional help", "work together", "i hear you", "talk to me", "i understand", "must be tough", "keep your hands away", "let's talk about something else", "i'm here to listen", "we can figure this out"],
            "failure_phrases": ["calm down", "drop the gun", "show me the gun", "put the gun down", "job", "wife", "why", "it's not that big a deal", "you need to", "plenty of fish", "new job", "counseling", "snap out of it"],
            "person_of_interest": "Mike",
        },
        2: {
            "name": "Suicidal Subject at City Bench - Jenny",
            "setup": textwrap.dedent("""\
                Units respond to Heim Barbeque on Magnolia, reference 10-56 (Suicide Attempt/Threat).
                Caller reports a female subject sitting on a bench outside, sobbing, holding a kitchen knife to her throat.
                Location is crowded with lunchtime patrons. Use caution approaching.
            """),
            "character_profile": "Jenny, distraught, holding kitchen knife to throat in public. Turning 40, recent breakup (boyfriend left for best friend), job loss imminent. Highly emotional, feeling hopeless, initially wants to be left alone. Risk of self-harm or charging if approached improperly.",
            "character_triggers": "Repeatedly mentioning her specific problems (age, boyfriend, job), dismissive language ('calm down', 'overreacting', 'ridiculous'), perceived aggression or lack of safety (officer too close without cover).",
            "character_reactions": "Initial defiance ('Leave me alone!'). High risk of escalation (charging officer or self-harm) if triggers hit or officer safety (D+C=T) neglected. Can calm down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.",
            "scenario_goal": "Ensure Jacob's safety (prevent running into traffic, self-harm), calm him down using gentle, quiet communication, build trust, locate his mother.",
            "rag_context": ["Jenny is turning 40 today.", "Her boyfriend left her yesterday for her best friend.", "She found out she is being laid off next week.", "The location (Heim BBQ) is crowded with patrons.", "Officer must consider Distance + Cover = Time (D+C=T).", "Moving the knife away from her throat is a key de-escalation sign.", "Active listening and genuine empathy are crucial.", "Avoid minimizing her problems or offering quick fixes."],
            "voice_name": "en-US-Studio-O", # Kept for reference

            "progress_threshold": 5, "failure_threshold": -4,
            "success_phrases": ["put the knife down", "we can get you help", "talk to me", "safe place", "i'm listening", "i hear you", "must be hard", "i want to help", "let's talk"],
            "failure_phrases": ["calm down", "snap out of it", "you're overreacting", "drop it", "you're being ridiculous", "drop the knife", "don't be stupid"],
            "person_of_interest": "Jenny",
        },
        3: {
            "name": "Teen Behaving Erratically - Jacob",
            "setup": textwrap.dedent("""\
                Units respond to the Fort Worth Science Museum, reference Disturbance.
                Caller reports a teenage male acting erratically near the entrance - yelling at himself,
                swinging arms, slapping himself. Crowd gathering, caller concerned subject may harm self or run into traffic. No weapons seen.
            """),
            "character_profile": "Jacob, teenager, exhibiting erratic behavior (yelping, arm flailing, self-slapping) likely due to distress. Possibly on the autism spectrum (stimming). Became separated from his mother in the museum, now scared, anxious, and overwhelmed by the crowd and noise.",
            "character_triggers": "Loud authoritative commands, yelling, direct physical approach without rapport, engaging with the mocking crowd instead of him, dismissive language ('stop being weird').",
            "character_reactions": "Escalates erratic behavior (more flailing, screaming, covering ears) if met with loud commands or aggression. May attempt to run away (potentially towards danger like a roadway) if overwhelmed. Calms down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.",
            "scenario_goal": "Ensure Jacob's safety (prevent running into traffic, self-harm), calm him down using gentle, quiet communication, build trust, locate his mother.",
            "rag_context": ["Jacob is yelling 'Where's Mom? I lost Mom!' between yelps.", "The behavior (slapping, yelping) is likely 'stimming' to self-soothe anxiety.", "Crowd members are laughing, filming, some yelling 'Taze him!'.", "He was last seen with his mother at the 'Dino-Dig' exhibit.", "Maintaining distance initially is wise until confirmed unarmed and rapport is built.", "A calm, quiet, patient tone is essential.", "Avoid direct, sustained eye contact initially if he seems overwhelmed.", "Reassure him about finding his mother."],
            "voice_name": "en-US-Wavenet-D", # Kept for reference
            "progress_threshold": 4, "failure_threshold": -3,
            "success_phrases": ["it's okay", "we'll find your mother", "you are safe", "i'm listening", "take your time", "can you tell me about", "i'm here to help", "quiet voice", "my name is"],
            "failure_phrases": ["calm down", "stop being weird", "just obey", "look at me", "snap out of it", "what's wrong with you", "stop that", "taser"],
            "person_of_interest": "Jacob",
        },
        4: {
             "name": "Angry Veteran - Ricardo",
             "setup": textwrap.dedent("""\
                 Units respond to Fort Worth East Regional Library, a voting location, reference Disturbance.
                 Caller reports a male subject outside yelling conspiracy theories at voters, refusing to leave.
                 Subject reportedly has a large hunting knife sheathed on his belt. Location is active with voters.
             """),
            "character_profile": "Ricardo, veteran, angry and highly paranoid about government/conspiracies (rigged elections, JFK, aliens, MK Ultra, reptilians). Yelling loudly near voters, causing disturbance. Has a large hunting knife sheathed on belt. Not suicidal, but highly agitated, distrustful of authority, potentially volatile.",
            "character_triggers": "Loud/aggressive commands, dismissing his beliefs ('crazy', 'stupid'), immediate threat presentation (Taser/weapon drawn), focusing on the knife prematurely, telling him to 'calm down' or 'leave now'.",
            "character_reactions": "Can panic or become defensively aggressive if confronted harshly, potentially drawing knife or running/charging. Willing to talk (rant extensively) if approached calmly. May build rapport if officer listens respectfully (without validating conspiracies) or acknowledges military service.",
            "scenario_goal": "Ensure public safety (manage crowd/distance, prevent access to knife), de-escalate Ricardo through calm communication and active listening, resolve situation without force (e.g., convince him to move away, offer ride home, voluntary compliance).",
            "rag_context": ["Ricardo believes the system is rigged and politicians are reptilians.", "He served in the military (branch/details unknown unless he shares).", "The large hunting knife is currently sheathed on his belt.", "Voters are present and concerned, creating a public safety issue.", "Less-lethal options should be ready but not displayed aggressively.", "Acknowledging his service or allowing him to vent (active listening) can help build rapport.", "Directly challenging his conspiracy theories is likely counter-productive.", "Try to move the conversation/person to a less public area if possible."],
             "voice_name": "en-US-Studio-Q", # Kept for reference
             "progress_threshold": 5, "failure_threshold": -4,
             "success_phrases": ["tell me more", "i hear you", "thank you for your service", "understand your concerns", "let's talk over here", "i want to understand", "what happened", "talk to me"],
             "failure_phrases": ["calm down", "you're crazy", "that's stupid", "that's not true", "just stop", "drop the knife", "put your hands up", "you need to leave now"],
             "person_of_interest": "Ricardo",
        },
        5: {
             "name": "Suicidal Domestic Disturbance - William",
             "setup": textwrap.dedent("""\
                 Units respond to a Domestic Disturbance, possible 10-56 (Suicide Attempt).
                 Caller, Becky, reports her husband William is in the garage with a rope around his neck,
                 standing on a chair, threatening to kill himself. Stated cause is related to caller finding something on his phone. Subject is reported as otherwise unarmed.
             """),
            "character_profile": "William, acutely suicidal, rope around neck, standing on chair in garage. Extremely guilty, ashamed, and despairing over infidelity (online activity found by wife Becky). Believes he 'deserves to die'. Not initially aggressive towards others but focused on self-harm.",
            "character_triggers": "Seeing drawn weapons (will invite officer to shoot him or may immediately act), loud commands, wife (Becky) yelling accusations or being present and hostile, dismissive comments.",
            "character_reactions": "Initial refusal ('leave us alone', 'come back when I'm dead'). High risk of immediate suicidal action (kicking chair) or inviting 'suicide by cop' if triggers are hit. Can be calmed and persuaded to step down if spoken to calmly, non-judgmentally, wife is removed/managed, and empathy/hope is offered.",
            "scenario_goal": "Prevent immediate suicide. Establish calm communication, build rapport, remove wife if she escalates, convince William to step down safely, arrange mental health evaluation (mental APP).",
            "rag_context": ["William is unarmed besides the rope around his neck.", "He is standing on a chair/stool under the rope tied to rafters.", "He feels immense guilt over something Becky found on his phone (infidelity via online videos).", "Becky is present and potentially hostile/accusatory, which is a major escalation risk.", "Lethal/Less-lethal weapons are inappropriate and highly likely to trigger suicidal action.", "Removing Becky from the immediate area is crucial if she's escalating.", "He can be reasoned with if approached calmly, non-judgmentally, and shown alternative solutions/hope.", "Focus on listening, empathy, and the value of his life."],
             "voice_name": "en-US-Wavenet-D", # Kept for reference
             "progress_threshold": 5, "failure_threshold": -5,
             "success_phrases": ["we can talk this out", "we can get you help", "you don't deserve to die", "you're safe", "step down", "i'm listening", "let's figure this out", "i want to help", "talk to me", "i understand this is hard"],
             "failure_phrases": ["just do it", "calm down", "shut up", "get down", "you should pay", "think about your wife", "get down now", "don't do it"],
             "person_of_interest": "William",
         },
    }
    # =======================================================

    while True:
        session_mgr = SessionManager(config, stt_client, tts_client, ollama_client)
        try:
            chosen_scenario_data = session_mgr.select_scenario(SCENARIOS)
            if not chosen_scenario_data:
                 logger.error("Scenario selection failed. Exiting.")
                 break
            session_mgr.play_scenario_setup()
            session_mgr.main_loop()
        except KeyboardInterrupt:
             logger.warning("Keyboard interrupt detected during session. Exiting.")
             break
        except Exception as main_loop_error:
            logger.error(f"An unexpected error occurred during the session: {main_loop_error}", exc_info=True)
            print(f"An error occurred: {main_loop_error}. Check logs for details.")

        try:
            while True:
                again = input("\nRun another scenario? (y/n): ").strip().lower()
                if again in ['y', 'n']: break
                else: print("Please enter 'y' or 'n'.")
            if again == 'n':
                print("Exiting trainer.")
                break
        except EOFError:
             print("\nInput stream closed. Exiting trainer.")
             break

    print("\n==== De-escalation Trainer Finished ====")
    print("==== Thank you for using the Police De-escalation Trainer ====")
    print("==== Please provide feedback to improve the training experience ====")
    print("==== Goodbye! ====")
