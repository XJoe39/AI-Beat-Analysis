import openai

import openai
import os
from pydub import AudioSegment
from pydub.playback import play
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from spleeter.separator import Separator
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure FFmpeg paths
try:
    AudioSegment.converter = r"C:\Users\cdc\Desktop\Uni\Level 6\Grad Project\FFmpeg\bin\ffmpeg.exe"
    AudioSegment.ffprobe = r"C:\Users\cdc\Desktop\Uni\Level 6\Grad Project\FFmpeg\bin\ffprobe.exe"
    # Verify FFmpeg is accessible
    if not os.path.exists(AudioSegment.converter):
        raise FileNotFoundError("FFmpeg not found at specified path")
except Exception as e:
    print(f"FFmpeg configuration error: {e}")
    exit()

# Set paths
DATA_DIR = r"C:\Users\cdc\Desktop\uni\Level 6\Grad Project\GTZAN\data\genres_original"
SAVE_DIR = r"C:\Users\cdc\Desktop\uni\Level 6\Grad Project\Saved Audio"
os.makedirs(SAVE_DIR, exist_ok=True)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# ChatGPT interaction
def chat_with_gpt(prompt, chat_history=[]):
    try:
        chat_history.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history
        )
        reply = response.choices[0].message.content.strip()
        chat_history.append({"role": "assistant", "content": reply})
        return reply, chat_history
    except Exception as e:
        print(f"ChatGPT error: {e}")
        return "I couldn't process that request.", chat_history

# Audio selection function
def choose_audio():
    try:
        print("Available genres:")
        for i, genre in enumerate(genres):
            print(f"{i + 1}. {genre}")

        genre_choice = int(input("Choose a genre by number: ")) - 1
        if genre_choice < 0 or genre_choice >= len(genres):
            raise ValueError("Invalid genre selection")

        genre = genres[genre_choice]
        genre_dir = os.path.join(DATA_DIR, genre)

        audio_files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        if not audio_files:
            raise FileNotFoundError("No audio files found in genre directory")

        print(f"\nAvailable audio files in {genre}:")
        for i, filename in enumerate(audio_files):
            print(f"{i + 1}. {filename}")

        file_choice = int(input("Choose an audio file by number: ")) - 1
        if file_choice < 0 or file_choice >= len(audio_files):
            raise ValueError("Invalid file selection")

        return os.path.join(genre_dir, audio_files[file_choice]), genre

    except Exception as e:
        print(f"Error selecting audio: {e}")
        return None, None

# Visualization functions
def plot_waveform(audio, sr):
    try:
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(audio, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Waveform error: {e}")

def plot_spectrogram(audio, sr):
    try:
        plt.figure(figsize=(14, 5))
        S = librosa.stft(librosa.util.normalize(audio))
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Spectrogram error: {e}")

def plot_chromagram(audio, sr):
    try:
        plt.figure(figsize=(18, 5))
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
        plt.colorbar()
        plt.title("Chromagram")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Chromagram error: {e}")

# Audio manipulation functions
def slice_audio(audio, start_sec, end_sec):
    try:
        if start_sec < 0 or end_sec > len(audio) / 1000 or start_sec >= end_sec:
            raise ValueError("Invalid time range")
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        return audio[start_ms:end_ms]
    except Exception as e:
        print(f"Slice error: {e}")
        return audio

def reverse_audio(audio):
    try:
        return audio.reverse()
    except Exception as e:
        print(f"Reverse error: {e}")
        return audio

def crossfade_audio(audio, duration_sec):
    try:
        if duration_sec <= 0:
            raise ValueError("Duration must be positive")
        duration_ms = int(duration_sec * 1000)
        return audio.fade_in(duration_ms).fade_out(duration_ms)
    except Exception as e:
        print(f"Crossfade error: {e}")
        return audio

def apply_reverb(audio):
    try:
        wet = audio.low_pass_filter(1200).high_pass_filter(300)
        return audio.overlay(wet, gain_during_overlay=-6)
    except Exception as e:
        print(f"Reverb error: {e}")
        return audio

def apply_pitch_shift(audio, sr, steps):
    try:
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        shifted = librosa.effects.pitch_shift(samples, sr=sr, n_steps=steps)
        shifted_samples = (shifted * 32767).astype(np.int16)
        return AudioSegment(
            shifted_samples.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio

def apply_eq(audio, gain):
    try:
        if not -20 <= gain <= 20:
            raise ValueError("Gain must be between -20dB and 20dB")
        return audio.apply_gain(gain)
    except Exception as e:
        print(f"EQ error: {e}")
        return audio

def apply_delay(audio, delay_sec, decay):
    try:
        if delay_sec <= 0 or decay <= 0:
            raise ValueError("Delay and decay must be positive")
        delay_ms = int(delay_sec * 1000)
        delayed = audio.delay(delay_ms).apply_gain(-20 * decay)
        return audio.overlay(delayed, position=delay_ms)
    except Exception as e:
        print(f"Delay error: {e}")
        return audio

def apply_loop(audio, loops):
    try:
        if loops < 1:
            raise ValueError("Loops must be ≥1")
        if loops > 10:  # Limit the number of loops to avoid excessive memory usage
            raise ValueError("Loops must be ≤10")
        return audio * loops
    except Exception as e:
        print(f"Loop error: {e}")
        return audio

def apply_modulation(audio, rate):
    try:
        if rate <= 0:
            raise ValueError("Rate must be positive")
        samples = np.array(audio.get_array_of_samples())
        t = np.arange(len(samples)) / audio.frame_rate
        modulator = np.sin(2 * np.pi * rate * t)
        modulated = (samples * modulator).astype(np.int16)
        return AudioSegment(
            modulated.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=1
        )
    except Exception as e:
        print(f"Modulation error: {e}")
        return audio

def play_audio(audio):
    try:
        samples = np.array(audio.get_array_of_samples())
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
        print("Chatbot: Finished playing the audio.")
    except Exception as e:
        print(f"Playback error: {e}")

def save_audio(file_name, audio):
    try:
        if not file_name.endswith('.wav'):
            file_name += '.wav'
        save_path = os.path.join(SAVE_DIR, file_name)
        audio.export(save_path, format="wav")
        print(f"Chatbot: Modified audio saved as '{save_path}'")
    except Exception as e:
        print(f"Save error: {e}")

def detect_tempo(audio, sr):
    try:
        tempo, beats = librosa.beat.beat_track(audio, sr=sr)
        return tempo
    except Exception as e:
        print(f"Tempo detection error: {e}")
        return None

def detect_key(audio, sr):
    try:
        key = librosa.key(audio, sr=sr)
        return key
    except Exception as e:
        print(f"Key detection error: {e}")
        return None

# Chatbot loop
def chatbot():
    print("Chatbot: Hi there! Type 'hello' to start the conversation.")
    while True:
        user_input = input("You: ").lower().strip()

        if user_input in ["hello", "hi", "hey"]:
            print(
                "Chatbot: Nice to meet you! I can chat or help with audio editing. Type 'chat' to talk or 'audio' to edit audio.")
            break
        elif user_input in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            return
        else:
            print("Chatbot: Please start by saying 'hello' or 'exit'.")

    chat_history = []
    audio_history = []

    while True:
        user_input = input("You: ").lower().strip()

        if user_input in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        elif "chat" in user_input:
            print("Chatbot: Let's chat! Ask me anything. Type 'back' to return to main menu.")
            while True:
                chat_input = input("You: ").strip()
                if chat_input.lower() in ["exit", "quit", "bye", "back"]:
                    print("Chatbot: Ending chat mode.")
                    break
                response, chat_history = chat_with_gpt(chat_input, chat_history)
                print(f"Chatbot: {response}")

        elif "audio" in user_input:
            print("Chatbot: Let's choose a genre and audio file first.")
            file_path, genre = choose_audio()
            if not file_path:
                continue

            try:
                audio = AudioSegment.from_file(file_path)
                x, sr = librosa.load(file_path, sr=None, duration=30)
                audio_history = [audio]  # Reset history with the original audio
                print(f"\nChatbot: Loaded {genre} audio from {os.path.basename(file_path)}")

                while True:
                    print("\nChatbot: Available commands:")
                    print("- 'slice [start] [end]' (seconds)")
                    print("- 'reverse'")
                    print("- 'crossfade [duration]'")
                    print("- 'reverb'")
                    print("- 'pitch [steps]'")
                    print("- 'eq [gain]'")
                    print("- 'delay [time] [decay]'")
                    print("- 'loop [count]'")
                    print("- 'modulation [rate]'")
                    print("- 'tempo'")
                    print("- 'key'")
                    print("- 'repeat last'")
                    print("- 'undo'")
                    print("- 'reset'")
                    print("- 'plot waveform/spectrogram/chromagram'")
                    print("- 'play' - 'save [name]' - 'exit'")

                    command = input("Chatbot: Command: ").lower().strip()
                    parts = command.split()

                    if not parts:
                        continue

                    try:
                        if parts[0] == "slice" and len(parts) >= 3:
                            start, end = float(parts[1]), float(parts[2])
                            audio = slice_audio(audio, start, end)
                            print(f"Sliced audio from {start}s to {end}s")
                            audio_history.append(audio)

                        elif parts[0] == "reverse":
                            audio = reverse_audio(audio)
                            print("Audio reversed")
                            audio_history.append(audio)

                        elif parts[0] == "crossfade" and len(parts) >= 2:
                            duration = float(parts[1])
                            audio = crossfade_audio(audio, duration)
                            print(f"Applied {duration}s crossfade")
                            audio_history.append(audio)

                        elif parts[0] == "reverb":
                            audio = apply_reverb(audio)
                            print("Applied reverb")
                            audio_history.append(audio)

                        elif parts[0] == "pitch" and len(parts) >= 2:
                            steps = float(parts[1])
                            audio = apply_pitch_shift(audio, sr, steps)
                            print(f"Pitch shifted by {steps} steps")
                            audio_history.append(audio)

                        elif parts[0] == "eq" and len(parts) >= 2:
                            gain = float(parts[1])
                            audio = apply_eq(audio, gain)
                            print(f"Applied {gain}dB EQ")
                            audio_history.append(audio)

                        elif parts[0] == "delay" and len(parts) >= 3:
                            delay = float(parts[1])
                            decay = float(parts[2])
                            audio = apply_delay(audio, delay, decay)
                            print(f"Applied {delay}s delay with {decay} decay")
                            audio_history.append(audio)

                        elif parts[0] == "loop" and len(parts) >= 2:
                            loops = int(parts[1])
                            audio = apply_loop(audio, loops)
                            print(f"Looped {loops} times")
                            audio_history.append(audio)

                        elif parts[0] == "modulation" and len(parts) >= 2:
                            rate = float(parts[1])
                            audio = apply_modulation(audio, rate)
                            print(f"Applied modulation at {rate}Hz")
                            audio_history.append(audio)

                        elif parts[0] == "tempo":
                            tempo = detect_tempo(x, sr)
                            if tempo:
                                print(f"Detected tempo: {tempo} BPM")

                        elif parts[0] == "key":
                            key = detect_key(x, sr)
                            if key:
                                print(f"Detected key: {key}")

                        elif parts[0] == "repeat last":
                            if len(audio_history) > 1:
                                audio = audio_history[-2]
                                print("Repeated last transformation")
                            else:
                                print("No previous transformations to repeat")

                        elif parts[0] == "undo":
                            if len(audio_history) > 1:
                                audio = audio_history[-2]
                                audio_history.pop()
                                print("Undid last transformation")
                            else:
                                print("No transformations to undo")

                        elif parts[0] == "reset":
                            audio = audio_history[0]
                            audio_history = [audio]
                            print("Reset to original audio")

                        elif parts[0] == "plot":
                            if len(parts) >= 2:
                                if parts[1] == "waveform":
                                    plot_waveform(x, sr)
                                elif parts[1] == "spectrogram":
                                    plot_spectrogram(x, sr)
                                elif parts[1] == "chromagram":
                                    plot_chromagram(x, sr)

                        elif parts[0] == "play":
                            play_audio(audio)

                        elif parts[0] == "save" and len(parts) >= 2:
                            save_name = ' '.join(parts[1:])
                            save_audio(save_name, audio)

                        elif parts[0] == "exit":
                            print("Chatbot: Exiting audio editor.")
                            break

                        else:
                            print("Chatbot: Command not recognized")

                    except Exception as e:
                        print(f"Error executing command: {e}")

            except Exception as e:
                print(f"Audio processing error: {e}")

        else:
            print("Chatbot: I didn't understand. Try 'chat', 'audio', or 'exit'.")

# Main execution
if __name__ == "__main__":
    chatbot()