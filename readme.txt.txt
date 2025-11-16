A simple AI-powered chatbot that applies real-time audio effects using natural language commands. Users can load audio files, apply effects like reverb, pitch shift, EQ, slicing, reversing, and visualize waveforms or spectrograms — all through text-based interaction.

Features

Natural-language command processing

Audio effects: reverb, EQ, pitch shift, reverse, slice, crossfade

Waveform, spectrogram, and chromagram visualization

Real-time audio playback

Save processed audio as .wav

Tech Stack

Python

Librosa

PyDub

Matplotlib

SoundDevice

OpenAI API

How to Run

Install dependencies:

pip install librosa pydub matplotlib sounddevice openai numpy


Add your OpenAI API key in the script.

Run the program:

python GradProject.py

Usage

Type “chat” to talk with the chatbot.

Type “audio” to load an audio file and apply effects.

Follow on-screen prompts for available commands.

Dataset

This project supports the GTZAN Genre Collection for sample audio files.

Future Improvements

Voice command support

More audio effects (echo, distortion, chorus, etc.)

Beat detection and tempo analysis

GUI interface

Usage Guide
Chat Mode

Type chat to start chatting with the AI.

Ask anything or interact naturally.

Type exit to leave chat mode.

Audio Mode

Type audio to load an audio file.

Choose a genre and a sample from the list.

Use available audio commands:

slice audio

reverse audio

crossfade audio

reverb

pitch shift

eq

plot waveform

plot spectrogram

plot chromagram

play audio

save file

Dataset

The project uses the GTZAN Genre Collection for testing audio files.
You can replace it with your own dataset if needed.

Future Improvements

Voice command support

Advanced audio effects (echo, chorus, distortion)

Beat detection and tempo analysis

Real-time spectrogram streaming

Simple GUI version