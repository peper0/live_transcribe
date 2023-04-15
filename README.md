Live Transcribe

Live Transcribe is a Python package that provides live, real-time transcription of audio based on OpenAI's Whisper API.

Currently, Live Transcribe supports only PulseAudio as an audio backend.

# Installation

1. **optional but highly recommended for low latency** Refer to the [Pytorch documentation guide](https://pytorch.org/)
   to install Pytorch **with CUDA support**.
1. `pip install live-transcribe`

# Usage

Just run:

    python -m live_transcribe

On the first usage, the OpenAI's Whisper model will be downloaded and cached.

See `live_transcribe --help` for options.

# Dependencies

Live Transcribe has the following dependencies:

    Python 3.8 or higher
    OpenAI-Whisper
    PulseAudio
