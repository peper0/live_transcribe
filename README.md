Live Transcribe

Live Transcribe is a Python package that provides live, real-time transcription based on OpenAI's Whisper. Works
offline.

Currently, Live Transcribe supports only PulseAudio as an audio backend.

# Installation

1. **optional but highly recommended for low latency** Refer to the [Pytorch documentation guide](https://pytorch.org/)
   to install Pytorch **with CUDA support**.
1. `pip install live-transcribe`

# Usage

Just run:

    python -m live_transcribe

or

    live_transcribe

If you want to transcribe from another audio device, than the default, use the `--device` option, e.g.:

    live_transcribe --list-devices
    live_transcribe --device "alsa_input.usb-046d_HD_Pro_Webcam_C920_8C0B5B0F-02.analog-stereo

On the first usage, the OpenAI's Whisper model will be downloaded and cached.

See `live_transcribe --help` for options.

# Dependencies

Live Transcribe has the following dependencies:

    Python 3.8 or higher
    OpenAI-Whisper
    PulseAudio
