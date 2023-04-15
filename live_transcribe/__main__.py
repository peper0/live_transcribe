import logging
import sys

import whisper

from live_transcribe.audio_pasimple import PaSimpleCapture
from live_transcribe.transcribe import rolling_transcribe

device_name = "bluez_output.00_1D_43_A0_C1_DF.a2dp-sink.monitor"
stream = sys.stdout
logging.basicConfig(level=logging.INFO)

with PaSimpleCapture(whisper.audio.SAMPLE_RATE, device_name) as pa:
    # for txt in  split_into_lines_live(rolling_transcribe(pa), 80, 100):
    for txt in rolling_transcribe(pa):
        stream.write(txt)
        stream.flush()
