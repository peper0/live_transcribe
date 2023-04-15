import argparse
import logging
import sys
from contextlib import suppress

import pulsectl
import whisper

from live_transcribe.audio_pasimple import PaSimpleCapture
from live_transcribe.transcribe import rolling_transcribe

logging.basicConfig(level=logging.INFO)


def obtain_default_device_name():
    pulse = pulsectl.Pulse('live_transcribe')
    sinks_by_name = {i.name: i for i in pulse.sink_list()}
    default_sink = sinks_by_name[pulse.server_info().default_sink_name]
    return default_sink.monitor_source_name


def print_devices():
    pulse = pulsectl.Pulse('live_transcribe')
    for i in pulse.source_list():
        with suppress(AttributeError):
            print(f"  {i.name}: {i.description}")


# program parameters
argparser = argparse.ArgumentParser()
argparser.add_argument("--device", help="audio device name; by default the monitor of the default sink is used")
argparser.add_argument("--model", help="model name", default="small")
argparser.add_argument("--language", help="language", default="en")
argparser.add_argument("--window",
                       type=float,
                       help="a length (in seconds) of audio sample passed each time to the model; greater values improve quality but increase latency",
                       default=4)
argparser.add_argument("--list-devices", action="store_true", help="list available audio devices and exit")

args = argparser.parse_args()

if args.list_devices:
    print_devices()
    sys.exit(0)

audio_device_name = args.device or obtain_default_device_name()
logging.info(f"Using audio device {audio_device_name}")
stream = sys.stdout

rate = whisper.audio.SAMPLE_RATE
with PaSimpleCapture(rate, device_name=audio_device_name) as pa:
    # for txt in  split_into_lines_live(rolling_transcribe(pa), 80, 100):
    for txt in rolling_transcribe(pa,
                                  model=args.model,
                                  language=args.language,
                                  audio_window_size=int(args.window * rate)):
        stream.write(txt)
        stream.flush()
