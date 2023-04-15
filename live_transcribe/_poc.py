import threading
from queue import Queue
from time import time

import numpy as np
import pasimple
import torch
import whisper
from colorama import Back, Cursor, Fore, Style
from whisper.tokenizer import get_tokenizer

from live_transcribe.text import merge_sequences, split_into_lines

MODEL = "small"
PA_FORMAT = pasimple.PA_SAMPLE_S32LE
NP_FORMAT = np.int32
SAMPLE_WIDTH = pasimple.format2width(PA_FORMAT)
CHANNELS = 1
SAMPLE_RATE = whisper.audio.SAMPLE_RATE
LENGTH = 0.1
LANGUAGE = "en"
MAX_LEN = SAMPLE_RATE * 5
FALSE = False

device_name = "bluez_output.00_1D_43_A0_C1_DF.a2dp-sink.monitor"
cookie = 0

if torch.cuda._is_compiled():
    DEVICE = "cuda"
else:
    print("CUDA is not available. Using CPU instead. You will experience a significant latency.")
    DEVICE = "cpu"

model_name = MODEL + ".en" if LANGUAGE == "en" else MODEL
model = whisper.load_model(model_name, device=DEVICE)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

chunks_queue = Queue()
cookie += 1


def record_loop():
    with pasimple.PaSimple(pasimple.PA_STREAM_RECORD, PA_FORMAT, CHANNELS, SAMPLE_RATE, device_name=device_name) as pa:
        my_cookie = cookie
        while cookie == my_cookie:
            # print("recording...")
            audio_data = pa.read(int(CHANNELS * SAMPLE_RATE * SAMPLE_WIDTH * LENGTH))
            # print("done...")
            chunks_queue.put((time(), audio_data))


rec_task = threading.Thread(target=record_loop)
rec_task.start()

timeout = LENGTH * 2

tokenizer = get_tokenizer(
    model.is_multilingual, language=LANGUAGE, task="transcribe"
)

best_lang = LANGUAGE

options = whisper.DecodingOptions(
    language=best_lang,
    suppress_tokens=[],
    suppress_blank=False,
    without_timestamps=False,
    fp16=False,
    length_penalty=0.1,
    # prompt=prev_text
)

MAX_NUM_LINES = 5
MAX_LINE_LEN = 80


# MAX_LEN = whisper.audio.N_SAMPLES
def trascribe_loop():
    audio_data = bytearray()
    my_cookie = cookie
    curr_text = ""
    prev_text = ""
    prev_end_ts = None
    while cookie == my_cookie:
        if FALSE:
            print(f"qsize: {chunks_queue.qsize()}")
        audio_data = bytearray(audio_data)
        end_ts, chunk = chunks_queue.get(timeout=timeout)
        audio_data += chunk
        while not chunks_queue.empty():
            end_ts, chunk = chunks_queue.get(timeout=timeout)
            audio_data += chunk

        audio_data = bytearray(audio_data[-MAX_LEN * SAMPLE_WIDTH:])
        audio = np.frombuffer(audio_data, dtype=NP_FORMAT)
        audio_normalized = (audio / np.max(audio)).astype(np.float32)

        audio_padded = whisper.pad_or_trim(audio_normalized)
        if FALSE:
            print(f"{len(audio_normalized)} -> {len(audio_padded)}")
        mel_cpu = whisper.log_mel_spectrogram(audio_padded)
        mel = mel_cpu.to(model.device)

        # _, probs = model.detect_language(mel)
        # best_lang = max(probs, key=probs.get)
        # print(f"Detected language: {best_lang}")

        # print(":::", end="")
        result = whisper.decode(model, mel, options)

        curr_text2 = " ".join(merge_sequences(curr_text.split(" "), result.text.split(" ")))
        if FALSE:
            print(curr_text)
            print(curr_text2)
            print(result.text)
        curr_text = curr_text2

        curr_text_lines = split_into_lines(curr_text, MAX_LINE_LEN)

        if len(curr_text_lines) > MAX_NUM_LINES:
            lines_to_remove = len(curr_text_lines) - MAX_NUM_LINES
            len_to_remove = sum(map(len, curr_text_lines[:lines_to_remove])) + lines_to_remove
            curr_text = curr_text[len_to_remove:]
            curr_text_lines = split_into_lines(curr_text, MAX_LINE_LEN)
            # sys.stderr.write(Cursor.DOWN(1))

        sys.stderr.write(Cursor.UP(len(curr_text_lines)) + "\n".join(curr_text_lines) + "\n")

        if FALSE:
            tokenizer.decode_with_timestamps(result.tokens)
            decoded_tokens, _ = tokenizer.split_to_word_tokens(result.tokens)
            print(prev_decoded_tokens)
            print(decoded_tokens)
            print(list(zip(prev_tokens, prev_decoded_tokens)))
            print(list(zip(result.tokens, decoded_tokens)))
            prev_end_ts
            end_ts
            print(prev_tokens)
            print(result.tokens)
            tokenizer.decode_with_timestamps(prev_tokens)
            tokenizer.decode_with_timestamps(result.tokens)

            prev_tokens = result.tokens
            prev_decoded_tokens = decoded_tokens
            prev_end_ts = end_ts
        # prev_text = result.text
    chunks_queue.task_done()


trans_task = threading.Thread(target=trascribe_loop)
trans_task.start()

import sys

print("11111111")
print("11111111")
sys.stdout.write(u"2222\u001b[5Deee")
print("333")

for i in range(0, 100):
    sys.stdout.write(u"\u001b[1000D" + str(i + 1) + "%")
    sys.stdout.flush()

import time, sys

print("\033[FMy text overwriting the previous line.")
print("\033[FMy text overwriting the previous line.")
print("\033[FMy text overwriting the previous line.")

import tqdm

for i in tqdm.tqdm(range(100)):
    time.sleep(0.1)

sys.stdout.write('aaa\x1b[Abbb')
sys.stdout.flush()


def move(y, x):
    print("\033[%d;%dH" % (y, x))


print("aaa")
move(-1, -1)
print("bbb")

from colorama import just_fix_windows_console, init

just_fix_windows_console()
init()

for i in range(5):
    sys.stderr.write(Fore.RED + 'some red text')
    sys.stderr.write(Fore.RED + 'some red text')
    sys.stderr.write(Fore.RED + 'some red text')
    sys.stderr.write(Cursor.UP(2) + Back.GREEN + 'and with a green background')
    sys.stderr.write(Style.DIM + 'and in dim text')
    sys.stderr.write(Style.RESET_ALL)
    sys.stderr.write('back to normal now')


def print_loop():
    for i in range(4):
        sys.stderr.write("aaaa\n")
        sys.stderr.write(Cursor.UP(2) + 'bbb\n')
        time.sleep(1)


rec_task = threading.Thread(target=record_loop)
rec_task.start()
