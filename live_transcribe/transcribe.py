import logging
from copy import copy
from numbers import Number
from typing import Generator, Iterable, Sequence

import numpy as np
import torch
import whisper

from live_transcribe.text import compute_delta, merge_texts


def rolling_transcribe(
        data_generator: Iterable[Sequence[Number]],
        model: str = "small",
        language: str = "en",
        audio_window_size: int = whisper.audio.SAMPLE_RATE * 5,
) -> Generator[str, None, None]:
    """
    Transcribe audio in a rolling fashion, i.e. the model is fed with the last `window_size` samples.
    :param data_generator: a generator that yields audio samples
    :param model: "tiny", "base", "small", "medium", "large"
    :param language: "en", "pl", ...
    :param audio_window_size: a number of samples to keep in the buffer (`whisper.audio.SAMPLE_RATE` samples per second)
    :return: a generator that yields transcriptions
    """
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        logging.warning("CUDA is not available. Using CPU instead. You will experience a significant latency.")
        DEVICE = "cpu"

    model_name = model + ".en" if language == "en" else model
    logging.info(f"Loading model {model_name}...")
    model = whisper.load_model(model_name, device=DEVICE)
    logging.debug(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    logging.info("Model loaded.")

    options = whisper.DecodingOptions(
        language=language,
        suppress_tokens=[],
        suppress_blank=False,
        without_timestamps=False,
        fp16=False,
        length_penalty=0.1,
        # prompt=prev_text
    )

    prev_text = ""
    audio_samples = None
    for new_audio_data in data_generator:
        len_from_previous = audio_window_size - len(new_audio_data)
        if audio_samples is None or len_from_previous < 0:
            audio_samples = copy(new_audio_data)
        else:
            audio_samples = np.concatenate((audio_samples[-len_from_previous:], new_audio_data))

        audio_normalized = (audio_samples / np.max(audio_samples)).astype(np.float32)
        audio_padded = whisper.pad_or_trim(audio_normalized)
        mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)
        result = whisper.decode(model, mel, options)

        curr_text = merge_texts(prev_text, result.text)
        yield compute_delta(prev_text, curr_text)
        prev_text = curr_text[-len(result.text):]
