import threading
from typing import Mapping, Optional

import numpy as np
import pasimple


class PaSimpleCapture:
    NP_TO_PA_FORMAT: Mapping[np.dtype, int] = {
        np.int32: pasimple.PA_SAMPLE_S32LE,
        np.float32: pasimple.PA_SAMPLE_FLOAT32LE,
        np.int16: pasimple.PA_SAMPLE_S16LE,
    }

    def __init__(self, rate: int, device_name: str = None, format: np.dtype = np.int32):
        if format not in self.NP_TO_PA_FORMAT:
            raise ValueError(f"Unsupported format: {format}; supported formats: {self.NP_TO_PA_FORMAT.keys()}")
        pa_format = self.NP_TO_PA_FORMAT[format]
        self._pa = pasimple.PaSimple(pasimple.PA_STREAM_RECORD, pa_format, 1, rate, device_name=device_name,
                                     app_name="live_transcribe")
        self._pa2: Optional[pasimple] = None
        self._chunk_size_bytes: int = rate // 256 * pasimple.format2width(pa_format)  # 1/256 s of latency
        self._stop = False
        self._thread = threading.Thread(target=self._run)
        self._mutex = threading.Lock()
        self._data = bytearray()
        self._np_type = format
        self._exception: Optional[BaseException] = None

    def _run(self):
        try:
            while not self._stop:
                # print("recording...")
                new_data = self._pa2.read(self._chunk_size_bytes)
                # print(f"got {len(new_data)} bytes")
                with self._mutex:
                    self._data += new_data
        except Exception as e:
            self._exception = e

    def __enter__(self) -> 'PaSimpleCapture':
        self._pa2 = self._pa.__enter__()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._stop = True
        self._thread.join()
        return self._pa.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._exception is not None:
            raise self._exception
        with self._mutex:
            data = self._data
            self._data = bytearray()

        return np.frombuffer(data, dtype=self._np_type)
