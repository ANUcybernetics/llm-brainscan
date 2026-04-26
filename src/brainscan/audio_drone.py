"""Optional sub-bass drone whose pitch tracks training loss."""

from __future__ import annotations

import threading

import numpy as np


class DroneOscillator:
    def __init__(
        self,
        sample_rate: int = 44100,
        min_hz: float = 40.0,
        max_hz: float = 60.0,
        gain_db: float = -18.0,
        device: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.gain = 10 ** (gain_db / 20.0)
        self.device = device
        self._phase = 0.0
        self._hz = (min_hz + max_hz) / 2.0
        self._lock = threading.Lock()
        self._stream = None

    def update_loss(self, loss: float, max_loss: float = 4.0) -> None:
        t = float(np.clip(loss / max_loss, 0.0, 1.0))
        new_hz = self.min_hz + (self.max_hz - self.min_hz) * (1.0 - t)
        with self._lock:
            self._hz = new_hz

    def start(self) -> None:
        import sounddevice as sd

        def callback(out, frames, _time, _status) -> None:
            with self._lock:
                hz = self._hz
            phase_inc = 2 * np.pi * hz / self.sample_rate
            samples = np.empty(frames, dtype=np.float32)
            for i in range(frames):
                samples[i] = np.sin(self._phase) * self.gain
                self._phase = (self._phase + phase_inc) % (2 * np.pi)
            out[:, 0] = samples

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
