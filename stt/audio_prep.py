"""
audio_prep.py
-------------
STT 에 보내기 전에 녹음 소리 크기를 맞춘다(CLOVA 엔진 공용).

어르신 목소리가 작거나 인형에서 멀리 계시면 녹음이 작게 들어와, 인식기가
작은 소리를 잡음으로 보고 말을 빠뜨리거나 틀리게 받아 적는다. 말소리 구간의
크기를 일정하게(TARGET_DBFS) 올려 보낸다.

- 말소리 구간만 보고 크기를 잰다(앞뒤 무음까지 평균 내면 작게 잡혀 너무 키운다).
- 너무 키우면 잡음도 함께 커지므로 MAX_GAIN_DB 까지만 올린다.
- 키운 뒤 소리가 깨지지(클리핑) 않게 피크가 PEAK_LIMIT 를 넘지 않을 만큼만 올린다.
- 이미 충분히 크면 건드리지 않는다(줄이지 않는다).
"""

import io

import numpy as np
import soundfile as sf

TARGET_DBFS = -20.0      # 말소리 구간 RMS 목표
MAX_GAIN_DB = 20.0       # 최대 10배까지만 키운다
PEAK_LIMIT = 0.89        # ≈ -1 dBFS
FRAME_MS = 30
SPEECH_RATIO = 0.25      # 가장 큰 프레임의 이 비율 넘는 프레임을 말소리로 본다
MIN_GAIN_DB = 1.0        # 이보다 적게 바뀌면 그대로 둔다


def _db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-9))


def normalize(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """말소리 크기를 맞춘 오디오와 적용한 이득(dB)을 돌려준다. audio 는 float(-1~1) 모노."""
    frame = max(1, int(sr * FRAME_MS / 1000))
    n = len(audio) // frame
    if n < 2:
        return audio, 0.0
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    loud = rms[rms >= rms.max() * SPEECH_RATIO]
    speech_rms = float(np.sqrt(np.mean(loud ** 2))) if len(loud) else float(rms.max())

    gain_db = min(TARGET_DBFS - _db(speech_rms), MAX_GAIN_DB)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        gain_db = min(gain_db, _db(PEAK_LIMIT / peak))   # 깨지지 않을 만큼만
    if gain_db < MIN_GAIN_DB:
        return audio, 0.0
    return (audio * (10 ** (gain_db / 20))).astype(np.float32), gain_db


def prepared_wav_bytes(audio_path: str, do_normalize: bool = True) -> tuple[bytes, float, float]:
    """녹음 파일을 읽어 (모노 16bit WAV 바이트, 길이 초, 적용한 이득 dB) 를 돌려준다."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    gain_db = 0.0
    if do_normalize:
        audio, gain_db = normalize(audio, sr)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), len(audio) / sr if sr else 0.0, gain_db
