import librosa
import numpy as np


def preprocess_audio(file_path, duration=2.5, sample_rate=22050, top_db=20):
    samples_per_clip = int(sample_rate * duration)

    y, _ = librosa.load(file_path, sr=sample_rate)

    y, _ = librosa.effects.trim(y, top_db=top_db)

    if len(y) > samples_per_clip:
        y = y[:samples_per_clip]
    else:
        padding = samples_per_clip - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    return y

