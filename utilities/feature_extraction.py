import numpy as np
import librosa
import librosa.feature


def preprocess_audio(file_path, duration=2.5, sample_rate=22050, top_db=20):
    y, _ = librosa.load(file_path, sr=sample_rate)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    target_length = int(sample_rate * duration)

    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')

    return y


def extract_features(file_path, duration=2.5, sample_rate=22050, top_db=20):
    y = preprocess_audio(file_path, duration, sample_rate, top_db)

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sample_rate), axis=1)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sample_rate), axis=1)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)

    return np.hstack([mfccs, chroma, spectral_contrast, zero_crossing_rate])
