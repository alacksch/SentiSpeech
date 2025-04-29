import numpy as np

from utilities.build_ravdess import buildravdess
from utilities.feature_extraction import extract_features
from imblearn.over_sampling import SMOTE


def prepare_data(ravdess_path, duration=2.5, sample_rate=22050, top_db=20):
    df = buildravdess(ravdess_path)

    x = []
    y = []

    for _, row in df.iterrows():
        features = extract_features(row['relative_path'], duration, sample_rate, top_db)
        x.append(features)
        y.append(row['emotion'])
    sm = SMOTE()
    xResampled, yResampled = sm.fit_resample(x, y)

    return np.array(xResampled), np.array(yResampled)