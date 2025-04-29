import os
import joblib

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utilities.build_data import prepare_data

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_speech_actors_01-24'))

X, y = prepare_data(data_path)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('svc', SVC())])

param_grid = {
    'pca__n_components': [0.90, 0.95, 0.99],
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svc__gamma': ['scale', 'auto']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X, y_encoded)

print("CV accuracy with the best parameters possible:", grid.best_score_)

joblib.dump(grid.best_estimator_, 'packages/emotion_pca_svm_pipeline.joblib')
joblib.dump(encoder, 'packages/label_encoder.joblib')