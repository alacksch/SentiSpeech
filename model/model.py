import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from utilities.build_data import prepare_data
import joblib

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_speech_actors_01-24'))
X, y = prepare_data(data_path)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y_encoded)

best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
os.makedirs('packages', exist_ok=True)
model_filename = 'packages/emotion_svm_model.joblib'
joblib.dump(best_model, model_filename)

scaler_filename = 'packages/scaler.joblib'
joblib.dump(scaler, scaler_filename)

label_encoder_filename = 'packages/label_encoder.joblib'
joblib.dump(label_encoder, label_encoder_filename)

print("Model, scaler, and label encoder have been saved successfully.")
