from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from utilities.build_data import prepare_data
import joblib

X, y = prepare_data('../audio_speech_actors_01-24')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

best_model = grid_search.best_estimator_

model_filename = 'packages/emotion_svm_model.joblib'
joblib.dump(best_model, model_filename)

scaler_filename = 'packages/scaler.joblib'
joblib.dump(scaler, scaler_filename)

label_encoder_filename = 'packages/label_encoder.joblib'
joblib.dump(label_encoder, label_encoder_filename)

print("Model, scaler, and label encoder have been saved successfully.")
