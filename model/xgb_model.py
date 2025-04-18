import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.build_data import prepare_data
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_speech_actors_01-24'))
X, y = prepare_data(data_path)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

xgb = XGBClassifier(eval_metric='mlogloss')

grid_search = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X, y_encoded)

best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)
print("Accuracy:", grid_search.best_score_)