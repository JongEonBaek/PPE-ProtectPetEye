import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# 1. 데이터 로드 (로컬 경로로 수정)
data_dir = "./data/feature_embedding/" # data/X_train.npy 등 파일 위치
x_train = np.load(os.path.join(data_dir, 'X_train.npy'))
x_test  = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape:  {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# 2. 기본 RandomForest 학습 및 평가
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
y_train_pred = rf_model.predict(x_train)
y_pred       = rf_model.predict(x_test)

print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy: ", accuracy_score(y_test, y_pred))

# 3. RandomizedSearchCV로 하이퍼파라미터 탐색
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 30, 50, 100],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rf, param_dist, n_iter=25, cv=5, verbose=2, random_state=42)
search.fit(x_train, y_train)
print("Best Parameters:", search.best_params_)
print("Best CV Score:", search.best_score_)
best_rf = search.best_estimator_

# 4. 최적 하이퍼파라미터로 5-Fold 교차검증 및 평가
best_params = search.best_params_
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_rf_model, x_train, y_train, cv=cv, scoring='accuracy')
print("CV Accuracy Scores:", scores)
print(f"Mean CV Accuracy: {scores.mean():.4f}")

best_rf_model.fit(x_train, y_train)
y_pred = best_rf_model.predict(x_test)

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)
ConfusionMatrixDisplay(conf_mat).plot()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))

# 5. 모델 저장 (n_estimators만 조정 원하면 변경)
model_path = os.path.join(os.getcwd(), 'best_randomforest_model.pkl')
joblib.dump(best_rf_model, model_path)
print(f"Model saved to {model_path}")
