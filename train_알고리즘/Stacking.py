import torch
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. 전처리된 데이터 불러오기
X_train_np = np.load("./data/feature_embedding/X_train.npy")
y_train_np = np.load("./data/feature_embedding/y_train.npy")
X_test_np = np.load("./data/feature_embedding/X_test.npy")
y_test_np = np.load("./data/feature_embedding/y_test.npy")

# 2.1. Stacking 기법에 사용할 Base Learners 정의
base_learners = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),  # k=5로 고정
    ('svm', SVC(probability=True, kernel='rbf', C=0.1)),  # C=0.1로 고정
    ('rf', RandomForestClassifier(max_depth=10, n_estimators=50))  # max_depth=10, n_estimators=50으로 고정
]

# 2.2. Meta Learner 정의
meta_learner = LogisticRegression(solver='lbfgs', C=0.1)

# 3.1. Stacking Classifier 구성
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# 4. 모델 학습
print("Training Stacking Classifier...")
stacking_clf.fit(X_train_np, y_train_np)

# 5.1. Train 데이터 평가
y_train_pred = stacking_clf.predict(X_train_np)
train_accuracy = accuracy_score(y_train_np, y_train_pred)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# 5.2. Test 데이터 평가
y_pred = stacking_clf.predict(X_test_np)
test_accuracy = accuracy_score(y_test_np, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 6. 학습된 모델 저장
model_path = "/data/eyes_data/best_stacking_model.pkl"
joblib.dump(stacking_clf, model_path)
print(f"Model saved to {model_path}")