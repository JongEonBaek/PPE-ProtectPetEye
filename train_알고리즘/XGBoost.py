import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

# 1. 데이터 경로 설정 (로컬 환경)
# 데이터(.npy) 파일을 `data/` 폴더에 위치시켜 주세요.
base_dir = "./data/feature_embedding/"
X_train_path = os.path.join(base_dir, 'X_train.npy')
y_train_path = os.path.join(base_dir, 'y_train.npy')
X_test_path  = os.path.join(base_dir, 'X_test.npy')
y_test_path  = os.path.join(base_dir, 'y_test.npy')

# 2. Numpy 배열 로드
X_train = np.load(X_train_path)
y_train = np.load(y_train_path)
X_test  = np.load(X_test_path)
y_test  = np.load(y_test_path)

print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

# 3. XGBoost 모델 정의
model = XGBClassifier(
    objective='multi:softmax',
    num_class=11,
    eval_metric='mlogloss',
    verbosity=1
)

# 4. 하이퍼파라미터 그리드 정의
param_grid = {
    'max_depth': [3, 6],
    'eta': [0.1, 0.05, 0.01],
    'n_estimators': [50, 100, 150]
}

# 5. GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1,
    return_train_score=True
)

# 6. 그리드 서치 실행
print("Starting GridSearchCV...")
grid_search.fit(X_train, y_train)

# 7. 결과 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Hyperparameters: {best_params}")
print(f"Best CV Accuracy: {best_score:.4f}")

# 8. 상세 CV 결과 함수
def print_detailed_results(gs):
    cv = gs.cv
    results = gs.cv_results_
    print("\nDetailed CV results:")
    for i, params in enumerate(results['params']):
        print(f"Params: {params}")
        print(f"  Mean Train Acc: {results['mean_train_score'][i]:.4f}")
        print(f"  Mean Test Acc:  {results['mean_test_score'][i]:.4f}")
        for fold in range(cv):
            train_score = results[f'split{fold}_train_score'][i]
            test_score  = results[f'split{fold}_test_score'][i]
            print(f"    Fold {fold}: train={train_score:.4f}, val={test_score:.4f}")
        print('-'*40)

print_detailed_results(grid_search)

# 9. 최적 모델로 테스트 예측 및 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test, y_pred)
print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. 모델 저장
model_file = os.path.join(base_dir, 'xgboost_best_model.json')
best_model.save_model(model_file)
print(f"Saved model to {model_file}")

# 11. 메타데이터 저장
data = grid_search.cv_results_
metadata = {
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'best_params': best_params,
    'best_cv_accuracy': best_score,
    'classification_report': classification_report(y_test, y_pred, output_dict=True),
    'cv_results': []
}
for i, params in enumerate(data['params']):
    entry = {
        'params': params,
        'mean_train_score': data['mean_train_score'][i],
        'mean_test_score': data['mean_test_score'][i],
        'fold_scores': {
            f'fold_{fold}': {
                'train': data[f'split{fold}_train_score'][i],
                'validation': data[f'split{fold}_test_score'][i]
            }
            for fold in range(grid_search.cv)
        }
    }
    metadata['cv_results'].append(entry)

meta_file = os.path.join(base_dir, 'xgboost_best_model_metadata.json')
with open(meta_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)
print(f"Saved metadata to {meta_file}")