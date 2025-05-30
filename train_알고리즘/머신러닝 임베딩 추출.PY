from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm
import numpy as np
import torchvision.models as models
import torch.nn as nn

# 데이터셋 경로 설정 
train_dir = './data/train'  # 학습 데이터 경로
test_dir = './data/test'    # 테스트 데이터 경로

# 데이터 전처리 변환 정의 (머신러닝에 적합한 방식)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 픽셀 값을 [0, 1]로 스케일링
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 제거
])

# 데이터셋 로드
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# 배치 크기 조정 (OOM 방지를 위해)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 학습된 ResNet 모델 로드 및 특징 추출용으로 수정
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 분류 레이어 제거
resnet.to(device)
resnet.eval()

# 데이터 로드 및 특징 벡터 추출
def extract_features(dataloader, model, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu())
            labels.append(targets)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

X_train, y_train = extract_features(train_loader, resnet, device)
X_test, y_test = extract_features(test_loader, resnet, device)

# NumPy로 변환
X_train_np = X_train.numpy()
y_train_np = y_train.numpy()
X_test_np = X_test.numpy()
y_test_np = y_test.numpy()

# Numpy 배열 저장
np.save("X_train.npy", X_train_np)
np.save("y_train.npy", y_train_np)
np.save("X_test.npy", X_test_np)
np.save("y_test.npy", y_test_np)
