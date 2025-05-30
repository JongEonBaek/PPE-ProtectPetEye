import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request

# Flask 앱 생성
app = Flask(__name__)

# 1) 클래스 설정
NUM_CLASSES = 11
CLASS_NAMES = [
    "결막염","궤양성각막질환","무질환(정상)","백내장","비궤양성각막질환",
    "색소침착성각막염","안검내반증","안검염","안검종양","유루증","핵경화"
]

# 2) 디바이스 및 전처리
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

# 3) 모델 불러오기
model = models.resnet50(weights=None, num_classes=NUM_CLASSES)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load("best_finetuned_resnet.pth", map_location=device))
model.to(device).eval()

# 4) 라우트: 업로드 폼
@app.route("/")
def index():
    return render_template("index.html")

# 5) 라우트: 파일 업로드 후 예측
@app.route("/predict", methods=["POST"])
def predict():
    # 업로드된 파일 읽기
    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 전처리 & 추론
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=4)

    # 결과 딕셔너리
    results = [(CLASS_NAMES[idx], float(probs[idx])) for idx in topk.indices]

    # 템플릿에 이미지(바이너리)와 결과 전달
    # base64로 인코딩하면 다시 띄울 수 있습니다.
    import base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return render_template("index.html", results=results, img_data=img_b64)

# 6) 앱 실행
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
