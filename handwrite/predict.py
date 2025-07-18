import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

#どっちか
TARGET_PATH = "handwrite/2.png"
TARGET_CHAR = "3a2a"

# NN.pyと同じ
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (64x64 → 32x32)
        x = self.pool(F.relu(self.conv2(x)))  # (32x32 → 16x16)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 画像前処理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# クラスラベル（手動でリストを設定するか、ImageFolder から取得）
class_names = sorted(os.listdir("chars"))  # chars ディレクトリが必要

# モデルの準備
model = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("MODEL/NN.pth", map_location=device))
model.eval()

# 推論関数
def predict(image_path, word="予測結果："):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # バッチ次元追加

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    predicted_label = class_names[predicted.item()]
    print(f"{word} {predicted_label}")

# 使用例
print("Start Predict")
predict(TARGET_PATH)
"""
for i in range(200):
    predict(f"chars/{TARGET_CHAR}/img_{i:04}_{TARGET_CHAR}.png", f"{i:04}")  # 3桁ゼロ埋め
"""