"""
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 前処理（リサイズ、正規化）
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# データセット読み込み
dataset = datasets.ImageFolder("chars", transform=transform)

# クラスラベル一覧（辞書化も可能）
print(dataset.class_to_idx)  # 例: {'3a2a': 0, '3a2b': 1, ...}

# データ分割（8:2で訓練／テスト）
from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# データローダー
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (64x64 → 32x32)
        x = self.pool(F.relu(self.conv2(x)))  # (32x32 → 64x64)
        x = self.pool(F.relu(self.conv3(x)))  # (64x64 → 128x128)
        x = self.pool(F.relu(self.conv4(x)))  # (128x128 → 256x256)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


import torch
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
print("Using device:", device)
if device.type == 'cuda':
    print("CUDA is available!")
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU not used. Using CPU.")
model = SimpleCNN(num_classes=len(dataset.classes)).to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(7):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "NN2.pth")