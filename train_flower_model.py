import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import copy
from tqdm import tqdm
import json # 新增：用於讀取 cat_to_name.json

# --- 1. 配置與參數設定 ---

# ⚠️ 數據集根目錄。請確保此路徑與您的 'dataset' 資料夾相對正確
data_dir = './dataset' 
MODEL_SAVE_PATH = 'flower_classifier.pth'

# 訓練參數
BATCH_SIZE = 32
NUM_CLASSES = 102 # 根據 Oxford 102 資料集設定
NUM_EPOCHS = 25 
LEARNING_RATE = 0.001

# 設定設備 (使用 GPU 還是 CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 2. 資料前處理與載入 ---

# 定義訓練集和驗證集的圖像轉換
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        # 標準化參數 (與 app.py 必須一致)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    # 針對 'valid' 資料集
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 載入圖像資料集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'valid']} 

# 建立 DataLoaders (用於批次載入數據)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}


# --- 3. 處理類別名稱映射 (從數字編號轉換為花卉名稱) ---

# 取得 PyTorch 排序的類別索引 (此時是 ['1', '10', '100', ...] 等數字字串)
pytorch_class_ids = image_datasets['train'].classes 

# 載入類別名稱映射表 cat_to_name.json
try:
    with open('cat_to_name.json', 'r', encoding='utf-8') as f:
        cat_to_name = json.load(f)
except FileNotFoundError:
    print("⚠️ 警告: cat_to_name.json 未找到，將使用數字資料夾名稱作為類別名稱。")
    cat_to_name = {k: k for k in pytorch_class_ids} # 備用：直接使用數字作為名稱

# 根據 PyTorch 的類別 ID 順序，建立最終的 class_names 列表
# 這確保了模型的輸出索引 (0, 1, 2...) 對應正確的花卉名稱。
final_class_names = [cat_to_name[class_id] for class_id in pytorch_class_ids]

print(f"✅ 成功載入 {len(final_class_names)} 個類別名稱。")

# 儲存最終的 class_names.txt 檔案，供 app.py 使用
with open('class_names.txt', 'w', encoding='utf-8') as f:
    for name in final_class_names:
        f.write(f"{name}\n")
print("✅ class_names.txt 檔案已儲存。")


# --- 4. 定義訓練函數 ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 迭代訓練和驗證階段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 設置模型為訓練模式
                # 學習率調整器 (只在訓練階段執行)
                if scheduler:
                    scheduler.step() 
            else:
                model.eval()   # 設置模型為評估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代數據
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Phase'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向傳播 (在訓練階段開啟梯度追蹤)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 後向傳播與優化 (僅在訓練階段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度複製最佳模型 (基於驗證集)
            if phase == 'valid' and epoch_acc > best_acc: 
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 每當有提升就儲存模型 (作為檢查點)
                torch.save(model.state_dict(), MODEL_SAVE_PATH) 

    time_elapsed = time.time() - since
    print(f'\n訓練完成，總共耗時 {time_elapsed // 60:.0f} 分鐘 {time_elapsed % 60:.0f} 秒')
    print(f'最佳驗證準確度: {best_acc:.4f}')

    # 載入並返回最佳模型權重
    model.load_state_dict(best_model_wts)
    return model

# --- 5. 初始化模型與優化器 ---

# 載入預訓練的 ResNet50 模型 (使用 ImageNet 權重進行遷移學習)
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 替換最後一層全連接層，適應我們的 NUM_CLASSES
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model_ft = model_ft.to(device)

# 定義損失函數
criterion = nn.CrossEntropyLoss()

# 定義優化器
optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

# 定義學習率排程器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# --- 6. 開始訓練 ---

if __name__ == '__main__':
    print(f"--- 開始訓練模型 (設備: {device}) ---")
    
    # 執行訓練
    final_best_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    
    # 最終儲存最佳模型 (如果訓練過程中沒有儲存)
    torch.save(final_best_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ 最終最佳模型權重已儲存為 {MODEL_SAVE_PATH}")
    print("您可以運行 app.py 來部署花卉辨識器了。")