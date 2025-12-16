import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import cv2  # 確保 cv2 (OpenCV) 模組已引入
import utils  # 【修復點一：添加 utils 模組引入】

# --- 1. 常量設定 ---
MODEL_PATH = 'flower_classifier.pth'
CLASS_NAMES_PATH = 'class_names.txt'
TEST_DATA_DIR = './dataset/test'

# 確保所有必要的檔案都存在
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
    st.error("❌ 模型或類別名稱檔案遺失！請檢查是否已成功推送 flower_classifier.pth 和 class_names.txt。")
    st.stop()

# --- 2. 數據載入和初始化（使用 st.cache_resource 解決重複載入） ---

@st.cache_resource
def load_model():
    """載入微調後的 ResNet50 模型並設定為評估模式"""
    # 載入模型結構，使用 IMAGENET1K_V1 權重作為起點
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 修改最後一層全連接層以匹配 102 個類別
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 102)

    # 載入訓練好的權重，強制在 CPU 上運行
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"❌ 無法載入模型權重: {e}")
        st.stop()

    model.eval()  # 設定為評估模式
    return model

@st.cache_resource
def load_class_names():
    """載入花卉類別名稱"""
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# 初始化模型和類別名稱
model_ft = load_model()
class_names = load_class_names()

# 圖像預處理轉換
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. 核心預測函數 ---

def predict_image(image_pil):
    """對 PIL 圖片進行預測"""
    input_tensor = data_transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model_ft(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 取得最高機率和類別索引
        top_p, top_class_idx = probabilities.topk(1, dim=1)
        
        # 轉換為 Python 標準類型
        predicted_index = top_class_idx.item()
        confidence = top_p.item()
        predicted_name = class_names[predicted_index]
        
        return predicted_name, confidence, predicted_index

# --- 4. 隨機選圖函數 (從 ./dataset/test/ 中選取) ---

def get_random_test_image_path(test_data_dir):
    """從測試集目錄中隨機選取一張圖片的路徑"""
    try:
        all_files = os.listdir(test_data_dir)
        
        # 篩選出圖片檔案
        image_files = [f for f in all_files 
                       if os.path.isfile(os.path.join(test_data_dir, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return None, f"測試集目錄 '{test_data_dir}' 中找不到任何圖片檔案。"
            
        random_image_file = random.choice(image_files)
        full_path = os.path.join(test_data_dir, random_image_file)
        return full_path, None
        
    except Exception as e:
        return None, f"讀取測試集時發生錯誤: {e}"

# --- 5. Streamlit UI 結構 ---

st.title("🌺 深度學習花卉辨識器 (HW5)")

# 初始化 session state 來儲存圖片路徑和熱圖顯示狀態
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'show_cam' not in st.session_state:
    st.session_state.show_cam = False

# --- 圖片選擇與隨機選圖 ---
st.header("🖼️ 選擇花卉圖片")

uploaded_file = st.file_uploader("上傳一張圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state.image_path = uploaded_file
    st.session_state.show_cam = False # 上傳新圖，重設 CAM 狀態

# 側邊欄控制
with st.sidebar:
    st.header("🕹️ 應用程式控制")
    
    # 隨機選圖按鈕 (確保 key 唯一)
    if st.button("🎲 隨機選取測試圖片", key="random_btn_final"): 
        random_path, error = get_random_test_image_path(TEST_DATA_DIR)
        
        if error:
            st.error(error)
        else:
            st.session_state.image_path = random_path
            st.session_state.show_cam = False
            st.rerun() 
            
    # CAM 顯示控制按鈕 (確保 key 唯一)
    if st.session_state.image_path:
        if st.button("🔥 顯示 Grad-CAM 熱圖", key="cam_btn_final"):
            st.session_state.show_cam = not st.session_state.show_cam
            # 這裡不使用 rerun，讓邏輯在主腳本中執行

# --- 6. 圖片處理與結果顯示 ---

current_image = None
if st.session_state.image_path:
    if isinstance(st.session_state.image_path, str):
        # 處理本地檔案路徑 (隨機選圖)
        current_image = Image.open(st.session_state.image_path).convert('RGB')
    else:
        # 處理 uploaded_file 物件 (用戶上傳)
        current_image = Image.open(st.session_state.image_path).convert('RGB')

if current_image:
    # 預測
    predicted_name, confidence, predicted_index = predict_image(current_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始圖片")
        st.image(current_image, caption="待辨識的花卉", use_column_width=True)

    with col2:
        st.subheader("預測結果")
        st.metric(label="預測花卉", value=predicted_name)
        st.metric(label="信心度", value=f"{confidence:.2%}")
        st.markdown(f"---")
        
        # 【修復點二：正確調用 Grad-CAM 邏輯】
        if st.session_state.show_cam:
            try:
                # 調用 utils.py 中定義的 generate_grad_cam 函數
                cam_image = utils.generate_grad_cam(
                    model_ft,           # PyTorch 模型
                    current_image,      # 原始 PIL 圖片
                    predicted_index,    # 預測的類別索引
                    data_transform      # 圖像預處理
                ) 
                
                st.subheader("🔥 Grad-CAM 熱圖")
                # 顯示由 utils 函數返回的 cam_image
                st.image(cam_image, caption="Grad-CAM 視覺化結果", use_column_width=True) 

            except Exception as e:
                st.error(f"❌ Grad-CAM 運算出錯: {e}")
                st.exception(e) # 顯示完整的錯誤堆疊資訊
                

else:
    st.info("請在左側上傳圖片或點擊按鈕隨機選取圖片開始辨識。")
