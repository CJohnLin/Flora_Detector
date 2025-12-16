import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import random
import io # 用於處理UploadedFile物件

# --- 1. 配置與模型路徑 ---
MODEL_PATH = 'flower_classifier.pth'
CLASS_NAMES_FILE = 'class_names.txt' 
NUM_CLASSES = 102 # 確保與您訓練時的類別數量一致
TARGET_LAYER = 'layer4' # ResNet 的目標捲積層
TEST_DATA_DIR = './dataset/test' # 測試資料集路徑


# --- 2. 輔助函數：模型載入與圖像轉換 ---

# 圖像轉換設定 (與訓練時的驗證集轉換必須一致)
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(path, num_classes):
    """載入 PyTorch 模型 (使用 ResNet50 結構)"""
    try:
        model = models.resnet50(weights=None) 
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        
        # 載入模型權重到 CPU
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        
        return model
    except FileNotFoundError:
        st.error(f"❌ 模型檔案未找到: {path}。請先執行 train_flower_model.py。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 載入模型失敗: {e}")
        st.stop()

@st.cache_data
def load_class_names(file_path):
    """載入花卉類別名稱"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.error(f"❌ 類別名稱檔案未找到: {file_path}")
        st.stop()

# 載入模型和類別名稱
model = load_model(MODEL_PATH, NUM_CLASSES)
class_names = load_class_names(CLASS_NAMES_FILE)


# --- 3. 核心功能：Grad-CAM 實作 ---

def generate_grad_cam(model, input_image_tensor, target_class_idx, target_layer_name):
    """計算 Grad-CAM 熱圖"""
    feature_map = None
    gradient = None

    # 定義 Hook 函數來擷取特徵圖和梯度
    def save_feature_map(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()
    
    def save_gradient(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0].detach()

    # 找到目標層
    target_layer = dict(model.named_modules())[target_layer_name]

    # 註冊 hooks
    feature_hook = target_layer.register_forward_hook(save_feature_map)
    gradient_hook = target_layer.register_backward_hook(save_gradient)

    # 前向傳播
    output = model(input_image_tensor)
    
    # 後向傳播 (計算目標類別的梯度)
    model.zero_grad()
    one_hot = torch.zeros(output.shape)
    one_hot[:, target_class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # 移除 hooks
    feature_hook.remove()
    gradient_hook.remove()

    # 計算 Grad-CAM 權重 (Alpha)
    pooled_gradients = torch.mean(gradient, dim=[2, 3], keepdim=True) 
    
    # 產生 CAM 
    cam = (feature_map * pooled_gradients).sum(dim=1, keepdim=True) 
    cam = torch.relu(cam)

    # 歸一化 CAM
    cam = cam / (cam.max() + 1e-8) 
    return cam.squeeze().cpu().numpy()

def overlay_heatmap(original_img, cam_mask):
    """將 Grad-CAM 熱圖覆蓋到原始圖片上"""
    img = np.array(original_img.convert("RGB"))
    H, W, _ = img.shape
    
    heatmap = cv2.resize(cam_mask, (W, H))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 結合熱圖和原始圖片 (Weighted Overlay)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(superimposed_img)


# --- 4. 隨機選圖函數 ---

def get_random_test_image_path(test_data_dir):
# ------------------------------------
    """從測試集目錄中隨機選取一張圖片的路徑 (直接從根目錄抽取)"""
    try:
        # 1. 取得 test 資料夾根目錄下的所有檔案
        all_files = os.listdir(test_data_dir)
        
        # 2. 過濾出所有有效的圖片檔案
        image_files = [f for f in all_files 
                       if os.path.isfile(os.path.join(test_data_dir, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return None, f"測試集目錄 '{test_data_dir}' 中找不到任何圖片檔案。"
            
        # 3. 隨機選擇一張圖片
        random_image_file = random.choice(image_files)
        
        # 返回該圖片的完整路徑
        full_path = os.path.join(test_data_dir, random_image_file)
        return full_path, None
        
    except Exception as e:
        return None, f"讀取測試集時發生錯誤: {e}"

# =========================================================
# === STREAMLIT UI 主體 ===
# =========================================================

st.set_page_config(page_title="🌸 花卉辨識器 (Grad-CAM 解釋)", layout="wide")

st.title("🌸 Q1 — 花卉辨識器 (Grad-CAM 解釋)")
st.markdown("上傳一張圖片或從測試集隨機選取一張，進行辨識和模型解釋。")

# 初始化 Session State 來儲存當前圖片來源
if 'image_source' not in st.session_state:
    st.session_state['image_source'] = None
    st.session_state['is_random'] = False

st.header("Upload or Select Image")

# --- 隨機選圖按鈕 ---
col_rand, col_upload = st.columns([1, 2])

with col_rand:
    if st.button("🎲 隨機選取測試圖片", use_container_width=True, help=f"從 {TEST_DATA_DIR} 隨機載入"):
        random_path, error = get_random_test_image_path(TEST_DATA_DIR)
        if error:
            st.error(error)
        elif random_path:
            st.session_state['image_source'] = random_path
            st.session_state['is_random'] = True
            st.toast(f"已從測試集隨機選取圖片。", icon='✅')

# --- 圖片上傳 ---
with col_upload:
    uploaded_file = st.file_uploader("或上傳您自己的圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 如果使用者上傳了新檔案，則更新狀態
    st.session_state['image_source'] = uploaded_file
    st.session_state['is_random'] = False


# --- 處理和顯示圖片 ---
image_to_process = None

if st.session_state['image_source']:
    source = st.session_state['image_source']
    
    if isinstance(source, str): # 隨機選圖 (路徑)
        image_to_process = Image.open(source)
        caption_text = f"隨機測試圖片 (檔案: {os.path.basename(source)})"
            
    else: # 上傳檔案 (UploadedFile 物件)
        image_to_process = Image.open(io.BytesIO(source.read())) # 從 BytesIO 讀取
        caption_text = f'使用者上傳的圖片 ({source.name})'
        source.seek(0) # 重置檔案指標，防止重複讀取

# --- 顯示結果區塊 ---

if image_to_process is not None:
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("🖼️ 原始圖片")
        st.image(image_to_process, caption=caption_text, use_column_width=True)
        
    with col2:
        st.subheader("💡 辨識與解釋結果")
        
        # 進行預測
        input_tensor = image_transforms(image_to_process).unsqueeze(0) 
        
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_perc = f"{confidence.item() * 100:.2f}%"

        st.success(f"**預測花卉:** {predicted_class}")
        st.info(f"**信心值:** {confidence_perc}")

        st.markdown("---")
        
        # --- Grad-CAM 計算與顯示按鈕 ---
        if st.button("🔥 顯示 Grad-CAM 熱圖", type="primary"):
            with st.spinner('計算 Grad-CAM 熱圖中...'):
                try:
                    # Grad-CAM 計算
                    # 必須重新運行 image_transforms，因為張量在之前已經使用過 (不能 retain_graph=True)
                    cam_mask = generate_grad_cam(
                        model, 
                        image_transforms(image_to_process).unsqueeze(0),
                        predicted_idx.item(), 
                        TARGET_LAYER
                    )
                    
                    heatmap_image = overlay_heatmap(image_to_process, cam_mask)
                    
                    st.subheader("🔥 Grad-CAM 熱圖解釋")
                    st.image(heatmap_image, caption="模型關注區域 (熱圖越紅表示越重要)", use_column_width=True)
                    st.caption(f"Grad-CAM 使用的模型層: `{TARGET_LAYER}`")
                    
                except Exception as e:
                    st.error(f"Grad-CAM 計算失敗，請檢查模型和 PyTorch 版本是否兼容: {e}")

# --- HW5 共同要求提醒 ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 HW5 共同要求")
st.sidebar.success("1. ChatGPT / AI Agent 對話過程 (必要)")
st.sidebar.success("2. GitHub Repository (必要)")
st.sidebar.success("3. Streamlit.app Demo 連結 (必要)")

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
import random
import io # 用於處理UploadedFile物件

# --- 1. 配置與模型路徑 ---
MODEL_PATH = 'flower_classifier.pth'
CLASS_NAMES_FILE = 'class_names.txt' 
NUM_CLASSES = 102 # 確保與您訓練時的類別數量一致
TARGET_LAYER = 'layer4' # ResNet 的目標捲積層
TEST_DATA_DIR = './dataset/test' # 測試資料集路徑


# --- 2. 輔助函數：模型載入與圖像轉換 ---

# 圖像轉換設定 (與訓練時的驗證集轉換必須一致)
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(path, num_classes):
    """載入 PyTorch 模型 (使用 ResNet50 結構)"""
    try:
        model = models.resnet50(weights=None) 
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        
        # 載入模型權重到 CPU
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        
        return model
    except FileNotFoundError:
        st.error(f"❌ 模型檔案未找到: {path}。請先執行 train_flower_model.py。")
        st.stop()
    except Exception as e:
        st.error(f"❌ 載入模型失敗: {e}")
        st.stop()

@st.cache_data
def load_class_names(file_path):
    """載入花卉類別名稱"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.error(f"❌ 類別名稱檔案未找到: {file_path}")
        st.stop()

# 載入模型和類別名稱
model = load_model(MODEL_PATH, NUM_CLASSES)
class_names = load_class_names(CLASS_NAMES_FILE)


# --- 3. 核心功能：Grad-CAM 實作 ---

def generate_grad_cam(model, input_image_tensor, target_class_idx, target_layer_name):
    """計算 Grad-CAM 熱圖"""
    feature_map = None
    gradient = None

    # 定義 Hook 函數來擷取特徵圖和梯度
    def save_feature_map(module, input, output):
        nonlocal feature_map
        feature_map = output.detach()
    
    def save_gradient(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0].detach()

    # 找到目標層
    target_layer = dict(model.named_modules())[target_layer_name]

    # 註冊 hooks
    feature_hook = target_layer.register_forward_hook(save_feature_map)
    gradient_hook = target_layer.register_backward_hook(save_gradient)

    # 前向傳播
    output = model(input_image_tensor)
    
    # 後向傳播 (計算目標類別的梯度)
    model.zero_grad()
    one_hot = torch.zeros(output.shape)
    one_hot[:, target_class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # 移除 hooks
    feature_hook.remove()
    gradient_hook.remove()

    # 計算 Grad-CAM 權重 (Alpha)
    pooled_gradients = torch.mean(gradient, dim=[2, 3], keepdim=True) 
    
    # 產生 CAM 
    cam = (feature_map * pooled_gradients).sum(dim=1, keepdim=True) 
    cam = torch.relu(cam)

    # 歸一化 CAM
    cam = cam / (cam.max() + 1e-8) 
    return cam.squeeze().cpu().numpy()

def overlay_heatmap(original_img, cam_mask):
    """將 Grad-CAM 熱圖覆蓋到原始圖片上"""
    img = np.array(original_img.convert("RGB"))
    H, W, _ = img.shape
    
    heatmap = cv2.resize(cam_mask, (W, H))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 結合熱圖和原始圖片 (Weighted Overlay)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(superimposed_img)


# --- 4. 隨機選圖函數 ---

def get_random_test_image_path(test_data_dir):
# ------------------------------------
    """從測試集目錄中隨機選取一張圖片的路徑 (直接從根目錄抽取)"""
    try:
        # 1. 取得 test 資料夾根目錄下的所有檔案
        all_files = os.listdir(test_data_dir)
        
        # 2. 過濾出所有有效的圖片檔案
        image_files = [f for f in all_files 
                       if os.path.isfile(os.path.join(test_data_dir, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return None, f"測試集目錄 '{test_data_dir}' 中找不到任何圖片檔案。"
            
        # 3. 隨機選擇一張圖片
        random_image_file = random.choice(image_files)
        
        # 返回該圖片的完整路徑
        full_path = os.path.join(test_data_dir, random_image_file)
        return full_path, None
        
    except Exception as e:
        return None, f"讀取測試集時發生錯誤: {e}"

# =========================================================
# === STREAMLIT UI 主體 ===
# =========================================================

st.set_page_config(page_title="🌸 花卉辨識器 (Grad-CAM 解釋)", layout="wide")

st.title("🌸 Q1 — 花卉辨識器 (Grad-CAM 解釋)")
st.markdown("上傳一張圖片或從測試集隨機選取一張，進行辨識和模型解釋。")

# 初始化 Session State 來儲存當前圖片來源
if 'image_source' not in st.session_state:
    st.session_state['image_source'] = None
    st.session_state['is_random'] = False

st.header("Upload or Select Image")

# --- 隨機選圖按鈕 ---
col_rand, col_upload = st.columns([1, 2])

with col_rand:
    if st.button("🎲 隨機選取測試圖片", use_container_width=True, help=f"從 {TEST_DATA_DIR} 隨機載入"):
        random_path, error = get_random_test_image_path(TEST_DATA_DIR)
        if error:
            st.error(error)
        elif random_path:
            st.session_state['image_source'] = random_path
            st.session_state['is_random'] = True
            st.toast(f"已從測試集隨機選取圖片。", icon='✅')

# --- 圖片上傳 ---
with col_upload:
    uploaded_file = st.file_uploader("或上傳您自己的圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 如果使用者上傳了新檔案，則更新狀態
    st.session_state['image_source'] = uploaded_file
    st.session_state['is_random'] = False


# --- 處理和顯示圖片 ---
image_to_process = None

if st.session_state['image_source']:
    source = st.session_state['image_source']
    
    if isinstance(source, str): # 隨機選圖 (路徑)
        image_to_process = Image.open(source)
        caption_text = f"隨機測試圖片 (檔案: {os.path.basename(source)})"
            
    else: # 上傳檔案 (UploadedFile 物件)
        image_to_process = Image.open(io.BytesIO(source.read())) # 從 BytesIO 讀取
        caption_text = f'使用者上傳的圖片 ({source.name})'
        source.seek(0) # 重置檔案指標，防止重複讀取

# --- 顯示結果區塊 ---

if image_to_process is not None:
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("🖼️ 原始圖片")
        st.image(image_to_process, caption=caption_text, use_column_width=True)
        
    with col2:
        st.subheader("💡 辨識與解釋結果")
        
        # 進行預測
        input_tensor = image_transforms(image_to_process).unsqueeze(0) 
        
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze()
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_perc = f"{confidence.item() * 100:.2f}%"

        st.success(f"**預測花卉:** {predicted_class}")
        st.info(f"**信心值:** {confidence_perc}")

        st.markdown("---")
        
        # --- Grad-CAM 計算與顯示按鈕 ---
        if st.button("🔥 顯示 Grad-CAM 熱圖", type="primary"):
            with st.spinner('計算 Grad-CAM 熱圖中...'):
                try:
                    # Grad-CAM 計算
                    # 必須重新運行 image_transforms，因為張量在之前已經使用過 (不能 retain_graph=True)
                    cam_mask = generate_grad_cam(
                        model, 
                        image_transforms(image_to_process).unsqueeze(0),
                        predicted_idx.item(), 
                        TARGET_LAYER
                    )
                    
                    heatmap_image = overlay_heatmap(image_to_process, cam_mask)
                    
                    st.subheader("🔥 Grad-CAM 熱圖解釋")
                    st.image(heatmap_image, caption="模型關注區域 (熱圖越紅表示越重要)", use_column_width=True)
                    st.caption(f"Grad-CAM 使用的模型層: `{TARGET_LAYER}`")
                    
                except Exception as e:
                    st.error(f"Grad-CAM 計算失敗，請檢查模型和 PyTorch 版本是否兼容: {e}")

# --- HW5 共同要求提醒 ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 HW5 共同要求")
st.sidebar.success("1. ChatGPT / AI Agent 對話過程 (必要)")
st.sidebar.success("2. GitHub Repository (必要)")
st.sidebar.success("3. Streamlit.app Demo 連結 (必要)")
st.sidebar.markdown("請將所有檔案推送到 GitHub，並部署 Streamlit。")