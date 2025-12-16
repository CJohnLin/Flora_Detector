# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# ------------------------------------------------------------------------------
# 1. Grad-CAM 核心函數 (取得特徵圖和梯度)
# ------------------------------------------------------------------------------

class GradCAM:
    """
    實現 Grad-CAM 邏輯的類別
    它會註冊鉤子（hooks）來擷取目標層的特徵圖（Feature Map）和梯度（Gradient）
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = self.find_target_layer(target_layer_name)
        self.feature = None  # 用於儲存特徵圖
        self.gradient = None # 用於儲存梯度
        self.handlers = []   # 用於儲存註冊的鉤子

        self.register_hooks()

    def find_target_layer(self, target_layer_name):
        """根據名稱查找模型中的目標層 (ResNet50 的最後一個卷積層是 layer4)"""
        if target_layer_name == 'layer4':
            return self.model.layer4
        
        # 如果模型結構不同，這裡需要調整
        raise ValueError(f"Target layer '{target_layer_name}' not found or not supported.")

    def save_feature(self, module, input, output):
        """前向鉤子：儲存特徵圖"""
        self.feature = output.detach()

    def save_gradient(self, module, grad_in, grad_out):
        """後向鉤子：儲存梯度（通常取第一個輸出梯度）"""
        self.gradient = grad_out[0].detach()

    def register_hooks(self):
        """註冊前向和後向鉤子"""
        # 前向鉤子：在執行目標層之後儲存特徵圖
        self.handlers.append(self.target_layer.register_forward_hook(self.save_feature))
        
        # 後向鉤子：在計算目標層的梯度時儲存梯度
        self.handlers.append(self.target_layer.register_full_backward_hook(self.save_gradient))

    def remove_hooks(self):
        """清除鉤子以釋放資源"""
        for handle in self.handlers:
            handle.remove()

    def __call__(self, input_tensor, target_index=None):
        """
        執行前向和後向傳播，並計算 Grad-CAM
        :param input_tensor: 經過預處理的輸入圖像 Tensor
        :param target_index: 預測的類別索引
        """
        self.model.zero_grad()
        
        # 1. 前向傳播：取得輸出
        output = self.model(input_tensor)
        
        # 2. 確定目標輸出
        if target_index is None:
            # 如果未指定，則使用最高預測的類別
            target_index = output.argmax(dim=1).item()
        
        # 3. 創建反向傳播的目標 Tensor
        target_output = output[0, target_index]
        
        # 4. 後向傳播：計算目標輸出對特徵圖的梯度
        target_output.backward()

        # 5. 取得梯度和特徵圖
        gradients = self.gradient
        features = self.feature

        # 6. 計算權重：對每個特徵圖通道的梯度進行全域平均（Global Average Pooling）
        # shape: (通道, 高度, 寬度) -> (通道, 1, 1)
        weights = F.adaptive_avg_pool2d(gradients, 1)

        # 7. 構建 CAM：權重乘以特徵圖
        # shape: (1, 通道, 高度, 寬度)
        cam = (weights * features).sum(dim=1, keepdim=True)

        # 8. 激活函數：ReLU 處理，只保留正值
        cam = F.relu(cam)

        # 9. 調整尺寸：將 CAM 調整到輸入圖像的尺寸 (224x224)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 10. 標準化到 [0, 1] 範圍
        cam = cam.squeeze(0).squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # 清除鉤子
        self.remove_hooks()
        
        return cam.cpu().numpy()

# ------------------------------------------------------------------------------
# 2. 疊加函數 (使用 OpenCV 進行圖像處理)
# ------------------------------------------------------------------------------

def show_cam_on_image(img_pil, cam_np):
    """
    將 Grad-CAM 熱圖疊加到原始圖片上。
    :param img_pil: 原始輸入的 PIL 圖像 (224x224)
    :param cam_np: 計算得到的 CAM 熱圖（NumPy 陣列，範圍 [0, 1]）
    :return: 疊加後的 PIL 圖像
    """
    # 1. 確保圖片尺寸匹配
    img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255
    
    # 2. 將 CAM 轉換為 255 灰階圖像
    heatmap = np.uint8(255 * cam_np)
    
    # 3. 使用 OpenCV 應用顏色映射 (COLORMAP_JET)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 4. 轉換顏色空間：OpenCV 預設為 BGR，需要轉換為 RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 5. 疊加：使用權重疊加原始圖像和熱圖 (0.4 是熱圖透明度)
    cam_img_np = np.float32(heatmap) * 0.4 + np.float32(img * 255) * 0.6
    
    # 6. 標準化並轉換為 8 位元整數
    cam_img_np = cam_img_np / np.max(cam_img_np)
    cam_img_np = np.uint8(255 * cam_img_np)
    
    # 7. 轉換回 PIL 圖像格式
    return Image.fromarray(cam_img_np)

# ------------------------------------------------------------------------------
# 3. Streamlit 呼叫主函數
# ------------------------------------------------------------------------------

def generate_grad_cam(model, image_pil, predicted_index, data_transform):
    """
    在 Streamlit app.py 中調用這個函數來生成 CAM 圖像
    :param model: PyTorch 模型
    :param image_pil: 原始輸入的 PIL 圖片
    :param predicted_index: 預測的類別索引
    :param data_transform: 圖像預處理轉換
    :return: 疊加了 Grad-CAM 的 PIL 圖像
    """
    # 1. 設置設備 (確保在 CPU 上運行以避免 Streamlit 雲端問題)
    device = torch.device('cpu')
    model.to(device)
    
    # 2. 預處理輸入圖片
    input_tensor = data_transform(image_pil).unsqueeze(0).to(device)
    
    # 3. 創建 GradCAM 實例 (ResNet50 的目標層是 layer4)
    cam_generator = GradCAM(model, target_layer_name='layer4')
    
    # 4. 執行 Grad-CAM 計算
    cam_np = cam_generator(input_tensor, target_index=predicted_index)
    
    # 5. 將熱圖疊加到原始圖像上
    cam_image = show_cam_on_image(image_pil, cam_np)
    
    return cam_image
