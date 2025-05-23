import torch
import torchvision.models as models
import os

# Step 1: 创建输出目录
onnx_dir = "../onnx_models"
os.makedirs(onnx_dir, exist_ok=True)

# Step 2: 加载 PyTorch 预训练 ResNet50 模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置为评估模式

# Step 3: 创建一个示例输入（batch_size=1, 3 channels, 224x224）
dummy_input = torch.randn(1, 3, 224, 224)

# Step 4: 设置导出路径
onnx_output_path = os.path.join(onnx_dir, "resnet50.onnx")

# Step 5: 导出为 ONNX 模型
torch.onnx.export(
    model,                         # PyTorch 模型
    dummy_input,                   # 示例输入
    onnx_output_path,              # 输出文件路径
    export_params=False,            # 是否导出权重
    opset_version=11,              # ONNX opset 版本
    do_constant_folding=True,      # 是否执行常量折叠优化
    input_names=['data_input'],    # 输入名称
    output_names=['logits'],       # 输出名称
    dynamic_axes={                 # 设置动态维度（可选）
        'data_input': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)

print(f"✅ 成功导出模型到: {onnx_output_path}")