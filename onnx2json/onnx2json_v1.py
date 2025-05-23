import onnx
from google.protobuf.json_format import MessageToJson
import argparse
import os


def convertToJson(onnx_model_path):
    # 解析为绝对路径，兼容相对路径输入
    onnx_model_path = os.path.abspath(onnx_model_path)

    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_model_path)
    message = MessageToJson(onnx_model)

    # 获取模型名称
    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]

    # 输出目录为 json_models
    json_output_dir = "./json_models"
    os.makedirs(json_output_dir, exist_ok=True)

    # 构建 JSON 文件路径
    json_model_path = os.path.join(json_output_dir, f"{model_name}_onnx2json_v1.json")

    # 写入 JSON 文件
    with open(json_model_path, "w") as fo:
        fo.write(message)

    print("✅ JSON 已保存到：", json_model_path)


def parseArgs():
    parser = argparse.ArgumentParser(description='ONNX model to JSON')
    parser.add_argument("--onnx_model", type=str, required=True, help="Path to ONNX model (relative or absolute)")
    args = parser.parse_args()
    print("📦 ONNX 模型路径：", args.onnx_model)
    return args.onnx_model


if __name__ == "__main__":
    print(">>>>>>>>>>> 开始转换 ONNX 模型为 JSON <<<<<<<<<<<<<")
    onnx_model_path = parseArgs()
    convertToJson(onnx_model_path)

# Run in Terminal
# python onnx2json_v1.py --onnx_model ./onnx_models/resnet50.onnx

