import onnx
from onnx.shape_inference import infer_shapes
import numpy as np

from google.protobuf.json_format import MessageToJson, Parse
import argparse
import os


def convertToJson(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    message = MessageToJson(onnx_model)

    dir_path, model_name = os.path.split(onnx_model_path)

    json_model_name = model_name.replace('.onnx', '.json')
    json_model_path = dir_path + "/" + json_model_name
    with open(json_model_path, "w") as fo:
        fo.write(message)
    print("Save json to ", json_model_path)


def parseArgs():
    parser = argparse.ArgumentParser(description='ONNX model to json')
    parser.add_argument("--onnx_model", type=str, required=True, help="Root path of ONNX model.")
    args = parser.parse_args()
    print("ONNX model: ", args.onnx_model)
    return args.onnx_model


if __name__ == "__main__":
    print(">>>>>>>>>>> Begin ONNX model convert <<<<<<<<<<<<<")
    onnx_model_path = parseArgs()
    if not os.path.isabs(onnx_model_path):
        raise ValueError("ERROR! --onnx_model should be the root path.")

    convertToJson(onnx_model_path)
