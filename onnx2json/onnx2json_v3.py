import onnx
import json
from onnx import numpy_helper

# 输入输出路径
onnx_path = "/onnx2json/onnx_models/resnet50.onnx"
json_path = "/onnx2json/json_models/resnet50_onnx2json_v3.json"

def onnx_to_json(onnx_path, json_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    json_model = {
        "network": [],
        "tensors": {},
        "inputs": [],
        "outputs": []
    }

    # 权重 initializer
    initializers = {init.name: numpy_helper.to_array(init).tolist() for init in graph.initializer}
    initializer_dims = {init.name: list(init.dims) for init in graph.initializer}

    # 节点处理
    for node in graph.node:
        op = {
            "name": node.name if node.name else node.output[0],
            "type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attrs": {}
        }

        # 属性提取
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                op["attrs"][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                op["attrs"][attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INTS:
                op["attrs"][attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                op["attrs"][attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                op["attrs"][attr.name] = attr.s.decode("utf-8")

        json_model["network"].append(op)

    # 所有张量 shape
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        name = value_info.name
        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        json_model["tensors"][name] = {
            "shape": shape
        }

    # 加入权重 tensor
    for name, data in initializers.items():
        json_model["tensors"][name] = {
            "shape": initializer_dims[name],
            "data": data
        }

    # 模型输入输出
    json_model["inputs"] = [i.name for i in graph.input if i.name not in initializers]
    json_model["outputs"] = [o.name for o in graph.output]

    # 保存JSON
    with open(json_path, "w") as f:
        json.dump(json_model, f, indent=2)
    print(f"✅ JSON 已保存至: {json_path}")

# 执行
onnx_to_json(onnx_path, json_path)