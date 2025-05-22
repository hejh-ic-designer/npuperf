import onnx
from onnx import helper, numpy_helper, shape_inference
from collections import defaultdict
import json
import numpy as np

def load_onnx_model(path):
    model = onnx.load(path)
    model = shape_inference.infer_shapes(model)
    return model

def get_tensor_shape_map(model):
    value_info = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    shape_map = {}
    for vi in value_info:
        shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = shape
    return shape_map

def normalize_op_type(op_type):
    # 标准化算子名称
    op_map = {
        'BatchFlatten': 'Reshape',
        'Flatten': 'Reshape',
        'Squeeze': 'Reshape',
        'Upsample': 'Resize'
    }
    return op_map.get(op_type, op_type)

def fuse_ops(nodes, shape_map):
    fused_nodes = []
    skip_idx = set()
    for i, node in enumerate(nodes):
        if i in skip_idx:
            continue
        fused_node = node

        # Conv + BiasAdd (Add)
        if node.op_type == 'Conv':
            if i + 1 < len(nodes) and nodes[i+1].op_type == 'Add':
                add_node = nodes[i+1]
                if is_bias_add(fused_node, add_node):
                    fused_node = fuse_conv_bias(fused_node, add_node)
                    skip_idx.add(i+1)

        # Conv + Mul / Add
        if fused_node.op_type == 'Conv':
            j = i + 1
            while j < len(nodes) and nodes[j].op_type in ['Add', 'Mul']:
                fused_node = fuse_conv_mul_add(fused_node, nodes[j])
                skip_idx.add(j)
                j += 1

        # Conv + Relu
        if fused_node.op_type == 'Conv' and i+1 < len(nodes) and nodes[i+1].op_type == 'Relu':
            fused_node = fuse_conv_relu(fused_node, nodes[i+1])
            skip_idx.add(i+1)

        fused_nodes.append(fused_node)
    return fused_nodes

def is_bias_add(conv_node, add_node):
    # 简化判断，仅适用于 Conv + bias Add
    return add_node.input[0] == conv_node.output[0] or add_node.input[1] == conv_node.output[0]

def fuse_conv_bias(conv_node, add_node):
    conv_node.name += "_fused_bias"
    conv_node.output[0] = add_node.output[0]
    return conv_node

def fuse_conv_mul_add(conv_node, node):
    conv_node.name += f"_fused_{node.op_type.lower()}"
    conv_node.output[0] = node.output[0]
    return conv_node

def fuse_conv_relu(conv_node, relu_node):
    conv_node.name += "_fused_relu"
    conv_node.output[0] = relu_node.output[0]
    return conv_node

def to_json_model(model, fused_nodes, shape_map):
    json_model = {
        "input_names": [i.name for i in model.graph.input],
        "layers": []
    }

    for node in fused_nodes:
        op = {
            "op_type": f"qnn.csi.{normalize_op_type(node.op_type.lower())}",
            "name": node.name,
            "inputs": [{"name": name, "dim": shape_map.get(name, []), "is_const": 0, "layout": "NCHW"} for name in node.input],
            "outputs": [{"name": name, "dim": shape_map.get(name, []), "is_const": 0, "layout": "NCHW"} for name in node.output],
            "attrs": {}
        }
        json_model["layers"].append(op)

    return json_model

def export_json(json_model, out_path):
    with open(out_path, "w") as f:
        json.dump(json_model, f, indent=2)

def convert_onnx_to_json(onnx_path, out_path):
    model = load_onnx_model(onnx_path)
    shape_map = get_tensor_shape_map(model)
    fused_nodes = fuse_ops(model.graph.node, shape_map)
    json_model = to_json_model(model, fused_nodes, shape_map)
    export_json(json_model, out_path)

# 示例调用
# convert_onnx_to_json("resnet50.onnx", "resnet50_converted.json")