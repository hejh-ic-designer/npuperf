import onnx
import json
import numpy as np
from onnx import numpy_helper, shape_inference

INPUT_ONNX_PATH = "./onnx_models/resnet50_export_params.onnx"
# 使用 export_params=True 导出的onnx模型
# graph.initializer 中包含所有的常量(weight, bias)
OUTPUT_JSON_PATH = "./json_models/resnet50_onnx2json_v5.json"

def get_tensor_shape(value_info):
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.HasField("dim_value") else 1)
    return shape

def build_tensor_map(graph):
    tensor_map = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        name = vi.name
        shape = get_tensor_shape(vi)
        tensor_map[name] = shape
    return tensor_map

def build_initializer_map(graph):
    initializer_map = {}
    for init in graph.initializer:
        name = init.name
        tensor = numpy_helper.to_array(init)
        initializer_map[name] = {
            "dim": list(tensor.shape),
            "data": tensor.tolist()
        }
    return initializer_map

def is_initializer(name, initializer_map):
    return name in initializer_map

def get_layout(shape, is_const):
    if not is_const:
        if len(shape) == 4:
            return "NCHW"
        elif len(shape) == 2:
            return "NC"
        elif len(shape) == 1:
            return "N"
    else:
        if len(shape) == 4:
            return "OIHW"
        elif len(shape) == 1:
            return "O"
        elif len(shape) == 2:
            return "OI"
    return ""

def parse_onnx_model(model):
    graph = model.graph
    tensor_map = build_tensor_map(graph)
    initializer_map = build_initializer_map(graph)

    input_names = [i.name for i in graph.input if not is_initializer(i.name, initializer_map)]

    layers = []
    tensor_id = 0

    for node in graph.node:
        # OP
        op = node.op_type
        op_name = node.name if node.name else f"{op}_{tensor_id}"
        tensor_id += 1

        layer = {
            "op_type": f"qnn.csi.{op.lower()}",
            "name": op_name
        }

        # Attributes
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode("utf-8")
        if attrs:
            layer["attrs"] = attrs

        # ✅ 对 conv 算子增强属性提取
        if op.lower() in ["conv", "conv2d"]:
            # 处理 kernel_shape → kernel_size
            if "kernel_shape" in attrs:
                attrs["kernel_size"] = attrs.pop("kernel_shape")


            # 提取 channels 信息（从第一个 weight 常量中维度[0]）
            if len(node.input) >= 2:
                weight_name = node.input[1]
                if weight_name in initializer_map:
                    weight_shape = initializer_map[weight_name]["dim"]
                    if len(weight_shape) >= 1:
                        attrs["channels"] = weight_shape[0]

            # 添加固定 layout 信息
            attrs["kernel_layout"] = "OIHW"
            attrs["data_layout"] = "NCHW"
            attrs["out_layout"] = ""



        # Inputs
        inputs = []
        for inp in node.input:
            if inp == "":
                continue
            is_const = 1 if is_initializer(inp, initializer_map) else 0
            inp_shape = tensor_map.get(inp, initializer_map.get(inp, {"dim": []})["dim"])
            layout = get_layout(inp_shape, is_const)
            inputs.append({
                "name": inp,
                "dim": inp_shape,
                "is_const": is_const,
                "layout": layout
            })
        layer["inputs"] = inputs

        # Outputs
        outputs = []
        for out in node.output:
            out_shape = tensor_map.get(out, [])
            layout = get_layout(out_shape, False)
            # “outputs” 肯定不是const，否则会被 do_constant_folding=True 执行常量折叠优化。所以这里 is_const=False。
            outputs.append({
                "name": out,
                "dim": out_shape,
                "is_const": 0,
                "layout": layout
            })
        layer["outputs"] = outputs

        layers.append(layer)

    return {
        "input_names": input_names,
        "layers": layers
    }

def main():
    model = onnx.load(INPUT_ONNX_PATH)

    # 🔧 加入这一行进行 shape 推理，补全中间层维度信息，使得“outputs”补全纬度信息
    model = shape_inference.infer_shapes(model)
    # 使用 onnx.shape_inference.infer_shapes(model) 来自动推断缺失的维度信息

    parsed_json = parse_onnx_model(model)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(parsed_json, f, indent=2)
    print(f"Converted ONNX to JSON and saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()