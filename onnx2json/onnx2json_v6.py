import onnx
import json
import numpy as np
from onnx import numpy_helper, shape_inference

class ONNX2JSONConverter:
    def __init__(self, onnx_path, json_path):
        """
        初始化转换器
        :param onnx_path: 输入的 ONNX 模型文件路径
        :param json_path: 输出 JSON 文件路径
        """
        self.onnx_path = onnx_path
        self.json_path = json_path

        # 定义 OP 类型映射表（onnx → json中标准op名）
        self.op_type_map = {
            "Relu": "qnn.csi.relu",
            "PRelu": "qnn.csi.prelu",
            "Conv": "qnn.csi.conv2d",
            "ConvTranspose": "qnn.csi.deconv2d",
            "AveragePool": "qnn.csi.avgpool2d",
            "GlobalAveragePool": "qnn.csi.global_avgpool2d",
            "GlobalMaxPool": "qnn.csi.global_maxpool2d",
            "MaxPool": "qnn.csi.maxpool2d",
            "Add": "qnn.csi.add",
            "Sub": "qnn.csi.subtract",
            "Mul": "qnn.csi.mul",
            "Gemm": "qnn.csi.dense",
            "MatMul": "qnn.csi.matmul",
            "Concat": "qnn.csi.concatenate",
            "Transpose": "qnn.csi.transpose",
            "Split": "qnn.csi.split",
            "LRN": "qnn.csi.lrn",
            "Softmax": "qnn.csi.softmax",
            "Flatten": "qnn.csi.reshape",
            "Clip": "qnn.csi.clip",
            "Sigmoid": "qnn.csi.sigmoid",
            "Mean": "qnn.csi.mean",
            "Tanh": "qnn.csi.tanh",
            "Pow": "qnn.csi.power",
            "Sqrt": "qnn.csi.sqrt",
            "Div": "qnn.csi.div",
            "Gather": "qnn.csi.take",
            "Erf": "qnn.csi.erf",
            "Slice": "qnn.csi.strided_slice",
            "ReduceVariance": "qnn.csi.variance",
            "Cast": "qnn.csi.cast",
            "Sin": "qnn.csi.sin",
            "Cos": "qnn.csi.cos",
            "Upsample": "qnn.csi.upsampling",
            "Exp": "qnn.csi.exp"
        }

    def get_tensor_shape(self, value_info):
        """
        提取张量的 shape 信息
        """
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.HasField("dim_value") else 1)
        return shape

    def build_tensor_map(self, graph):
        """
        构建张量名到 shape 的映射表（包括中间张量）
        """
        tensor_map = {}
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
            name = vi.name
            shape = self.get_tensor_shape(vi)
            tensor_map[name] = shape
        return tensor_map

    def build_initializer_map(self, graph):
        """
        收集模型中的所有初始化常量（如权重和偏置）
        """
        initializer_map = {}
        for init in graph.initializer:
            name = init.name
            tensor = numpy_helper.to_array(init)
            initializer_map[name] = {
                "dim": list(tensor.shape),
                "data": tensor.tolist()
            }
        return initializer_map

    def is_initializer(self, name, initializer_map):
        """
        判断某个输入是否是初始化常量（权重、偏置）
        """
        return name in initializer_map

    def get_layout(self, shape, is_const):
        """
        根据 shape 和是否是常量判断 layout 类型
        """
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

    def normalize_op_type(self, op):
        """
        将 ONNX 中的算子名称标准化为 json 所需的 qnn.csi.* 格式
        """
        if op not in self.op_type_map:
            raise ValueError(f"Unsupported op_type '{op}' encountered! Please update op_type_map.")
        return self.op_type_map[op]

    def parse_model(self, model):
        """
        解析 ONNX 模型并转换为目标 JSON 数据结构
        """
        graph = model.graph
        tensor_map = self.build_tensor_map(graph)
        initializer_map = self.build_initializer_map(graph)

        input_names = [i.name for i in graph.input if not self.is_initializer(i.name, initializer_map)]

        layers = []
        tensor_id = 0

        for node in graph.node:
            op = node.op_type
            op_type_norm = self.normalize_op_type(op)
            op_name = node.name if node.name else f"{op}_{tensor_id}"
            tensor_id += 1

            layer = {
                "op_type": op_type_norm,
                "name": op_name
            }

            # 解析 attributes 属性字典
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

            # 如果是 Conv 算子则进一步补充属性
            if op == "Conv":
                if "kernel_shape" in attrs:
                    attrs["kernel_size"] = attrs.pop("kernel_shape")
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    if weight_name in initializer_map:
                        weight_shape = initializer_map[weight_name]["dim"]
                        if len(weight_shape) >= 1:
                            attrs["channels"] = weight_shape[0]
                attrs["kernel_layout"] = "OIHW"
                attrs["data_layout"] = "NCHW"
                attrs["out_layout"] = ""

            layer["attrs"] = attrs

            # 输入信息（包含常量/权重/偏置）
            inputs = []
            for inp in node.input:
                if inp == "":
                    continue
                is_const = 1 if self.is_initializer(inp, initializer_map) else 0
                inp_shape = tensor_map.get(inp, initializer_map.get(inp, {"dim": []})["dim"])
                layout = self.get_layout(inp_shape, is_const)
                inputs.append({
                    "name": inp,
                    "dim": inp_shape,
                    "is_const": is_const,
                    "layout": layout
                })
            layer["inputs"] = inputs

            # 输出信息（默认不是常量）
            outputs = []
            for out in node.output:
                out_shape = tensor_map.get(out, [])
                layout = self.get_layout(out_shape, False)
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

    def convert(self):
        """
        执行模型加载 → shape 推理 → 转换 → 保存 JSON 文件
        """
        print(f"🔄 正在加载 ONNX 模型: {self.onnx_path}")
        model = onnx.load(self.onnx_path)

        # 利用 shape_inference 补全缺失维度
        model = shape_inference.infer_shapes(model)

        # 解析模型为 json 格式
        json_data = self.parse_model(model)

        # 写入输出 JSON 文件
        with open(self.json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"✅ 已保存 JSON 文件至: {self.json_path}")

# 入口函数
def main():
    converter = ONNX2JSONConverter(
        onnx_path="./onnx_models/resnet50_export_params.onnx",
        json_path="./json_models/resnet50_onnx2json_v6.json"
    )
    converter.convert()

if __name__ == "__main__":
    main()