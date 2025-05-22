# NPU Perf JSON 文件格式README 

---

## Version: 2025.5.22 ---by czy

本文档以ResNet50为例，用于说明模型转换后的 JSON 文件结构，包括各字段的含义、层结构、融合策略和注意事项。

---

## 🧾 文件整体结构

该 JSON 文件为一个对象，包含两个主字段：

```json
{
  "input_names": [...],
  "layers": [...]
}
```
---
## 1. input_names 主字段

	•	类型：List[str]
	•	说明：模型图的输入名称列表。
	
示例：
```json
"input_names": ["data_input"]
```
---

## 2. layers 主字段
	•	类型：List[LayerObject]
	•	说明：表示模型中每一层的计算节点（算子），按照执行顺序排列。

每个 LayerObject 的结构如下：
```json
{
  "op_type": "qnn.csi.conv2d",
  "name": "conv2d_conv1_fuse_bias_add_relu_1",
  "attrs": { ... },
  "inputs": [ ... ],
  "outputs": [ ... ]
}
```

---

## 3. layers 主字段解析

🔹 op_type

	•	类型：str
	•	说明：该层算子的类型，已标准化并带有前缀 qnn.csi.。
	•	示例：
	•	qnn.csi.conv2d
	•	qnn.csi.relu
	•	qnn.csi.add
	•	qnn.csi.reshape



🔹 name

	•	类型：str
	•	说明：该层的唯一名称。常带有融合信息，如 _fuse_add, _fuse_relu，表示本层已融合其他算子。



🔹 attrs

	•	类型：dict
	•	说明：该层的属性信息，不同算子有不同字段。

#### Conv2D 常见属性字段：

| 属性名         | 类型         | 说明                         |
|----------------|--------------|------------------------------|
| `channels`     | int          | 输出通道数                   |
| `kernel_size`  | [int, int]   | 卷积核大小                   |
| `strides`      | [int, int]   | 步长                         |
| `padding`      | [int, int, int, int] | 上下左右 padding |
| `groups`       | int          | 分组卷积数                   |
| `dilation`     | [int, int]   | 膨胀系数                     |
| `data_layout`  | str          | 数据格式，一般为 `NCHW`      |
| `kernel_layout`| str          | 权重格式，如 `OIHW`          |

---

### 🔹 `inputs` / `outputs`

- 类型：`List[TensorObject]`
- 每个输入或输出是一个 Tensor 对象：

```json
{
  "name": "tensor_0",
  "dim": [1, 64, 112, 112],
  "is_const": 0,
  "layout": "NCHW"
}
```

#### 字段说明：

| 字段名      | 类型         | 说明                          |
|-------------|--------------|-------------------------------|
| `name`      | str          | Tensor 名称（连接上下层）     |
| `dim`       | List[int]    | Tensor 维度，形如 [N, C, H, W] |
| `is_const`  | int (0 or 1) | 是否为常量（如权重）           |
| `layout`    | str          | 数据布局，常见为 `NCHW`        |

---

## 4. 可能的融合逻辑说明（op fusion）

参考阿里玄铁: https://www.xrvm.cn/document?temp=graph-optimization&slug=hhb

在模型转换过程中进行融合优化以提升效率。常见的融合模式如下：

| 融合模式                | 合并到的算子 | 说明                            |
|-------------------------|--------------|---------------------------------|
| Conv + BiasAdd          | Conv2D       | Bias 合并进 conv 权重偏置       |
| Conv + Add / Mul        | Conv2D       | 元素级操作合入 Conv 输出后处理   |
| Conv + ReLU / Clip      | Conv2D       | 激活函数合入卷积后               |
| Pad + Conv              | Conv2D       | 通过修改 padding 实现融合       |
| Reshape + Dense (Gemm)  | Dense        | 将 reshape 合并到全连接层输入端 |
| LayerNorm 子图模式       | LayerNorm    | 多个算子识别合并成 LayerNorm    |

合并信息通常体现在 `name` 字段中，如：

```json
"name": "conv2d_block1_fuse_bias_add_relu"
```


## 📌 示例层（Conv + ReLU 融合）

```json
{
  "op_type": "qnn.csi.conv2d",
  "name": "conv2d_conv1_fuse_bias_add_relu_1",
  "attrs": {
    "channels": 64,
    "kernel_size": [7, 7],
    "strides": [2, 2],
    "padding": [3, 3, 3, 3],
    "groups": 1,
    "dilation": [1, 1],
    "data_layout": "NCHW",
    "kernel_layout": "OIHW"
  },
  "inputs": [...],
  "outputs": [...]
}
```

---

## 📎 注意事项

1. 所有数据维度使用 `NCHW` 格式表示。
2. `is_const = 1` 表示输入是常量（权重、偏置、均值等）。
3. 不同类型的算子可能缺失 `attrs` 字段（如 ReLU、Add）。
4. `output` 的 `name` 将作为下游层 `input` 的 `name`。

---

## ✅ 推荐工具链支持

- ONNX → JSON 转换脚本（自定义）
- 模型优化支持（图融合、常量折叠）
- 后端硬件编译器（依据 JSON 层生成代码）

---

如需进一步解析某层结构或添加自定义算子支持，可在此文档基础上扩展。


