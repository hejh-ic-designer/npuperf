
# 📘 NPU Perf `json2workload.py` 讲解 README

本文件详尽解析了 `npuperf` 项目中 `json2workload.py` 脚本的逻辑和用途。该脚本将 JSON 格式的神经网络描述转换为适配不同 DLA 架构的 Python 字典格式 workload 文件。

---

## 📥 输入

- JSON 网络结构文件（如 `resnet50.json`）
- DLA 映射规则 Python 模块（如 `inputs/Mapping/Meta_prototype.py`）

## 📤 输出

- Python 字典形式的 workload 文件（如 `workload_resnet50.py`）

---

## 🧱 类结构：`Json2WorkloadParser`

### 初始化 `__init__`

- 加载 JSON 网络描述
- 加载 DLA mapping 信息
- 设置是否融合激活函数（如 ReLU）

---

## 🔧 核心方法说明

### 1️⃣ `pick_mapping(mapping_path_or_dict)`

- 支持路径或直接传入 dict 的两种方式加载 DLA mapping

### 2️⃣ `pick_json_workload(json_workload_or_path)`

- 读取 JSON 文件，提取 `layers` 和 `input_names`

### 3️⃣ `set_dummy_operator(merge_activation_function)`

- 配置需要剔除的“无效”算子（如 softmax、reshape）
- 若不融合激活函数，则也将 `relu`, `prelu` 剔除

### 4️⃣ `run()`

主入口函数，依次执行：
- `parse_input_layer()`
- `get_io_name()`
- `parse_layer()`
- `update_operand_source_dict()`

---

### 5️⃣ `parse_input_layer()`

- 将输入 Tensor（如图像）解析为 `workload` 中的输入层

---

### 6️⃣ `get_io_name()`

- 构建计算图的结构：每层的输入名、输出名和算子类型
- 特殊处理 `deconv2d` 展开为多个 `Conv_form_deconv` 和一个 `Contract`

---

### 7️⃣ `parse_layer()`

- 根据不同 `op_type` 派发给对应解析器（Parser）：
  - Conv → `ConvParser`
  - Dense → `DenseParser`
  - Pooling → `PoolParser`
  - Add/Mul → `AddParser` / `MulParser`
  - DeConv → `DeConvParser`（多层展开）
- Dummy 层会被删除或融合进前一层（如 ReLU）

---

### 8️⃣ `update_operand_source_dict()`

- 替换所有输入输出之间的“名字连接”为真实的 ID 编号
- 处理 Add、Concat 等多输入操作

---

## 🔍 私有方法说明

- `__change_input(name)`：根据 name 找到来源层的 `real_id`
- `__get_source_id(id)`：反查连接到当前层的 ID 列表

---

## 📄 输出文件格式示例

```python
workload = {
    -1: {
        "equation": "input",
        "loop_dim_size": {"B": 1, "K": 3, "OY": 224, "OX": 224},
        ...
    },
    0: {
        "equation": "K,B,OY,OX += W*I",
        "operator_type": "Conv",
        "operand_source": {"I": [-1], "W": [...]},
        ...
    },
    ...
}
```

---

## 📊 可视化支持

```python
from visualization.graph.dnn import visualize_dnn_graph
```

- 使用 `DNNWorkload` 和 `visualize_dnn_graph()` 生成图结构可视化

---

## ⚙️ 支持的运行模式

### ✅ MODE 1：遍历所有 DLA

```python
for DLA in mapping_path_list:
    ...
```

### ✅ MODE 2：只运行一个 DLA（常用）

```python
DLA_name = 'Meta_prototype'
...
```

---

## ✅ 总结

| 模块                | 功能                           |
|---------------------|--------------------------------|
| `Json2WorkloadParser` | 核心解析器                     |
| `Mapping` 模块        | DLA 硬件约束 / 分配信息           |
| 各类 `xxxParser`     | 单个算子的 workload 生成器        |
| `visualize_dnn_graph` | 可视化网络结构图                 |

---

如需支持更多算子或自定义结构，可以在 parser 类中扩展。
