# NPUPERF 的自由定义hardware的使用模式

## 说明

对于硬件输入，若需要自定义硬件架构进行设计空间探索，则可以通过配置一个json格式的模板文件来定制化hardware输入

## 符号对照表

符号 | 含义
--- | ---
W | 权重
I | 输入特征
O | 输出特征
K | 输出通道数 & 卷积核个数
C | 输入通道数 & 卷积核通道数
OX | 输出特征宽度
OY | 输出特征高度
W/I/O | 表示操作数共享，三种操作数可以共享此buffer

## 模板

例如，要定义一个含有以下规格参数的hardware：

- MAC 阵列级别
  - 算力：2048 个 MACs
  - 空间映射的并行度：输出通道 8，输入通道 32，输出特征高度方向 4，输出特征宽度方向 2
- Local Buffer 级别
  - 存在一个用于存放权重的sram，大小为 128 KB
  - 存在一个用于存放输入的sram，大小为 64 KB
  - 存在一个用于存放输出的sram，大小为 128 KB
- global buffer 级别
  - 权重、输入特征、输出特征可以共享此global buffer
  - 大小为 1 MB
  - 带宽为 256 bit/cycle
- dram
  - 权重、输入特征、输出特征可以共享此dram
  - 大小不考虑（近乎无穷大，因为这个没有任何影响）
  - 带宽为 192 bit/cycle


```json
{
    "MAC_unroll": {
        "K": 8,
        "C": 32,
        "OX": 4,
        "OY": 2
    },
    "local_buffers": [
        {
            "op": "W",
            "size": 128
        },
        {
            "op": "I",
            "size": 64
        },
        {
            "op": "O",
            "size": 128
        }
    ],
    "global_buffer": {
        "op": "W/I/O",
        "size": 1,
        "bandwidth": 256
    },
    "dram": {
        "op": "W/I/O",
        "bandwidth": 192
    }
}
```

- **NOTE！**
  - MAC array 的展开维度至少是 2 个，各维度的乘积为MAC 总个数
  - local buffer size 的单位为 KB
  - gloabl buffer size 的单位为 MB
  - 若不指定dram 的size(例如size为None或没有size一条)，将认为其为几乎无穷大
    - 这是因为dram size一般不作为一个探索参数
    - 若指定，则单位为 GB，如 `size: 4`
  - 关于内存的操作数共享，一般而言对于高级别的内存会共享三种操作数，低级别的local buffer可以单独存放一种操作数，也可以作为IO共享的buffer
  - 可以有多级的local buffer，完全自定义，例如一级较小的 I buffer和O buffer，然后以及较高的I/O buffer，等等

