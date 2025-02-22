# 自定义配置硬件架构的例子

## 算力和memory需要匹配

MAC数量越多, 算力越大时, 需要更大的LB buffer和更大的带宽来匹配, 一个可能示例如下:(仅供参考)

| #Macs|   |    |   |   | W LB size | I LB size | O LB size | GB size | DDR size | GB bandwidth | DDR bandwidth |
| ---  | --| -- | --| --| -- | --- | --- | --- | --- | --- | --- |
|      | K | C  |OX |OY |  weight buffer | input buffer | acc buffer | global buffer | dram |  |  | 
| 256  | 4 | 16 | 2 | 2 |  16KB  | 16KB  | 32KB  | 512KB | 4GB | 128 bit/c | 96 bit/c  | 
| 512  | 8 | 16 | 2 | 2 |  32KB  | 32KB  | 64KB  | 512KB | 4GB | 128 bit/c | 96 bit/c  | 
| 1024 | 8 | 32 | 2 | 2 |  64KB  | 64KB  | 128KB | 1MB   | 4GB | 128 bit/c | 96 bit/c  | 
| 2048 | 8 | 32 | 4 | 2 |  128KB | 64KB  | 128KB | 1MB   | 4GB | 256 bit/c | 192 bit/c | 
| 4096 | 8 | 64 | 4 | 2 |  128KB | 64KB  | 256KB | 2MB   | 4GB | 256 bit/c | 192 bit/c | 

## 寄存器配置

以1024MACs为例，MAC Array 展开度为 (**K 8 | C 32 | OX 2 | OY 2**)

硬件定义文件中添加Reg 内存实例时，需要使用变量 `served_dimensions` 为MAC Array 配置 I-reg, W-reg, O-reg

- W-reg: Weight 在 K C 维度上unroll，所以served_dimensions={(0, 0, 1, 0), (0, 0, 0, 1)}，配置 K x C = 256 个W-reg

- O-reg: OFM 在OX OY K维度上unroll，所以served_dimensions={(0, 1, 0, 0)}，配置 K x OX x OY = 32 个O-reg

- I-reg: IFM 在 C 维度上unroll，所以served_dimensions={(1, 0, 0, 0)}，配置 C x OX x OY = 128 个I-reg

**注意：** 

- `served_dimensions` 是一个集合类型，若有多个维度，用多个tuple表示，每一个tuple均为 **one-hot vector**

- 对于Reg以上级别的buffer，应设置为 `served_dimensions = 'all'`

- 若Reg是完全unroll的，可以设为 `served_dimensions = set()` ，表示没有served dimension

## 带宽匹配

Reg 级别应和上一级buffer(一般为local buffer)的带宽匹配

首先计算reg级别总的带宽，例如 W-reg size = 8 (1B)，而根据权重的 `served_dimensions={(0, 0, 1, 0), (0, 0, 0, 1)}`，所以权重的reg级别整体 bandwidth = 8 x 256 = 2048 bit/cycle，所以权重上一级的 weight local buffer 的bandwidth 应与之匹配

对于O，单个O-reg的带宽应当匹配 partial sum precision