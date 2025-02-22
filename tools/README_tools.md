# Domian-specific DLA Perf Tool Documentation
这里是专用领域神经网络加速器的建模工具，用于硬件架构的设计空间探索和网络部署性能分析

建模工具在KU Lueven大学MICAS实验室的两个开源框架[Zigzag, DeFiNES](#framwork)的基础上，做了进一步的优化和改进，以支持更多网络结构、硬件架构以及神经网络算子

建模工具有三大概念：**Workload, Hardware和Mapping**，其中:
- Workload 描述了要执行的神经网络的网络信息，以DAG(Directed Acyclic Graph)的方式建模，交由后层评估
- Hardware描述了物理层面的硬件架构，包含计算阵列和相匹配的内存层次结构(Memory Hierarchy)，hardware 以内存为中心建模，考虑算子的各种循环维度在内存层级上的展开和排列，以分析整体的延迟、功耗、利用率等指标
- Mapping 描述了一个算子在硬件上的映射方式，分为空间映射(spatial mapping)和时间映射(temporal mapping)两部分，其中空间映射显示了算子在计算阵列上的空间并行度(循环展开)，时间映射显示了各个维度的时间映射顺序(循环交换)，在建模工具中和LOMA搜索引擎相关

## Installing Perf Tool

建模使用的环境为python，下面以Anaconda 为例创建所需要的python 环境

### Prerequisites

- git
- pip
- Anaconda (python >= 3.10)

### Installation

1. 安装 [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) 环境

2. 下载整个目录文件 (`git clone` this repo)

3. 使用terminal 或Anaconda Prompt 执行以下步骤:
   -  使用 `cd` 转到顶层目录
   -  从 `environment.yml` 文件创建python 环境
       ```
       conda env create -f environment.yml
       ```
   -  激活新环境:
       ```
       conda activate NPUPERFenv
       ```

4. 设置包路径:
   - 使用`cd` 转到顶层目录
   - 设置包路径，将当前工作目录加入到PYTHONPATH环境变量中
    ```bash
    export PYTHONPATH=${PWD}:${PYTHONPATH}
    ```
    如果在Windows下，应使用
    ```bash
    $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
    ```

## DEMO 1: Layer-by-Layer

在这个示例中，我们以逐层执行网络的方式进行性能评估，需要运行的顶层脚本为 [main_LBL.py](/tools/main_LBL.py)，其中定义了将要运行的workload路径和hardware路径

### First Run

- 我们选择的网络为 `resnet50`，定义网络的文件位于[workload_resnet50.py](/npuperf/inputs/WL_fromjson/Meta_prototype/workload_resnet50.py)

- 我们选择的DLA为 `Meta_prototype`，定义DLA的文件位于[Meta_prototype.py](/npuperf/inputs/HW/Meta_prototype.py)

- 在终端输入
```
python main_LBL.py
```
即可逐层评估网络，每层依次输出能量、延迟等结果信息

- runtime?

About 0.5 - 5 seconds per layer

- 运行其他网络和DLA

如需更改运行的网络和DLA，请在文件[main_LBL.py](/tools/main_LBL.py#L15)中修改网络和DLA的路径

### Analyzing Results

在路径[/outputs/Meta_prototype-resnet50/](/outputs/Meta_prototype-resnet50/)下会逐层输出json格式的评估结果，每一个文件都对应了网络的一层，输出的评估结果中包含了energy, latency, MAC utilization, memory utilization, temporal mapping等信息

若需要添加或减少显示信息，请在[cost_model.py](/npuperf/classes/cost_model/cost_model.py)中修改`__simplejsonrepr__`函数

## DEMO 2: Layer-Fusion

在这个示例中，我们以深度优先的方式执行网络进行性能评估，需要运行的顶层脚本为 [main_artifact.py](/npuperf/main_artifact.py)，其中定义了将要运行的workload路径和hardware路径，迭代的tile size和 深度优先模式(DF modes)等

### First Run

- 我们选择的网络为 `fsrcnn`，定义网络的文件位于[workload_fsrcnn.py](/npuperf/inputs/WL/Meta_prototype/workload_fsrcnn.py)

- 我们选择的DLA为 `Meta_prototype_DF`，定义DLA的文件位于[Meta_prototype_DF.py](/npuperf/inputs/HW/Meta_prototype_DF.py)

- 在终端输入
```
python main_artifact.py
```
即可以深度优先的方式评估网络，深度优先的tile size和overlap的三种cache策略(即DF modes)会分别按照顶层的给值依次迭代，总共会遍历108种不同的DF策略组合(6x6 X-dim and Y-dim tile size, 3 DF modes)，layer stacking依据权重数据大小在on-chip memory上的排布情况自动推断，评估完成后输出一个[图表](/result_plot/Fig12.pdf)以展示不同的DF策略下能量和延迟情况

- run time?

默认为18小时(使用单CPU线程，设置`loma_lpf_limit = 8`)，调整LOMA时间映射搜索引擎的外部参数`loma_lpf_limit`可以对【运行速度 - 质量】进行trade off，其设置位于顶层脚本[main_artifact.py](/tools/main_artifact.py#L78)，其值越大程序运行时间越长，搜索的粒度越细，结果可能越好，原论文为保证搜索到最佳结果将其设定为8。`loma_lpf_limit = 6` 时可减小运行时长到1小时以内

- 若需要DeFiNES [paper](#paper)中更多图表信息，在终端输入
```
python plot_artifact.py
```
这可以抓取上一步生成的pickle信息，并制作绘图，结果均存储于路径[/result_plot/](/result_plot/)


> 注：这是DeFiNES框架原本的使用逻辑，更多请参考DeFiNES的[paper](#paper)和[framwork](#framwork)

### Analyzing Results

在路径[/result_pickle_files/](/result_pickle_files/)下以pickle格式存储了所有tile size和DF mode组合情况下的评估结果，在路径[/result_plot/Fig12.pdf](/result_plot/Fig12.pdf)下展示了所有108种DF策略下的能量和功耗信息，从中可以清晰的抓取到最佳深度优先参数点


## User Guide

以下文档将提供perf tool更多的模块信息，总体上，可以将此框架分为五个大的部分：[workload](#workload), [hardware](#hardware-architecture), [mapping](#mapping), [stages](#stages), [outputs](#outputs)

另外，还有对HHB tool的[接口支持](#hhb-2-workloadpy-parser)，可以实现网络输入文件自动化

### Workload

workload通常意义上指用于深度学习的神经网络，一个网络由很多个layer构成，layer之间通过连接关系组合成一个整体，每一个layer都表示一个算子

workload位于路径[/npuperf/inputs/WL/](/npuperf/inputs/WL/)下，作为输入文件，可以手动定义，也可以通过对HHB tool的接口支持自动化从json格式的网络描述解析得到

workload描述中，不仅包含网络层本身的信息，而且必须包含映射信息(mapping)，映射信息和部署的硬件相关，所以针对每一个硬件，workload描述都是不同的。在对HHB tool的json格式的网络文件自动化解析时，除了输入json file外，也同时要输入要部署硬件的mapping信息，位于[/npuperf/inputs/Mapping/](/npuperf/inputs/Mapping/)

在建模中将workload构造为计算图DAG对象，具体见 [DNNWorkload](/npuperf/classes/workload/dnn_workload.py)

#### operator support

Perf tool 对不同的算子类型有不同的建模方式：
1. 考虑数据在memory之间搬移，和MAC array上计算的算子，被解析成 [LayerNode](/npuperf/classes/workload/layer_node.py) 对象
    - Conv
    - DeConv
    - Depthwise Conv
    - Mul
    - Add
    - Pool
    - Fullyconnect
2. 考虑数据在memory之间搬移的算子，被解析成 [MemNode](/npuperf/classes/workload/mem_node.py) 对象
    - Concat
    - Transpose
3. 考虑可以fuse到前一层的算子，一般为非线性激活函数，并不会单独作为一层去解析，而是在前一层算子中添加一个激活函数的标识，和前层一起在cost model评估
    - Relu
    - PRelu
    - LeakeyRelu
4. 被认为没有加速空间，不会去建模的算子
    - Reshape
    - LRN
    - Softmax
    - ...

#### Manual layer definition

每一个layer使用一个dict对象来定义，包含以下属性：

- **operator_type**: 算子类型
- **equation**: 当前layer的计算等式。维度使用小写字母，操作数(operand)使用大写字母，'O'应当始终作为输出的操作数，输入操作数可以随意命名
- **equation_relations**: 上述等式中前后维度的关系。在卷积算子中，卷积核步长stride和卷积核膨胀系数dilation在其中体现
- **loop_dim_size**: 不同维度的大小。上述equation_relations中等号左侧的维度并不会提供，并且自动推断。更多见[loop notation](#loop-notation)
- **operand_precision**: 操作数的精度(bit)，'O'应当始终作为输出的精度，'O_final'表示最终输出的精度
- **operand_source**: 当前层输入操作数的来源id，用于创建NN图
- **constant_operands**: 当前层的常量操作数，不依赖于前层的计算
- **operand_source_dimension_mapping**: 对于和前层有依赖关系的输入操作数，当前layer输入层维度和前层输出维度的对应关系 *(注：仅在Layer Fuison模式下需要)*
- **core_allocation**: 执行当前层的core
- **spatial_mapping**: 该层使用的空间并行化策略，和对应的硬件MAC array相关
- **memory_operand_links**: 虚拟内存操作数(I1, I2, O)与实际算法操作数之间的对应关系。在卷积算子中，I1对应I，I2对应W，O对应O

#### Loop notation

以下是循环维度的记法，用于描述layer中的各个维度，来自论文[zigzag](https://ieeexplore.ieee.org/document/9360462)

- **B**: batch size
- **K**: output channels
- **C**: input channels
- **OY**: output rows
- **OX**: output columns
- **FY**: kernel rows
- **FX**: kernel columns

### Hardware Architecture

这里介绍硬件是如何建模的，硬件的描述文件位于[/npuperf/inputs/HW/](/npuperf/inputs/HW/)，一个硬件DLA建模需要的模块对象有以下这些

#### Operational Unit

要加速网络推理，需要使用已经训练好的权重数据对中间特征数据进行乘累加运算。运算单元通常是乘加器，执行两个数据元素的乘加运算

Operational Unit对象包含的参数有：

- **input_precision**: List of input operand (data) precision in number of bits for each input operand (typically 2 for Multiplier).
- **output_precision**: The bit precision of the operation's output.
- **energy_cost**: Energy of executing a single multiplication.
- **area**: The HW area overhead of a single multiplier.

#### Operatioanl Array

推理一个NN需要上百万次乘加运算，加速器通常拥有一个可执行这些运算的运算单元阵列，通常称为PE Array，这样可以大大加快计算速度，并提高能效

PE Array拥有多个维度，每个维度都有一个大小

Operatioanl Array对象包含的参数有：

- **operational_unit**: The operational unit from which the array is built.
- **dimensions**: The dimensions of the array. This should be defined as a dict, with the keys being the identifier of each dimension of the array (typically ‘D1’, ‘D2, …) and the values being the size of this dimension (i.e. the size of the array along that dimension).

#### Memory Instance

为了存储运算阵列中用于计算的不同激活和权重，需要以分层方式连接不同的Memory Instance。这些实例规定了每个存储器的容量、写入和读取这些memory的成本、带宽以及读写端口数量

Memory Instance对象包含的参数有：

- **name**: A name for the instance
- **size**: The memory size in bits.
- **r_bw/w_bw**: A read and write bandwidth in number of bits per cycle.
- **r_cost/w_cost**: A read and write energy cost.
- **area**: Area overhead of the instance.
- **r_port/w_port/rw_port**: The number of read/write/read-write ports the instance has available.
- **latency**: The latency of an access in number of cycles.

#### Memory Hierarchy

除了知道每个内存实例的规格外，内存层次结构还编码了内存与操作阵列和其他内存实例之间的互连信息。这种相互连接是通过多次调用 `add_memory()`来实现的，其中第一次调用添加了第一级内存，它连接到操作阵列，后面的调用连接到较低级别的内存。这样就建立了一个存储器层次结构

要知道内存是应该连接到运算阵列还是另一个更低级的内存，就需要知道内存中将存储哪些数据。为了将算法与硬件分离开来，采用了 "memory operands"的概念（与 "algorithmic operands"相对，后者通常是 I/O 激活和权重 W）。可以将内存操作数视为虚拟操作数，随后将通过 `memory_operand_links` 属性与映射文件中的实际算法操作数相对应

与operational unit 可以unroll（形成一个operational array）的方式类似，内存也可以unroll ，其中每个内存可以伴随单个operational unit，也可以伴随operational array 中一个或多个维度的所有operational unit。这可以通过 `served_dimensions` 属性进行编码，该属性指定该内存级别的单个内存实例是否为该维度的所有操作单元提供服务。其应该是一组 one-hot-encoded tuples

最后，内存实例的不同read/write/read-write 端口分配给层次结构中可能出现的不同数据移动。层次结构中有四种类型的数据移动：from high (fh), to high (th), from low (fl), to low (tl)

在撰写本文时，可通过以下语法将这些数据手动链接到read/write/read-write 端口之一：`{port_type}_port_{port_number}`，port_type 为 r、w 或 rw，port_number 等于端口号，从 1 开始，可分配多个相同类型的端口。另外，如果没有向 `add_memory()` 调用提供参数，这些参数将作为默认值自动生成

在内部，`MemoryHierarchy` 对象扩展了 `NetworkX DiGraph` 对象，因此可以使用其方法

Memory Hierarchy对象包含的参数有：

- **operational_array**: The operational array to which this memory hierarchy will connect. This is required to correctly infer the interconnection through the operational array’s dimensions. Through the add_memory() calls it adds a new MemoryLevel to the graph. This requires for each call a:
- **memory_instance**: A MemoryInstance object you are adding to the hierarchy.
- **operands**: The virtual memory operands this MemoryLevel stores.
- **port_alloc**: The directionality of the memory instance’s different ports, as described above.
- **served_dimensions**: The different dimensions that this memory level will serve, as described above.

#### Core

operational array 和 memory hierarchy 共同构成了加速器的一个核心(Core)

Core对象包含的参数有：

- **id**: The id of this core.
- **operational_array**: The operational array of this core.
- **memory_hierarchy**: The memory hierarchy of this core.

#### HW Accelerator Model

多个core 被组合到硬件加速器中，这是模拟硬件行为的顶层对象

HW Accelerator Model对象包含的参数有：

- **name**: A user-defined name for this accelerator.
- **core_set**: The set of cores comprised within the accelerator.
- **global_buffer**: A memory instance shared across cores. This is currently un-used.

#### Modelled examples

按照论文DeFiNES中的硬件设置，一共建模了5个DNN加速器，分别为Meta prototype, TPU, Edge TPU, Ascend, Tesla NPU。依据深度优先策略又优化了5个相应的DF 变体架构，以更好的支持深度优先调度

![Specific settings.png](https://user-images.githubusercontent.com/55059827/183848886-c85b9950-5e49-47c9-8a47-ad05062debc3.png)

为了进行公平合理的比较，DeFiNES 将所有架构规范化为 1024 个 MAC 和最大 2MB 全局片上缓冲区（Global Buffer），但保留其空间展开和本地缓冲区设置，如上图Idx 1/3/5/7/9 所示。此外，还为每个规范化架构构建了一个变体（通过改变其片上存储器层次结构），在名称后面用 "DF "表示，如上图Idx 2/4/6/8/10 所示

### Mapping

mapping定义了网络算子在硬件资源上映射的方式。分为两部分：spatial mapping 和temporal mapping，前者由用户自定义在网络描述中，后者由LOMA搜索引擎自动得到最优解

当同一个网络部署在不同的硬件资源上时，应当在网络描述中修改相应的mapping 信息，和mapping相关的词条为：`core_allocation`, `spatial_mapping`, `memory_operand_links`

#### Stationary data flow

对于卷积映射而言，有多种数据流模式。在硬件加速中，需要将IFM数据和权重数据经过相应的buffer传到PE Array上进行卷积计算

这时，依据不同buffer的大小配置、卷积计算数据量的大小区别，采用不同的数据流映射方式才可能在最大程度上优化卷积的硬件加速

##### Principles

在perf tool中区别不同的stationary数据流需要使用到操作数和循环维度的相关性原则，详情见论文zigzag

- **output stationary**

在output stationary中，我们希望卷积输出的部分和数据在未完成全部累加时，充分的复用部分和数据，以节省部分和数据在buffer间的传递开销和用于暂存部分和的buffer容量

在perf tool中，为了实现output stationary，就要在时间映射的同时体现出对输出特征数据充分的reuse，和这部分相关的是LOMA 时间映射搜索引擎

在temporal mapping中，根据W，I，O三种操作数和七个维度的相关性原则中，需要将和'O' 相关的维度排列在mapping 的外侧，而将无关的维度排列在内侧，所以相当于对LOMA engine添加一定的约束，缩小搜索空间

- **weight stationary**

和上一个类似，在weight stationary中，我们希望权重数据可以被充分的reuse，以减小权重数据在buffer间的搬移和权重buffer的容量开销

在perf tool中，为了实现weight stationary，就要在时间映射的同时体现出对权重数据充分的reuse，和这部分相关的是LOMA 时间映射搜索引擎

在temporal mapping中，根据W，I，O三种操作数和七个维度的相关性原则中，需要将和'W' 相关的维度排列在mapping 的外侧，而将无关的维度排列在内侧，所以相当于对LOMA engine添加一定的约束，缩小搜索空间

- **input stationary**

input stationary的情况稍微复杂一点，因为IX 和IY 的维度是间接的被OX，OY和FX，FY所决定的

在间接决定的等式中，如果索引（OX + FX）不变时，会出现对IX 方向上IFM 的reuse，这种IFM 复用的情况可以被分为三种(详见论文zigzag),分别为空间上的reuse、时空上的reuse 和时间上的reuse

所以input stationary往往和空间映射是相关的（而非像output stationary和weight stationary那样是解耦合的），设置input stationary时，请务必注意现在所采用的空间映射方式

##### run stationary

在顶层脚本中，需要将 `LomaStage` 更改为 `StationaryLomaStage`，并在顶层设置参数 `stationary` 为 `O`, `W`, 或`I`

对于input stationary，如果是空间上对IFM reuse 的情况，则空间映射中应当同时对 FY 和 OY 进行展开，或同时对 FX 和 OX 进行展开

对于时空和时间上对IFM reuse 的情况，可以需要设置文件 `engine_stationary` 中的参数 [temporal_dim_for_I_relevant](/npuperf/classes/opt/temporal/loma/engine_stationary.py#L73)，以符合实际的input stationary 数据流行为

### Outputs

#### Layer-by-Layer outputs

本节解释 ZigZag 的逐层执行生成的输出文件

目前有三个预定义的`SaveStage`，位于[SaveStage.py](/npuperf/classes/stages/SaveStage.py)，它们以不同的方式将结果保存到 `.json` 文件或 `pickle` 文件中

将`CostModelEvaluation`对象保存为 `.json` 文件需要了解对象的相关/不相关属性。这由`SaveStage`内的 `complexHandler` 处理

`complexHandler` 负责处理应针对传递给它的每个对象调用的 json 表示法。例如，在 `SimpleSaveStage` 中，`CostModelEvaluation` 对象的 `__simplejsonrepr__` 方法指定了如何将其转换为 json 格式

#### Layer-Fuison outputs

本节解释 DeFiNES 的深度优先调度生成的输出文件

目前有两个预定义的`DumpStage`，位于[DumpStage.py](/npuperf/classes/stages/DumpStage.py)，它们以不同的方式将结果保存到 `pickle` 文件中

当`DumpStage`在完成迭代后将所有结果的列表转储到pickle文件时，`StreamingDumpStage`会在迭代前打开该文件，并将结果逐一写入该文件。这意味着，如果程序被终止，将写入不完整的结果，并且由于没有缓存结果列表以供以后写入，因此在系统内存上更容易。此外，如果文件名以 `.gz` 结尾，这个也会写入压缩文件

### HHB 2 Workload.py Parser

本工具支持从HHB tool导出的json 格式的标准网络描述文件，自动解析得到perf tool所需要的.py格式的网络描述文件，使用的脚本为 [json2workload.py](/json2workload.py)

#### run

- 需要从json文件解析得到workload，请首先在[json2workload.py](/json2workload.py)的`if __name__ == '__main__'`下面设定要解析的网络文件名称 (由变量`NN` 定义)，并确定即将被解析的json文件路径正确(变量`json_workload_path`)

- 然后，在终端输入
```
python json2workload.py
```
被解析的json文件位于[/npuperf/inputs/WL_json/](/npuperf/inputs/WL_json/)，解析输出的workload.py文件位于[/npuperf/inputs/WL_fromjson](/npuperf/inputs/WL_fromjson)

在解析完成后，可以根据需要显示解析结果的网络可视化，可供检查网络结构是否正确

#### process explain

解析的逻辑为 **network.json + hardware_mapping.py &rArr; workload.py**
- network.json 文件位于[/npuperf/inputs/WL_json/](/npuperf/inputs/WL_json/)
- hardware_mapping.py 文件位于[/npuperf/inputs/Mapping/](/npuperf/inputs/Mapping/)
- workload.py 文件位于[/npuperf/inputs/WL_fromjson/](/npuperf/inputs/WL_fromjson/)

生成的workload file不仅包含网络每层的operator type, dimension size, precision等信息，还包括和映射相关的mapping信息，每一个硬件都有一个相对应的[mapping](/npuperf/inputs/Mapping/)文件，这包含了要映射的硬件PE Array的空间展开度，硬件memory与操作数之间的字母映射等信息

> 注：每一个被解析的算子都应该在mapping文件中有相应的定义，否则在解析时将不知道为当前层添加怎样的mapping信息

在运行时按照算子级别逐层进行读取和解析，Perf tool 对不同的算子类型有不同的建模方式：

1. 考虑数据在memory之间搬移，和MAC array上计算的算子
    - Conv
    - DeConv
    > 注：对于Deconv 算子，这里参考了[NVDLA对Deconv算子的硬件加速](http://nvdla.org/hw/v1/ias/unit_description.html#deconvolution)，将一个deconv层分解为并行的(x_stride * y_stride)个conv层和一个contract 层，这一步即在Parser 中实现，所以传入perf tool 的 `workload.py` 已经不含deconv层，而是被拆解之后的conv层和contract层
    - Depthwise Conv
    - Mul
    - Add
    - Pool
    - Fullyconnect
2. 考虑数据在memory之间搬移的算子
    - Concat
    - Transpose
    - contract
3. 考虑可以fuse到前一层的算子，一般为非线性激活函数
    - Relu
    - PRelu
    - LeakeyRelu
4. 被认为没有加速空间，在解析时考虑删除的算子
    - Reshape
    - LRN
    - Softmax
    - ...

对上述第一类和第二类算子，抓取到网络层的信息后会传到对应算子的Parser进行解析，以输出对应的workload描述

对上述第三类算子，可以直接在上一层的描述中添加一个`activation_function`词条，以达到和前层融合计算的效果，一般并不会把此类算子当作单独的一层来建模

对上述第四类算子，则直接在解析时删去，并保持网络层前后的连接关系，因为此类算子被认为没有硬件加速的空间，硬件的执行逻辑一般是固定的，所以在建模中暂不考虑


## 输入接口
### Mode1：modeling for existing Architecture
针对给定的硬件模板，运行指定网络，并可选是否简单配置顶层的参数
在npuperf的 [./tools/main.py](/tools/main.py)中，定义了控制台输入的组织形式：
```python
parser = argparse.ArgumentParser(description="Setup npuperf inputs")
parser.add_argument('--nn', metavar='Network name', required=True, help='module name to hhb networks, e.g. fsrcnn2x')
parser.add_argument('--hw', metavar='Hardware name', required=True, help='module name to the accelerator, e.g. example_wioGB')
parser.add_argument('--flow', 
                    metavar='Temporal data flow',
                    choices=['O', 'W'],
                    required=False,
                    help='three types of Temporal mapping search method, default to Free completely LOMA engine, set "O" to Fully output stationary, set "W" to Fully Weight stationary')
parser.add_argument('--gb_size',
                    metavar='hardware config info: the global buffer size (MB)',
                    required=False,
                    help='Optional: change Global Buffer size based on selected hardware, e.g. 3')
parser.add_argument('--gb_bw',
                    metavar='hardware config info: the global buffer bandwidth (bit/cycle)',
                    required=False,
                    help='Optional: change Global Buffer bandwidth based on selected hardware, e.g. 256')
parser.add_argument('--dram_bw',
                    metavar='hardware config info: the dram bandwidth (bit/cycle)',
                    required=False,
                    help='Optional: change dram bandwidth based on selected hardware, e.g. 256')

```

1. `--nn`：neural network，指定要运行的网络名称，网络是来自HHB 工具的json格式的网络描述
2. `--hw`：hardware，指定要运行的硬件名称
3. `--flow`：（optional）data flow，指定要以何种数据流映射运行
   1. 若不指定，采用搜索空间最大的 LOMA 搜索引擎，可能搜索到最佳时间映射点
   2. 指定为“O”：采用 fully output stationary
   3. 指定为“W”：采用 fully weight stationary
4. `--gb_size`：（optional）global buffer size，单位为 MB
5. `--gb_bw`：（optional）global buffer bandwidth，单位为 bit/cycle
6. `--dram_bw`：（optional）dram bandwidth，单位为 bit/cycle

### Mode2：Design Space Exploration for customize Architecture

对于硬件输入，若需要自定义硬件架构进行设计空间探索，则可以通过配置一个json格式的模板文件来定制化
在npuperf的 [./tools/main_dse.py ](/tools/main_dse.py)中，定义了控制台输入的组织形式：

```python
parser = argparse.ArgumentParser(description="Setup npuperf inputs")
parser.add_argument('--nn', metavar='Network name', required=True, help='module name to hhb networks, e.g. fsrcnn2x')
parser.add_argument('--hw', metavar='Hardware config file name', required=True, help='file name to the user-defined json accelerator config file, e.g. example_wioGB')
parser.add_argument('--flow', 
                    metavar='Temporal data flow', 
                    choices=['O', 'W'],
                    required=False, 
                    help='three types of Temporal mapping search method, default to Free completely LOMA engine, set "O" to Fully output stationary, set "W" to Fully Weight stationary')
```

这时，hardware输入给定的 `--hw`以json格式呈现，里面定义了定制化hardware的详细的信息，json文件的路径应当为 `npuperf\inputs\hw_config\*.json`

例如，**要定义一个含有以下规格参数的hardware**：

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

**对应的 json 模板如下：**
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
其中，符号对照关系为：

| **符号** | **含义** |
| --- | --- |
| W | 权重 |
| I | 输入特征 |
| O | 输出特征 |
| K | 输出通道数 & 卷积核个数 |
| C | 输入通道数 & 卷积核通道数 |
| OX | 输出特征宽度 |
| OY | 输出特征高度 |
| W/I/O | 表示操作数共享，三种操作数可以共享此buffer |

- **NOTE！** 
   - MAC array 的展开维度至少是 2 个，各维度的乘积为MAC 总个数
   - local buffer size 的单位为 KB
   - gloabl buffer size 的单位为 MB
   - 若不指定dram 的size(例如size为None或没有size一条)，将认为其为几乎无穷大 
      - 这是因为dram size一般不作为一个探索参数
      - 若指定，则单位为 GB，如 `size: 4`
   - 关于内存的操作数共享，一般而言对于高级别的内存会共享三种操作数，低级别的local buffer可以单独存放一种操作数，也可以作为IO共享的buffer
   - 可以有多级的local buffer，完全自定义，例如一级较小的 I buffer和O buffer，然后以及较高的I/O buffer，等等

## 输出接口
### csv格式的性能汇总
运行 tools/save_csv.py 可以从已经生成的一系列json 结果文件中提取并整理数据，在一张csv表格中按每一个layer进行统计，并汇总本次 network-hardware experiment 的所有性能信息，包含的统计信息主要包含：

- layer info
   - layer index
   - layer name from HHB network
   - layer loop size：当前层要计算的循环维度大小
- latency
   - actual latency：最终的latency
   - ideal latency：理想的latency，MAC 利用率 100%
   - spatial stall：由于layer size和MAC Array空间上Mismatch导致的空间利用率下降，而产生的额外的latency
   - temporal stall：由于memory之间数据流冲突导致的时间维度上的额外的延迟
   - onloading latency：输入数据从DRAM加载到开始卷积流水线计算的数据加载时间
   - offloading latency：输出数据从MAC Array计算完毕到全部输出至DRAM的数据搬移时间
- MAC utilization
   - spatial utilization：考虑空间维度Mismatch的MAC利用率
   - temporal utilization：在上一个基础上，考虑时间维度Mismatch的MAC利用率
   - last utilization：在上一个基础上，考虑纯的数据加载和搬移后的最终的MAC利用率
- energy
   - total energy：执行本层的总功耗
   - MAC energy percentage：MAC Array计算需要的功耗与本层功耗占比
   - Mem energy percentage：数据在Memory之间的数据搬移需要的功耗与本层功耗占比
- percentage of whole Network
   - latency percentage：当前层占整个网络的latency百分比
   - energy percentage：当前层占整个网络的energy百分比
- memory utilization
   - register file utilization：寄存器级别的 buffer 利用率
   - local buffer utilization：local buffer 利用率
   - global buffer utilization：global buffer 利用率
   - dram utilization：DRAM 利用率
- Total layers energy & latency：整个网络的延迟&能量等详细信息的总和
- Total Network FPS：通过总延迟换算的帧率信息
### Json 格式的每层详细性能评估信息
#### layer info

- layer name：来自原始HHB网络层的网络名称，作为网络层定位信息
- layer id：序号
- layer loop size：当前层要计算的循环维度大小
#### energy
:::info
total_energy = mem_energy (can be breakdown) + MAC_energy
:::

- total energy：执行本层的总功耗
- mem energy：数据在Memory之间的数据搬移需要的功耗
- MAC energy：MAC Array计算需要的功耗
- energy breakdown：按三种操作数(WIO)分开的每一级的数据搬移能耗
#### MAC utilization

- MAC spatial utilization：考虑空间维度Mismatch的MAC利用率
- MAC utilization 0：在上一个基础上，考虑时间维度Mismatch的MAC利用率
- MAC utilization 1：在上一个基础上，考虑纯的数据加载后的MAC利用率
- MAC utilization 2：在上一个基础上，再考虑数据搬移后的最终的MAC利用率
#### latency
:::info
total latency(mC) = MAC latency + onloading latecny + offloading latency
MAC latency = ideal computation cycle + spatial stall + temporal stall
:::

- total latency：最终的latency
- MAC latency：MAC Array整个的计算时间
- data onloading latency：输入数据从DRAM加载到开始卷积流水线计算的数据加载时间
- data offloading latency：输出数据从MAC Array计算完毕到全部输出至DRAM的数据搬移时间
- ideal computation latency：理想的MAC 利用率 100%下的latency
- spatial stall：由于layer size和MAC Array空间上Mismatch导致的空间利用率下降，而产生的额外的latency
- temporal stall：由于memory之间数据流冲突导致的时间维度上的额外的延迟
- SS_comb_collect：每一级memory level的每一个端口的时间维度的stall/slack的统计（正为stall，负为slack）
#### memory

- mem_data_move_instance：每一个memory instance在执行本层的所有读写数据量的总和
- mem_utili_instance：每一个memory instance在执行本层的memory 利用率


## Reference

paper 是相关论文，framwork 是参考的开源框架

### Paper

- **General idea of ZigZag**
    - [L. Mei, P. Houshmand, V. Jain, S. Giraldo and M. Verhelst, **"ZigZag: Enlarging Joint Architecture-Mapping Design Space Exploration for DNN Accelerators"**, in IEEE Transactions on Computers, vol. 70, no. 8, pp. 1160-1174, 1 Aug. 2021, doi: 10.1109/TC.2021.3059962.](https://ieeexplore.ieee.org/document/9360462)

- **Latency model**
    - [L. Mei, H. Liu, T. Wu, H. E. Sumbul, M. Verhelst and E. Beigne, **"A Uniform Latency Model for DNN Accelerators with Diverse Architectures and Dataflows,"** 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE), Antwerp, Belgium, 2022, pp. 220-225, doi: 10.23919/DATE54114.2022.9774728.](https://lirias.kuleuven.be/retrieve/661303)

- **Temporal mapping search engine**
    - [A. Symons, L. Mei and M. Verhelst, **"LOMA: Fast Auto-Scheduling on DNN Accelerators through Loop-Order-based Memory Allocation,"** 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS), Washington DC, DC, USA, 2021, pp. 1-4, doi: 10.1109/AICAS51828.2021.9458493.](https://ieeexplore.ieee.org/document/9458493)

- **DeFiNES: Extend zigzag to support Layer-Fusion**
    - [L. Mei, K. Goetschalckx, A. Symons and M. Verhelst, **" DeFiNES: Enabling Fast Exploration of the Depth-first Scheduling Space for DNN Accelerators through Analytical Modeling,"** 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2023](https://arxiv.org/abs/2212.05344)

- **Apply ZigZag for different design space exploration case studies**

    - [P. Houshmand, S. Cosemans, L. Mei, I. Papistas, D. Bhattacharjee, P. Debacker, A. Mallik, D. Verkest, M. Verhelst, **"Opportunities and Limitations of Emerging Analog in-Memory Compute DNN Architectures,"** 2020 IEEE International Electron Devices Meeting (IEDM), San Francisco, CA, USA, 2020, pp. 29.1.1-29.1.4, doi: 10.1109/IEDM13553.2020.9372006](https://ieeexplore.ieee.org/abstract/document/9372006)

    - [V. Jain, L. Mei and M. Verhelst, **"Analyzing the Energy-Latency-Area-Accuracy Trade-off Across Contemporary Neural Networks,"** 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS), Washington DC, DC, USA, 2021, pp. 1-4, doi: 10.1109/AICAS51828.2021.9458553](https://ieeexplore.ieee.org/abstract/document/9458553)

    - [S. Colleman, T. Verelst, L. Mei, T. Tuytelaars and M. Verhelst, **"Processor Architecture Optimization for Spatially Dynamic Neural Networks,"** 2021 IFIP/IEEE 29th International Conference on Very Large Scale Integration (VLSI-SoC), Singapore, Singapore, 2021, pp. 1-6, doi: 10.1109/VLSI-SoC53125.2021.9607013](https://ieeexplore.ieee.org/abstract/document/9607013)

    - [S. Colleman, P. Zhu, W. Sun and M. Verhelst, **"Optimizing Accelerator Configurability for Mobile Transformer Networks,"** 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS), Incheon, Korea, Republic of, 2022, pp. 142-145, doi: 10.1109/AICAS54282.2022.9869945](https://ieeexplore.ieee.org/document/9869945)


### Framwork

- [DeFiNES](https://github.com/KULeuven-MICAS/DeFiNES): MICAS (KU Leuven)实验室开源的支持层融合的深度学习加速器建模和设计空间探索框架
- [zigzag](https://github.com/KULeuven-MICAS/zigzag): MICAS (KU Leuven)实验室开源的用于深度学习加速器的硬件架构映射设计空间探索框架
