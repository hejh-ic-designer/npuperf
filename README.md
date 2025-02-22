# Domian-specific DLA Perf Tool Documentation
这里是专用领域神经网络加速器的建模工具，用于硬件架构的设计空间探索和网络部署性能分析

建模工具在KU Lueven大学MICAS实验室的两个开源框架[Zigzag, DeFiNES](#Reference)的基础上，做了一些优化和改进，以支持更多网络结构、硬件架构以及神经网络算子

## Contribution
1. [json row NN file](/npuperf/inputs/hhb_networks/), 从原始ONNX网络导出的json网络文件, 供解析使用
2. [json parser](/npuperf/classes/workload/json_parser/), 支持上述json网络的解析
3. [hw gen](/npuperf/classes/opt/hw_gen/), 支持使用简单配置参数生成一个HW对象
4. [easy hw config](/npuperf/inputs/hw_config/), 利用hw gen,可以支持输入此类型的简单硬件描述
5. [other tool](/tools/), 一些脚本, 灵活运用可减小工作量

## Installation

1. 安装 [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) 环境

2. clone整个目录文件 (`git clone` this repo)

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

4. 设置包路径: (出现类似“ModuleNotFoundError: No module named 'npuperf'”的错误时, 请使用)
   - 使用`cd` 转到顶层目录
   - 设置包路径，将当前工作目录加入到PYTHONPATH环境变量中
    ```bash
    export PYTHONPATH=${PWD}:${PYTHONPATH}
    ```
    如果在Windows下，应使用
    ```bash
    $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
    ```

## DEMO

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

若需要添加或减少显示信息，请在[cost_model.py](/npuperf/classes/cost_model/cost_model.py#L1064)中修改`__simplejsonrepr__`函数


## Reference

- [DeFiNES](https://github.com/KULeuven-MICAS/DeFiNES): MICAS (KU Leuven)实验室开源的支持层融合的深度学习加速器建模和设计空间探索框架
- [zigzag](https://github.com/KULeuven-MICAS/zigzag): MICAS (KU Leuven)实验室开源的用于深度学习加速器的硬件架构映射设计空间探索框架
