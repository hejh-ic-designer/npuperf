# NpuPerf 支持 Transformer 系列网络的 profiling 说明文档

1. hhb networks: [/npuperf/inputs/hhb_networks/](/npuperf/inputs/hhb_networks/)

2. Hardware：[/npuperf/inputs/HW/](/npuperf/inputs/HW/)

3. （运行结果的输出文件）json文件位于：[/outputs/](/outputs/)

4. （运行结果的输出文件）可视化Figure位于：[/result_plot/](/result_plot/)

5. 使用的顶层脚本(使用已有架构)：[/tools/main.py](/tools/main.py)

6. 使用的顶层脚本(使用自定义架构)：[/tools/main_dse.py](/tools/main_dse.py)
   1. 自定义架构需要在[/npuperf/inputs/hw_config/](/npuperf/inputs/hw_config/)下创建一个配置文件，文件名应为架构名称
   2. 关于接口的更多信息见：[接口](/tools/README_tools.md#输入接口)


运行顶层脚本，在控制台输入：
1. 使用已有架构
```bash
python tools/main.py --nn <network-name> --hw <hardware-name>
```
2. 自定义新的架构
```bash
python tools/main_dse.py --nn <network-name> --hw <hardware-config-file-name>
```

e.g.:
```bash
python tools/main.py --nn bert_small --hw Meta_prototype
```

---

**注 1**：参考文件 [environment.yml](/environment.yml) 安装对应版本要求的python(至少3.10)，以及相应的依赖包

**注 2**：运行脚本时如果出现报错 `ModuleNotFoundError: No module named 'npuperf'`，需要设置python包路径

- 使用`cd` 转到顶层目录
- linux 和 MacOs 下，使用
```bash
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

- Windows下，使用
```bash
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
```

**注 3**：在breakdown.png 和 total.png 两个图中，没有包含 Memory 操作的算子，如Concatenate，Transpose, Split

**注 4**：本框架存在 `时间映射搜索粒度` 和`框架运行时间`之间的trade-off，可以通过在顶层脚本中修改`loma_lpf_limit`的值去调整。**目前设置为6**，是为了程序运行的更快一些，但遇到大尺度算子时可能因为搜索粒度粗而得到次优的映射点；可以通过适当调大`loma_lpf_limit`的值 (例如7-10)去得到更好的结果(但是程序可能会运行缓慢，运行一个网络可能超过30min :joy:)

**注 5**：在运行时，HHB Json Networks会先解析成为 .py 格式的网络描述，如果你想查看 .py 格式的网络描述，可以使用脚本 [export_wl_from_json.py](/tools/export_wl_from_json.py) 来导出

1. 在 `export_wl_from_json.py` 中设置你要运行的架构

```python
# 这个例子是 Meta_prototype
from npuperf.inputs.HW.Meta_prototype import accelerator
```

2. 然后在变量 `nn_list` 中设置你想导出的网络
3. 运行此脚本

```bash
python tools/export_wl_from_json.py
```

**注 6**：查看 json 网络中算子类型以及数量的脚本：[/tools/show_op.py](/tools/show_op.py)

**注 7**：有了运行结果后（outputs文件夹下的json结果文件），就可以使用脚本 [save_tiny_csv.py](/tools/save_tiny_csv.py) 将数据导出到一个csv的文件中，注意在 `if __name__ == '__main__:` 下设置文件夹目录
