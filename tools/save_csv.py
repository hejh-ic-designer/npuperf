import ast
import csv
import json
import os
import re


class csv_HEAD:

    def __init__(self):
        ''' 初始化最初的表头 '''
        # 设置表头, 两个表头必须一一对应起来
        csv_dict_head = {}
        ## 表头 0
        csv_dict_head["Layer"] = [
            "layer info",  # 1
            "",  # 2
            "latency",  # 3
            "",  # 4
            "",  # 5
            "",  # 6
            "",  # 7
            "",  # 8
            "MAC utilization",  # 9
            "",  # 10
            "",  # 11
            "energy",  # 12
            "",  # 13
            "",  # 14
            "percentage of whole Network",  # 15
            "",  # 16
        ]
        ## 表头 1
        csv_dict_head["Node"] = [
            # layer info
            "index",  # 1
            "name",  # 2
            # latency
            "actual latency(mC)",  # 3
            "ideal latency(mC)",  # 4
            "spatial stall(mC)",  # 5
            "temporal stall(mC)",  # 6
            "onloading latency(mC)",  # 7
            "offloading latency(mC)",  # 8
            # MAC utilization
            "spatial utilization(%)",  # 9
            "temporal utilization(%)",  # 10
            "last utilization(%)",  # 11
            # energy
            "total energy(mJ)",  # 12
            "MAC energy percentage(%)",  # 13
            "Mem energy percentage(%)",  # 14
            # percentage of whole Network
            "latency percentage(%)",  # 15
            "energy percentage(%)",  # 16
        ]
        assert len(csv_dict_head["Node"]) == len(
            csv_dict_head["Layer"]), f'{csv_dict_head} \n {len(csv_dict_head["Node"])} \t { len(csv_dict_head["Layer"])}'
        self.csv_dict_head = csv_dict_head

    def get_full_head(self, mem_ins_len, mem_name_list: list[str]):
        mem_list_head_1 = ["memory utilization"] + [""] * (mem_ins_len - 1)  # ['memory utilization', '', '', '', '', ''] for example_wioGB
        self.csv_dict_head["Layer"] += mem_list_head_1
        self.csv_dict_head["Node"] += mem_name_list
        assert len(self.csv_dict_head["Node"]) == len(
            self.csv_dict_head["Layer"]), f'{self.csv_dict_head} \n {len(self.csv_dict_head["Node"])} \t { len(self.csv_dict_head["Layer"])}'
        return self.csv_dict_head


def get_mem_ins_len(json_dict_data):
    return len(json_dict_data["memory"]["mem_utili_instance"])


def get_mem_ins_to_set_HEAD(file_list: list[str]):
    for file_name in file_list:
        if not file_name.endswith("Conv.json"):
            continue
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as ff:  # 打开当前 Node 的json文件
            data = json.load(ff)
        if data.get("memory"):
            return (get_mem_ins_len(data), list(data["memory"]["mem_utili_instance"].keys()))


def get_total_layers_row(folder_path):
    # match All layers, find total en & la
    with open(folder_path + "/All_Layers.json", "r") as f_all_layer:
        data_total = json.load(f_all_layer)
        total_energy_network = data_total['energy(mJ)']
        total_latency_network = data_total['latency(mC)']
    total_layers_row = [
        "Total layers",  # 0
        "All",  # 1
        "All",  # 2
        total_latency_network,  # 3
        data_total['latency breakdown']["ideal computation latency"],  # 4
        data_total['latency breakdown']["spatial stall"],  # 5
        data_total['latency breakdown']["temporal stall"],  # 6
        data_total['latency breakdown']["onloading latency"],  # 7
        data_total['latency breakdown']["offloading latency"],  # 8
        '-',  # 9
        '-',  # 10
        '-',  # 11
        total_energy_network,  # 12
        data_total["energy breakdown"]["mac energy"] / total_energy_network,  # 13
        data_total["energy breakdown"]["memory energy"] / total_energy_network,  # 14
        100,  # 15
        100,  # 16
    ]
    return total_layers_row, total_energy_network, total_latency_network, data_total


def get_csv_2_rows_HEAD(file_list: list[str]):
    mem_len, mem_names = get_mem_ins_to_set_HEAD(file_list)
    # mem_names = [_split_mem_name_str_(mem_name) for mem_name in mem_names_raw]    # 还不能把 '-' 后面的操作数符号去掉, 因为会混淆两个同名的mem
    csv_dict_head = csv_HEAD().get_full_head(mem_len, mem_names)
    return csv_dict_head, mem_names, mem_len  # dict with 2 keys, which are 2 rows of head


def _add_I2_if_only_I1_in_mem_name_(mem_name: str) -> str:
    name, ops = tuple(mem_name.split('-'))
    ops_li = ast.literal_eval(ops)
    if 'I1' in ops_li and 'I2' not in ops_li:
        # "sram_64KB-['I1']" -> "sram_64KB-['I1', 'I2']"
        ops_li.append('I2')
        return name + '-' + str(ops_li)
    elif ops_li == ['I1', 'I2', 'O']:
        # "dram-['I1', 'I2', 'O']" -> "dram-['I1', 'O', 'I2']"
        return name + '-' + "['I1', 'O', 'I2']"
    return mem_name


def set_csv(folder_path: str):
    file_list = os.listdir(folder_path)
    csv_dict_head, mem_names, mem_len = get_csv_2_rows_HEAD(file_list)  # get HEAD
    total_layers_row, total_energy_network, total_latency_network, data_total = get_total_layers_row(folder_path)  # get total layers data

    # save csv
    ## csv 路径
    csv_file_name = folder_path.split("/")[-1]  # csv_file_name 实际上是 experiment id
    csv_file_path = f"outputs_csv/" + csv_file_name + ".csv"
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    ## 按行写入 csv
    with open(csv_file_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        ### 写两行表头
        for key, values in csv_dict_head.items():
            csv_writer.writerow([key] + values)
        ### 遍历 文件夹下的 json 文件
        for file_name in file_list:
            this_row = [0] * (len(csv_dict_head["Node"]) + 1)  ### 先创建一个长度为表头的list，里面填充0
            #### 过滤文件
            if (not file_name.endswith(".json")) or (
                    file_name.endswith("simple.json")) or file_name.startswith("All"):  # 滤掉 非json、 simple json、All_Layers
                continue
            #### 提取数据
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as ff:  # 打开当前 Node 的json文件
                data = json.load(ff)
            #### match Node
            match_LayerNode = re.match(r"LayerNode*", file_name)  # 匹配 LayerNode
            match_MemNode = re.match(r"MemNode*", file_name)  # 匹配 MemrNode
            if match_LayerNode:
                this_row[:] = [
                    file_name[:-5],  # 0 表头，最左侧一列
                    data['layer']['layer id'],  # 1
                    data['layer']['layer name'],  # 2
                    data["latency"]["total_latency"],  # 3
                    data["latency"]["ideal computation"],  # 4
                    data["latency"]["spatial stall"],  # 5
                    data["latency"]["temporal stall (SS_comb)"],  # 6
                    data["latency"]["data onloading latency"],  # 7
                    data["latency"]["data offloading latency"],  # 8
                    data["MAC utilization"]["MAC spatial utilization"] * 100,  # 9
                    data["MAC utilization"]["MAC utilization 0"] * 100,  # 10
                    data["MAC utilization"]["MAC utilization 2"] * 100,  # 11
                    data["energy"]["total_energy(mJ)"],  # 12
                    data["energy"]["MAC energy"] * 100 / data["energy"]["total_energy(mJ)"],  # 13
                    data["energy"]["mem energy"] * 100 / data["energy"]["total_energy(mJ)"],  # 14
                    data["latency"]["total_latency"] * 100 / total_latency_network,  # 15
                    data["energy"]["total_energy(mJ)"] * 100 / total_energy_network,  # 16
                ]
                if len(data["memory"]["mem_utili_instance"]) == mem_len:
                    this_row += [data["memory"]["mem_utili_instance"][mem_name] for mem_name in mem_names]  # 把 buffer 利用率的信息补在后面
                else:  # add or mul
                    this_row += [data["memory"]["mem_utili_instance"].get(_add_I2_if_only_I1_in_mem_name_(mem_name), 0)
                                 for mem_name in mem_names]  # 把 buffer 利用率的信息补在后面
            elif match_MemNode:
                indices_to_change = [0, 1, 2, 3, 12, 14, 15, 16]  # 一行中要设置的序号（其余为0）
                set_values = [
                    file_name[:-5],  # 0 表头，最左侧一列
                    data['layer id'],  # 1
                    data['layer name'],  # 2
                    data["latency(mC)"],  # 3
                    data["energy(mJ)"],  # 12
                    100,  # 14
                    data["latency(mC)"] * 100 / total_latency_network,  # 15
                    data["energy(mJ)"] * 100 / total_energy_network,  # 16
                ]
                this_row = [set_values[indices_to_change.index(i)] if i in indices_to_change else 0 for i in range(len(csv_dict_head["Node"]) + 1)]
            else:
                raise NameError(file_name)
            assert len(this_row) == len(csv_dict_head["Node"]) + 1
            #### 写每层的数据
            csv_writer.writerow(this_row)
        ### 写 total 数据
        csv_writer.writerow(total_layers_row)
        csv_writer.writerow([
            'Total_Fps',
            'All',
            'All',
            (1.0 / total_latency_network * 1000),
            (1.0 / data_total['latency breakdown']["ideal computation latency"] * 1000),
        ])
    return csv_file_path


def sort_csv_with_layer_id(csv_file_path):
    # 按照 layer id 进行排序
    with open(csv_file_path, 'r', newline='') as to_sort_file:
        csv_reader = csv.reader(to_sort_file)
        data = list(csv_reader)
        head_0, head_1, *layers, total_0, total_1 = data  # 去头去尾, 取出中间需要排序的数据
        sorted_data = sorted(layers, key=lambda x: int(x[1]))

    with open(csv_file_path, 'w', newline='') as sorted_file:  # 排好序后, 再写入
        writer = csv.writer(sorted_file)
        writer.writerows([head_0, head_1])
        writer.writerows(sorted_data)
        writer.writerows([total_0, total_1])


if __name__ == '__main__':
    #! 存放 output json 的文件夹路径
    folder_path = "outputs/example_wioGB--resnet50"

    csv_file_path = set_csv(folder_path)
    sort_csv_with_layer_id(csv_file_path)
    print(f"---------- export csv file to {csv_file_path} DONE! ----------")
