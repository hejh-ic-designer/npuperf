import csv
import json
import os
import re

# 存放 output json 的文件夹路径
folder_path = "outputs/example_wioGB--mv1"
file_list = os.listdir(folder_path)
# print(len(file_list))
# print(file_list)

latency_dict = {}

# 设置表头
latency_dict_head = {}
latency_dict_head["node"] = [
    "id",
    "actual latency(ms)",
    "ideal latency(ms)",
    "spatial stall(ms)",
    "temporal stall(ms)",
    "spatial utilization",
    "spatial temporal utilization",
    "last utilization",
    "latency percentage",
]

for file_name in file_list:
    ##匹配节点
    aaa = r"LayerNode*"
    match = re.match(aaa, file_name)  # 匹配 LayerNode
    if match and file_name.endswith(".json") and not file_name.endswith("simple.json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
            # get total_actual_latency
            latency = data["latency"]["total_latency"]
            ideal_computation = data["latency"]["ideal computation"]
            spatial_stall = data["latency"]["spatial stall"]
            temporal_stall = data["latency"]["temporal stall (SS_comb)"]
            MAC_spatial_utilization = data["MAC utilization"]["MAC spatial utilization"]
            MAC_spatial_temporal_utilization = data["MAC utilization"][
                "MAC utilization 0"
            ]
            MAC_utilization = data["MAC utilization"]["MAC utilization 2"]
            # formatted_percentage = f"{percentage:.2f}%"

            layernode_name = file_name[:-5]
            layernode_str = layernode_name.split("_")[1]
            layernode_num = int(layernode_str)

            latency_dict[layernode_name] = [
                layernode_num,
                latency,
                ideal_computation,
                spatial_stall,
                temporal_stall,
                MAC_spatial_utilization,
                MAC_spatial_temporal_utilization,
                MAC_utilization,
            ]
        # file.close()
# print(latency_dict)

# 使用sorted()函数和lambda表达式对字典进行重新排序
latency_dict = dict(sorted(latency_dict.items(), key=lambda x: x[1]))

# 遍历字典的键值对
total_actual_latency = 0
total_ideal_latency = 0
for key, value in latency_dict.items():
    total_actual_latency += value[1]
    total_ideal_latency += value[2]
# print(total_actual_latency)
for key, value in latency_dict.items():
    percent = value[1] / total_actual_latency
    value.append(percent)
latency_dict["total_actual_latency"] = [
    0,
    total_actual_latency,
    total_ideal_latency,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]
actual_fps = 1.0 / total_actual_latency * 1000
ideal_fps = 1.0 / total_ideal_latency * 1000
latency_dict["total_fps"] = [0, actual_fps, ideal_fps, 0, 0, 0, 0, 0, 0]

##拼接
latency_dict_head.update(latency_dict)

# save csv
# csv_file_path = 'resnet50_latency.csv'
csv_file_name = folder_path.split("/")[-1]
csv_file_path = f"outputs_csv/" + csv_file_name + ".csv"
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
with open(csv_file_path, "w", newline="") as file:
    csv_writer = csv.writer(file)
    for key, value in latency_dict_head.items():
        csv_writer.writerow(
            [
                key,
                value[0],
                value[1],
                value[2],
                value[3],
                value[4],
                value[5],
                value[6],
                value[7],
                value[8],
            ]
        )
print(f"---------- export csv file to {csv_file_path} DONE! ----------")
