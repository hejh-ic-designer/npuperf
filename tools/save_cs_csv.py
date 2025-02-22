import csv
import json
import os
import re
from tools.Mem_hier_gen import CS0, CS1, CS2, CS3, CS4, CS5, CS6, CS7, nb_of_mems


class csv_HEAD:

    def __init__(self, mems_list):
        ''' 初始化最初的表头 '''
        # 设置表头, 两个表头必须一一对应起来
        csv_dict_head = {}
        csv_dict_head["Mem_hier"] = ["index"] + mems_list + [
            # energy
            "total energy(mJ)",  # 2
            # latency
            "total latency(mC)",  # 3
            "ideal latency(mC)",  # 4
            "spatial stall(mC)",  # 5
            "temporal stall(mC)",  # 6
            "onloading latency(mC)",  # 7
            "offloading latency(mC)",  # 8
        ]
        self.csv_dict_head = csv_dict_head

    def get_full_head(self):
        return self.csv_dict_head


def set_csv(folder_path: str, mems_list):
    file_list = os.listdir(folder_path)
    csv_dict_head = csv_HEAD(mems_list).get_full_head()  # get HEAD

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
        for id, file_name in enumerate(file_list):
            #### 提取数据
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as ff:  # 打开当前 Node 的json文件
                data = json.load(ff)

            mem_hier_name = file_name[:-16]  # 0 表头，最左侧一列, 16是为了去掉  _All_layers.json
            gb_lbs = mem_hier_name.split('+')
            if len(gb_lbs) == 1:
                lbs = gb_lbs[0].split('_')
                mems_to_insert = lbs
            elif len(gb_lbs) == 2:
                gb, lbs = gb_lbs
                lbs = lbs.split('_')
                mems_to_insert = [gb] + lbs
            else:
                raise ValueError
            reform = [re.findall(r'\d+', a)[0] for a in mems_to_insert]
            if reform[0] in ['0', 0]:
                reform[0] = '0.5'

            #### match Node
            this_row = [mem_hier_name, id] + reform + [
                data['energy(mJ)'],  # 2
                data['latency(mC)'],  # 3
                data['latency breakdown']['ideal computation latency'],  # 4
                data['latency breakdown']['spatial stall'],  # 5
                data['latency breakdown']['temporal stall'],  # 6
                data['latency breakdown']['onloading latency'],  # 7
                data['latency breakdown']['offloading latency'],  # 8
            ]
            #### 写每个文件的数据
            csv_writer.writerow(this_row)
    return csv_file_path


if __name__ == '__main__':
    CASE_list = [
        # CS0,
        # CS1,
        # CS2,
        # CS3,
        # CS4,
        # CS5,
        # CS6,
        CS7,
    ]

    # NN = 'fsrcnn2x'
    NN = 'mv1'
    for CASE in CASE_list:

        folder_path = f"outputs/Case_Study_{str(CASE.__name__)[-1]}--{NN}"  #! 存放 output json 的文件夹路径
        mems_list = nb_of_mems[str(CASE.__name__)]

        csv_file_path = set_csv(folder_path, mems_list)
        print(f"---------- export csv file to {csv_file_path} DONE! ----------")
