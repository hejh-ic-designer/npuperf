import json
import os
from collections import Counter


def check_WIO_or_XYO(layer: dict):
    assert (len(layer["inputs"]) == 3) or (len(layer["inputs"]) == 2), f'unexpected layer: {layer}'
    if layer["inputs"][1]["is_const"] and not layer['inputs'][0]['is_const']:
        return 'tensor_x_weight'
    elif (not layer["inputs"][1]["is_const"]) and (not layer['inputs'][0]['is_const']):
        return 'tensor_x_tensor'
    else:
        raise ValueError(f'unexpected layer dict: {layer}')


def check_op_percent(Counter_di: dict):
    dummy_op = set({
        'qnn.csi.lrn',
        'qnn.csi.softmax',
        'qnn.csi.reshape',
        'qnn.csi.clip',
        'qnn.csi.sigmoid',
        'qnn.csi.mean',
        'qnn.csi.tanh',
        'qnn.csi.power',
        'qnn.csi.sqrt',
        'qnn.csi.div',
        'qnn.csi.take',
        'qnn.csi.erf',
        'qnn.csi.strided_slice',
        'qnn.csi.variance',
        'qnn.csi.cast',
        'qnn.csi.sin',
        'qnn.csi.cos',
        'qnn.csi.upsampling',
        'qnn.csi.exp',
    })
    dummy_op.update({'qnn.csi.relu', 'qnn.csi.prelu'})

    sumsum = sum(Counter_di.values())
    dummy_in_di = {op: ct for op, ct in Counter_di.items() if f'qnn.csi.{op}' in dummy_op}
    dummy_percent = sum(dummy_in_di.values()) * 100 / sumsum

    print(f'total layers: {sumsum}, dummy percent: {dummy_percent}, dummy ops: {dummy_in_di.keys()}')
    for op, ct in Counter_di.items():
        print(f'{op}: {ct*100/sumsum}')


def show_op(nn):
    """ 显示json格式的网络描述下一个网络的所有算子类型 """
    json_workload_path = f'npuperf/inputs/hhb_networks/{nn}'
    with open(json_workload_path) as f:
        json_all = json.load(f)
    layer_li = json_all['layers']
    OP_li_all = []
    matmul_all = []
    add_all = []
    mul_all = []
    for layer in layer_li:
        op_type: str = layer['op_type'].split('.')[-1]
        if op_type == 'matmul':
            matmul_all.append(check_WIO_or_XYO(layer))
        if op_type == 'add':
            add_all.append(check_WIO_or_XYO(layer))
        if op_type == 'mul':
            mul_all.append(check_WIO_or_XYO(layer))
        OP_li_all.append(op_type)
    operator_counter = Counter(OP_li_all)
    matmul_counter = Counter(matmul_all)
    add_counter = Counter(add_all)
    mul_counter = Counter(mul_all)
    # op_li = set(OP_li_all)
    # print(f'Network: {nn:25} // {op_li}')
    nn = nn[:-5]
    print('---' * 30)
    print(f'Network: {nn:10} {len(OP_li_all):3} layers // {operator_counter}')
    print(f'matmul: {matmul_counter}, add: {add_counter}, mul: {mul_counter}')
    check_op_percent(operator_counter)


if __name__ == '__main__':
    WL_json_path = f'npuperf/inputs/hhb_networks'
    path_list = os.listdir(WL_json_path)  # os.listdir 列出文件夹中的文件
    print(f'path_list is: {path_list}')
    for nn in path_list:
        show_op(nn)  #显示一个网络里的所有算子类型
