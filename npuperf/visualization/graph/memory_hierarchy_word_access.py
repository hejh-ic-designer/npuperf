import networkx as nx
import matplotlib.pyplot as plt
import math

from npuperf.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy

def visualize_memory_hierarchy_graph_Word_Access(G: MemoryHierarchy, mem_word_access: dict[str, list] = None, save_path = None):
    """
    Visualizes a memory hierarchy graph with data word access info.
    """
    generations = list(nx.topological_generations(G))

    pos = {}
    node_list = []
    node_size_list = []
    node_label_dict = {}
    for gen_idx, generation in enumerate(generations):
        y = gen_idx
        node_size = (gen_idx + 1) * 300
        for node_idx, node in enumerate(generation):
            # print(node.operands)
            if node.operands == ['I2']:
                x = 1
            elif node.operands == ['I1']:
                x = 2
            elif node.operands == ['I1', 'O']:
                x = 3
            elif node.operands == ['O']:
                x = 4
            else:
                x = 2.5
            pos[node] = (x, y)
            node_list.append(node)
            node_size_list.append(node_size)
            node_label_dict[node] = f"{node.name}\n{node.operands}"

    # nx.draw(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)
    # plt.figure(figsize=(100, 80))   #! 图片大小
    fig, ax= plt.subplots(figsize=(16, 10))   # 原本是 (10, 8)

    # 在绘图之前设置X和Y轴的范围
    x_min = min(x for x, _ in pos.values())
    x_max = max(x for x, _ in pos.values())
    y_min = min(y for _, y in pos.values())
    y_max = max(y for _, y in pos.values())

    # 添加一些额外的空白边距，以确保所有内容都可见
    x_margin = 0.2  # 调整X轴范围的额外空白边距
    y_margin = 0.2  # 调整Y轴范围的额外空白边距

    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    nx.draw_networkx(G, pos=pos, node_shape='s', nodelist=node_list, node_size=node_size_list, labels=node_label_dict)

    plt.title(G.name)

    # 添加箭头和文字
    half_len_list = [math.sqrt(node_size) / 200 for node_size in node_size_list]

    for id, (node, (x, y)) in enumerate(pos.items()):
        # print(node.name)    # 可以利用这两个
        # print(node.mem_level_of_operands)
        # 先取数据
        single_node_word_access = {}
        for operand, lvl in node.mem_level_of_operands.items():
            if operand == 'I1':
                single_node_word_access[operand] = mem_word_access['I'][lvl]    # 取出来一个 4waydatamoving
            elif operand == 'I2':
                single_node_word_access[operand] = mem_word_access['W'][lvl]    # 取出来一个 4waydatamoving
            else: # 'O'
                single_node_word_access[operand] = mem_word_access['O'][lvl]    # 取出来一个 4waydatamoving

        four_arrow_word_access = {}

        rd_out_to_high_0 = [four_way.rd_out_to_high for four_way in single_node_word_access.values()]
        rd_out_to_high_0 = [str(i) for i in rd_out_to_high_0]   # int 转成 str
        four_arrow_word_access[0] = '+'.join(rd_out_to_high_0)

        wr_in_by_low_1 = [four_way.wr_in_by_low for four_way in single_node_word_access.values()]
        wr_in_by_low_1 = [str(i) for i in wr_in_by_low_1]   # int 转成 str
        four_arrow_word_access[1] = '+'.join(wr_in_by_low_1)

        wr_in_by_high_2 = [four_way.wr_in_by_high for four_way in single_node_word_access.values()]
        wr_in_by_high_2 = [str(i) for i in wr_in_by_high_2]   # int 转成 str
        four_arrow_word_access[2] = '+'.join(wr_in_by_high_2)

        rd_out_to_low_3 = [four_way.rd_out_to_low for four_way in single_node_word_access.values()]
        rd_out_to_low_3 = [str(i) for i in rd_out_to_low_3]   # int 转成 str
        four_arrow_word_access[3] = '+'.join(rd_out_to_low_3)



        half_len = half_len_list[id]

        # 四个箭头的位置
        arrow_positions_end = [
            (x-1.2*half_len, y+0.5*half_len),
            (x-1.2*half_len, y-1.5*half_len),
            (x+1.2*half_len, y+1.5*half_len),
            (x+1.2*half_len, y-0.5*half_len),
            ]

        arrow_positions_start = [
            (x-1.2*half_len, y+1.5*half_len),
            (x-1.2*half_len, y-0.5*half_len),
            (x+1.2*half_len, y+0.5*half_len),
            (x+1.2*half_len, y-1.5*half_len),
            ]
        # 箭头的方向
        arrow_directions = ['up', 'up', 'down', 'down']

        for index, (position_st, position_ed, direction)in enumerate(zip(arrow_positions_start, arrow_positions_end, arrow_directions)):
            # 画四个箭头
            # index 0~3: 左上，左下，右上，右下。 分别为：rd/\, wr/\, wrV, rdV
            plt.annotate("", xy=position_st, xytext=position_ed,
                         arrowprops=dict(arrowstyle='->',
                                         lw=1.5, alpha=0.7))

            # 添加文字，把变量 memory_word_access 标识在箭头旁边
            text_offset = 0.03
            if direction == 'up':
                plt.text(position_ed[0], position_ed[1]-text_offset, f'{four_arrow_word_access[index]}', fontsize = 10, ha='right')
            else:
                plt.text(position_ed[0], position_ed[1]+text_offset, f'{four_arrow_word_access[index]}', fontsize = 10, ha='left')

    fig.tight_layout()
    plt.savefig(save_path, dpi=100)
    # plt.show()


# meta_prototype:
# rf_1B
# {'I2': 0}
# rf_2B
# {'O': 0}
# sram_32KB
# {'I1': 0}
# sram_64KB
# {'I2': 1}
# sram_1MB_A
# {'I1': 1, 'O': 1}
# sram_1MB_W
# {'I2': 2}
# dram
# {'I1': 2, 'I2': 3, 'O': 2}

