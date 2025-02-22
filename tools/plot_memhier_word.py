"""
从pickle 文件中读取 List[CME], 作图
"""
import pickle
import os
from npuperf.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph_Word_Access

experiment_name = 'example_wioGB--resnet18'
# pickle_path = f'outputs/{experiment_name}/{experiment_name}-saved_list_of_cmes.pickle'  #! 路径注意区分 stationary
pickle_path = f'outputs_OS/{experiment_name}/{experiment_name}-saved_list_of_cmes.pickle'  #! 路径注意区分 stationary
with open(pickle_path, "rb") as handle:
    list_of_cme = pickle.load(handle)

# 要保存的文件夹路径
save_directory = f'output_word_access_graph/{experiment_name}/'
# 检查目录是否存在，如果不存在则创建它
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for id, cme in enumerate(list_of_cme):
    mem_hier = cme.accelerator.get_core(1).get_memory_hierarchy()
    mem_word_access = cme.memory_word_access
    layer_type = cme.layer.TYPE
    save_path = f'{save_directory}layer_{id}_{layer_type}.png'
    if layer_type != 'Add':
        visualize_memory_hierarchy_graph_Word_Access(mem_hier, mem_word_access, save_path)
print(f'Done! picture export to path {save_directory}')
