import pickle
from npuperf.visualization.results.print_mapping_tofile import print_mapping_tofile
from npuperf.visualization.results.print_mapping import print_mapping
from npuperf.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph

experiment_name = 'example_wioGB--resnet18'
# Load in the pickled CME
pickle_path = f'outputs/{experiment_name}/{experiment_name}-saved_list_of_cmes.pickle'
with open(pickle_path, 'rb') as fp:
    cme_for_the_multi_layer = pickle.load(fp)

#= single layer
# visualize_memory_hierarchy_graph(cme_for_the_single_layer[0].accelerator.cores[0].memory_hierarchy)
# for cme in cme_for_the_single_layer:
#     print_mapping(cme)

#= multi layer
# visualize_memory_hierarchy_graph(cme_for_the_multi_layer[0].accelerator.cores[0].memory_hierarchy)
# for cme in cme_for_the_multi_layer:
#     print_mapping(cme)

#= multi layer, print to file
# file_path = f'outputs/{experiment_name}/{experiment_name}-temporal_mapping.txt'
file_path = f'result_plot/{experiment_name}-temporal_mapping.txt'
# visualize_memory_hierarchy_graph(cme_for_the_multi_layer[0].accelerator.cores[0].memory_hierarchy)
print_mapping_tofile(cme_for_the_multi_layer, file_path)
print(f'Done!, print temporal mapping at path {file_path}')
