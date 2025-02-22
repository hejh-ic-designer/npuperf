from npuperf.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph, visualize_memory_hierarchy_graph_new
import argparse
import importlib
parser = argparse.ArgumentParser(description="Setup memory hierarchy visualization")
parser.add_argument('--hw', metavar='hardware name', required=True, help='module path to hardware, e.g. example_wioGB')
args = parser.parse_args()
hardware_path = '.'.join(['npuperf', 'inputs', 'HW', args.hw])
module = importlib.import_module(hardware_path)
accelerator = module.accelerator
memhier = accelerator.get_core(1).get_memory_hierarchy()

# visualize_memory_hierarchy_graph(memhier)
visualize_memory_hierarchy_graph_new(memhier)
