from npuperf.visualization.graph.dnn import visualize_dnn_graph
from npuperf.classes.workload.dnn_workload import DNNWorkload

from npuperf.inputs.WL_fromjson.Meta_prototype.workload_mv1 import workload


workload = DNNWorkload(workload)

visualize_dnn_graph(workload)