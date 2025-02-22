import logging as _logging
from npuperf.classes.opt.hw_gen.core_generator import CoreGenerator
from npuperf.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph_new

_logging_level = _logging.INFO
_logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
_logging.basicConfig(level=_logging_level,
                     format=_logging_format,
                     )
logger = _logging.getLogger(__name__)
logger.setLevel(_logging.DEBUG)

core_info = {   # 2048 MACs
    'MAC_unroll': {
        'D1': ('K', 8),
        'D2': ('C', 32),
        'D3': ('OX', 4),
        'D4': ('OY', 2)
    },
    'local_buffers': [
        {
            'op': 'W',
            'size': 128 * 1024 * 8
        },
        {
            'op': 'I',
            'size': 64 * 1024 * 8
        },
        {
            'op': 'O',
            'size': 128 * 1024 * 8
        },
    ],
    'global_buffer':{
        'op': 'W/I/O',
        'size': 1 * 1024 * 1024 * 8,
        'bandwidth': 256
    },
    'dram': {
        'op': 'W/I/O',
        'size': 4 * 1024 * 1024 * 1024 * 8,
        'bandwidth': 192
    },
}

# core_info = {     # 1024 MACs
#     'MAC_unroll': {
#         'D1': ('K', 8),
#         'D2': ('C', 32),
#         'D3': ('OX', 2),
#         'D4': ('OY', 2)
#     },
#     'local_buffers': [
#         {
#             'op': 'W',
#             'size': 64 * 1024 * 8
#         },
#         {
#             'op': 'I',
#             'size': 64 * 1024 * 8
#         },
#         {
#             'op': 'O',
#             'size': 128 * 1024 * 8
#         },
#     ],
#     'dram': {
#         'op': 'W/I/O',
#         'size': 4 * 1024 * 1024 * 1024 * 8,
#         'bandwidth': 96
#     },
# }

coregen = CoreGenerator.from_dict(core_info)
filename = './output_coregen-tst.json'
coregen.save_to_json(file_path=filename)
visualize_memory_hierarchy_graph_new(coregen.get_core().get_memory_hierarchy())
