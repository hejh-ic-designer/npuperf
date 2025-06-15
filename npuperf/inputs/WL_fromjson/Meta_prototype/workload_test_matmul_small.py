workload = {
    -1: {
        'equation': 'input',
        'loop_dim_size': {'B': 1, 'G': 128},
        'precision': 8,
        'core_allocation': 1,
        'memory_operand_links': {'O': 'I1'}
    },
    0: {
        'operator_type': 'Matmul',
        'equation': 'O[k][ox]+=W[k][c]*I[c][ox]',
        'equation_relations': [],
        'loop_dim_size': {'K': 1024, 'C': 1024, 'OX': 1024},
        'operand_precision': {'O': 16,'O_final': 8,'W': 8,'I': 8},
        'operand_source': {'I': [-1]},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C',2), 'D3': ('OX', 4), 'D4': ('OX', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },
    1: {
        'operator_type': 'Matmul',
        'equation': 'O[k][ox]+=W[k][c]*I[c][ox]',
        'equation_relations': [],
        'loop_dim_size': {'K': 512, 'C': 2048, 'OX': 512},
        'operand_precision': {'O': 16,'O_final': 8,'W': 8,'I': 8},
        'operand_source': {'I': [0]},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C',2), 'D3': ('OX', 4), 'D4': ('OX', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },
    2: {
        'operator_type': 'Matmul',
        'equation': 'O[k][ox]+=W[k][c]*I[c][ox]',
        'equation_relations': [],
        'loop_dim_size': {'K': 128, 'C': 2048, 'OX': 128},
        'operand_precision': {'O': 16,'O_final': 8,'W': 8,'I': 8},
        'operand_source': {'I': [1]},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C',2), 'D3': ('OX', 4), 'D4': ('OX', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    },
    3: {
        'operator_type': 'Matmul',
        'equation': 'O[k][ox]+=W[k][c]*I[c][ox]',
        'equation_relations': [],
        'loop_dim_size': {'K': 2048, 'C': 2048, 'OX': 2048},
        'operand_precision': {'O': 16,'O_final': 8,'W': 8,'I': 8},
        'operand_source': {'I': [2]},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C',2), 'D3': ('OX', 4), 'D4': ('OX', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
}
