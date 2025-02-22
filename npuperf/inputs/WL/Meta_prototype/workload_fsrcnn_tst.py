workload = {    
    -1: {'equation': 'input',
         'loop_dim_size':  {'B': 1, 'K': 56, 'OY': 550, 'OX': 970},
         'precision': 8,
         'core_allocation': 1,
         'memory_operand_links': {'O': 'I1'}
         }
    ,    
    0: {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ox][oy]',
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 56, 'OY': 550, 'OX': 970},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'constant_operands': ['W'],
        # 'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 12), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,    
    1: {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fx'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 56, 'OY': 550, 'OX': 970, 'FX': 1, 'FY': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        # 'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 12), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
}