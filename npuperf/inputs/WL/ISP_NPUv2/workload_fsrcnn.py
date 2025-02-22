workload = {
    -1: {'equation': 'input',
         'loop_dim_size':  {'B': 1, 'K': 1, 'OY': 540, 'OX': 960},
         'precision': 8,
         'core_allocation': 1,
         'memory_operand_links': {'O': 'I1'}
        }
    ,
    0: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 56, 'C': 1, 'OY': 540, 'OX': 960, 'FY': 5, 'FX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    1: {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ox][oy]',
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 56, 'OY': 540, 'OX': 960},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    2: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    3: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    4: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [3]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    5: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 12, 'C': 12, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [4]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    6: {'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ox][oy]',
        'loop_dim_size': {'B': 1, 'K': 56, 'C': 12, 'OY': 540, 'OX': 960},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [5]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    7: {  # Addition of layer 1 (residual path) and layer 3 (main path)
        'equation': 'O[b][g][oy][ox]=X[b][g][oy][ox]+Y[b][g][oy][ox]',
        'equation_relations': [],
        'loop_dim_size': {'B': 1, 'G': 56, 'OY': 540, 'OX': 960},
        'operand_precision': {'O': 16, 'O_final': 8, 'X': 8, 'Y': 8},
        'operand_source': {'X': [0], 'Y': [6]},
        'constant_operands': [],
        'operand_source_dimension_mapping': {'X': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}, 'Y': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('G', 4), ('OY', 4)], 'D2': [('OX', 32)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'X': 'R', 'Y': 'I1'}
    }
    # ,
    # 7: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
    #     'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
    #     'loop_dim_size': {'B': 1, 'K': 1, 'C': 56, 'OY': 1620, 'OX': 2880, 'FY': 9, 'FX': 9},
    #     'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
    #     'operand_source': {'W': [], 'I': [6]},
    #     'constant_operands': ['W'],
    #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
    #     'core_allocation': 1,
    #     'spatial_mapping': {'D1': [('C', 16)], 'D2': [('OX', 32)], 'D3' : [('OY', 4)]},  # Must match with the dimensions of core 1
    #     'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    #     }
}