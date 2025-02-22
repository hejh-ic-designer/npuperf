# pe处理粒度：32w*4h*4ic*4oc
# tile：32w*32h
# xolp和yolp均做cache

# 网络：
# =====================================================================================================================================================================
# Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Kernel Shape              Mult-Adds
# =====================================================================================================================================================================
# SESR_M3_1_3_DNDM                         [1, 1, 540, 960]          [1, 3, 540, 960]          --                        --                        --
# ├─Conv2d: 1-1                            [1, 1, 540, 960]          [1, 16, 540, 960]         416                       [5, 5]                    215,654,400
# ├─Conv2d: 1-2                            [1, 16, 540, 960]         [1, 16, 540, 960]         2,320                     [3, 3]                    1,202,688,000
# ├─PReLU: 1-3                             [1, 16, 540, 960]         [1, 16, 540, 960]         1                         --                        1
# ├─Conv2d: 1-4                            [1, 16, 540, 960]         [1, 16, 540, 960]         2,320                     [3, 3]                    1,202,688,000
# ├─PReLU: 1-5                             [1, 16, 540, 960]         [1, 16, 540, 960]         1                         --                        1
# ├─Conv2d: 1-6                            [1, 16, 540, 960]         [1, 16, 540, 960]         2,320                     [3, 3]                    1,202,688,000
# ├─PReLU: 1-7                             [1, 16, 540, 960]         [1, 16, 540, 960]         1                         --                        1
# ├─Conv2d: 1-8                            [1, 16, 540, 960]         [1, 3, 540, 960]          1,203                     [5, 5]                    623,635,200
# =====================================================================================================================================================================


workload = {
    -1: {'equation': 'input',
         'loop_dim_size':  {'B': 1, 'K': 1, 'OY': 540, 'OX': 960},
         'precision': 8,
         'core_allocation': 1,
         'memory_operand_links': {'O': 'I1'}
        }
    ,
# ├─Conv2d: 1-1                            [1, 1, 540, 960]          [1, 16, 540, 960]         416                       [5, 5]                    215,654,400
    0: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 1, 'OY': 540, 'OX': 960, 'FY': 5, 'FX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
# ├─Conv2d: 1-2                            [1, 16, 540, 960]         [1, 16, 540, 960]         2,320                     [3, 3]                    1,202,688,000
    1: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
# ├─Conv2d: 1-4                            [1, 16, 540, 960]         [1, 16, 540, 960]         2,320                     [3, 3]                    1,202,688,000
    2: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
# ├─Conv2d: 1-6                            [1, 16, 540, 960]         [1, 16, 540, 960]         2,320                     [3, 3]                    1,202,688,000
    3: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
# ├─Conv2d: 1-8                            [1, 16, 540, 960]         [1, 3, 540, 960]          1,203                     [5, 5]                    623,635,200
    4: {'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 3, 'C': 16, 'OY': 540, 'OX': 960, 'FY': 5, 'FX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [3]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': [('K', 4), ('OY', 4)], 'D2': [('OX', 32)], 'D3' : [('C', 4)]},  # Must match with the dimensions of core 1
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
}