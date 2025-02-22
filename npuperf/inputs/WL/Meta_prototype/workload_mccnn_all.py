workload = {
    -1: {
        'equation': 'input',
         'loop_dim_size':  {'B': 1, 'K': 1, 'OY': 1242, 'OX': 376},
         'precision': 8,
         'core_allocation': 1,
         'memory_operand_links': {'O': 'I1'}
         }
    ,
    0: {'operator_type':'Conv',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 64, 'C': 1, 'OY': 1248, 'OX': 382, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [-1]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        }
    ,
    1: {'operator_type':'Conv',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 32, 'C': 64, 'OY': 1246, 'OX': 380, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        }
    ,
    2: {'operator_type': 'dw Conv',         # Conv dw / s1
        'equation': 'O[b][g][oy][ox]+=W[g][fy][fx]*I[b][g][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 32, 'OY': 1244, 'OX': 378, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    3: {'operator_type':'pw Conv',
        'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 32, 'C': 32, 'OY': 1244, 'OX': 378, 'FY': 1, 'FX': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'G'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        }
    ,
    4: {
        'operator_type': 'Add',  # Addition
        'equation': 'O[b][g][oy][ox]=X[b][g][oy][ox]+Y[b][g][oy][ox]',
        'equation_relations': [],
        'loop_dim_size': {'B': 1, 'G': 32, 'OY': 1242, 'OX': 376},
        'operand_precision': {'O': 16, 'O_final': 8, 'X': 8, 'Y': 8},
        'operand_source': {'X': [1], 'Y': [3]},
        'constant_operands': [],
        'operand_source_dimension_mapping': {'X': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}, 'Y': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'X': 'I1', 'Y': 'I1'}
        #        'operator_type': 'Transpose',
        #        'equation': 'transpose',
        #        'equation_relations': [],
        #        'loop_dim_size': {'B': 1, 'G': 32, 'OX': 374, 'OY': 1240},
        #        'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
        #        'operand_source': {'I': [3]},
        #        'constant_operands': [],
        #        'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}},
        #        'core_allocation': 1,
        #    #     'spatial_mapping': {},
        #        'memory_operand_links': {'O': 'O', 'I': 'I1'}
        },     

    5: {'operator_type':'PS_Conv',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 32, 'C': 32, 'OY': 1240, 'OX': 374, 'FY': 3, 'FX': 3},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [4]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        # 'post_process': 'pixel_shuffle'
        },

    6: {
       'operator_type': 'Transpose',
       'equation': 'transpose',
       'equation_relations': [],
       'loop_dim_size': {'B': 1, 'G': 32, 'OX': 374, 'OY': 1240},
       'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
       'operand_source': {'I': [5]},
       'constant_operands': [],
       'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}},
       'core_allocation': 1,
   #     'spatial_mapping': {},
       'memory_operand_links': {'O': 'O', 'I': 'I1'}},     #todo 这个还要再看，要层融合的话，得跟前面对的上

    7: {
       'operator_type': 'Concat',  # 在网络描述中，输入是若干OX OY相同的特征图，但是通道数不一定相同，分别用X1, X2, ... 表示，输出为 O
       'equation': 'concat',
       'equation_relations': [],
       'loop_dim_size': {'B': 1, 'G': 96, 'OX': 374, 'OY': 1240},
       'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
       'operand_source': {'X1': [3], 'X2': [5], 'X3': [6] },    
       'constant_operands': [],
       'operand_source_dimension_mapping': {'X1': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}, 'X2': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}, 'X3': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}},   # 
       'core_allocation': 1,
   #     'spatial_mapping': {},
       'memory_operand_links': {'O': 'O', 'I': 'I1'}},     #????? 不确定咋弄


    8: {'operator_type': 'Pool',  # max pool, stride 2
        'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'G': 96, 'OY': 617, 'OX': 184, 'FX': 5, 'FY': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'I': 8, 'W': 0},
        'operand_source': {'W': [], 'I': [7]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'G': 'G'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('G', 32), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'I': 'I1', 'W': 'I2'}
    }
    ,
    9: {'operator_type': 'Conv',        # stride 2
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 32, 'C': 96, 'OY': 306, 'OX': 89, 'FY': 7, 'FX': 7},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [8]},
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'G'}},
        'constant_operands': ['W'],
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2), 'D3': ('OX', 4), 'D4': ('OY', 4)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,

    10: {'operator_type': 'Fc',  # fc
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
        'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 10, 'C': 871488, 'OY': 1, 'OX': 1, 'FY': 1, 'FX': 1},        # k = 10, 不足展开度32
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [9]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'G'}},
        'core_allocation': 1,
        'spatial_mapping': {'D1': ('K', 32), 'D2': ('C', 2)},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }

}
