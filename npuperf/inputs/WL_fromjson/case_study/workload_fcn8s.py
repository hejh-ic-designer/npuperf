workload ={   -1: {   'equation': 'input',
            'loop_dim_size': {'B': 1, 'G': 3, 'OY': 160, 'OX': 160},
            'precision': 8,
            'core_allocation': 1,
            'memory_operand_links': {'O': 'I1'}},
    0: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 160, 'OX': 160, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [-1]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_0_PART_0_0_fuse_bias_add_Conv_0_1'},
    1: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 64, 'C': 64, 'OY': 160, 'OX': 160, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [0]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_2_PART_0_3_fuse_bias_add_Conv_2_4'},
    2: {   'operator_type': 'Pool',
           'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
           'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'G': 64, 'OY': 80, 'OX': 80, 'FY': 2, 'FX': 2},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 0, 'I': 8},
           'operand_source': {'W': [], 'I': [1]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('G', 8)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'max_pool2d_MaxPool_4_6'},
    3: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 128, 'C': 64, 'OY': 80, 'OX': 80, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [2]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_5_PART_0_7_fuse_bias_add_Conv_5_8'},
    4: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 128, 'C': 128, 'OY': 80, 'OX': 80, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [3]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_7_PART_0_10_fuse_bias_add_Conv_7_11'},
    5: {   'operator_type': 'Pool',
           'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
           'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'G': 128, 'OY': 40, 'OX': 40, 'FY': 2, 'FX': 2},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 0, 'I': 8},
           'operand_source': {'W': [], 'I': [4]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('G', 8)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'max_pool2d_MaxPool_9_13'},
    6: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 256, 'C': 128, 'OY': 40, 'OX': 40, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [5]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_10_PART_0_14_fuse_bias_add_Conv_10_15'},
    7: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 256, 'C': 256, 'OY': 40, 'OX': 40, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [6]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_12_PART_0_17_fuse_bias_add_Conv_12_18'},
    8: {   'operator_type': 'Conv',
           'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
           'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'K': 256, 'C': 256, 'OY': 40, 'OX': 40, 'FY': 3, 'FX': 3},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
           'operand_source': {'W': [], 'I': [7]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'conv2d_Conv_14_PART_0_20_fuse_bias_add_Conv_14_21'},
    9: {   'operator_type': 'Pool',
           'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
           'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
           'loop_dim_size': {'B': 1, 'G': 256, 'OY': 20, 'OX': 20, 'FY': 2, 'FX': 2},
           'operand_precision': {'O': 32, 'O_final': 8, 'W': 0, 'I': 8},
           'operand_source': {'W': [], 'I': [8]},
           'constant_operands': ['W'],
           'core_allocation': 1,
           'spatial_mapping': {'D1': ('G', 8)},
           'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
           'name': 'max_pool2d_MaxPool_16_23'},
    10: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 256, 'OY': 20, 'OX': 20, 'FY': 3, 'FX': 3},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [9]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_17_PART_0_24_fuse_bias_add_Conv_17_25'},
    11: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 20, 'OX': 20, 'FY': 3, 'FX': 3},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [10]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_19_PART_0_27_fuse_bias_add_Conv_19_28'},
    12: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 20, 'OX': 20, 'FY': 3, 'FX': 3},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [11]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_21_PART_0_30_fuse_bias_add_Conv_21_31'},
    13: {   'operator_type': 'Pool',
            'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
            'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'G': 512, 'OY': 10, 'OX': 10, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 0, 'I': 8},
            'operand_source': {'W': [], 'I': [12]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'max_pool2d_MaxPool_23_33'},
    14: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 3, 'FX': 3},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [13]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_24_PART_0_34_fuse_bias_add_Conv_24_35'},
    15: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 3, 'FX': 3},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [14]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_26_PART_0_37_fuse_bias_add_Conv_26_38'},
    16: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 3, 'FX': 3},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [15]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_28_PART_0_40_fuse_bias_add_Conv_28_41'},
    17: {   'operator_type': 'Pool',
            'equation': 'O[b][g][oy][ox]+=W[fx][fy]*I[b][g][ix][iy]',
            'equation_relations': ['ix=2*ox+1*fx', 'iy=2*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'G': 512, 'OY': 5, 'OX': 5, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 0, 'I': 8},
            'operand_source': {'W': [], 'I': [16]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'max_pool2d_MaxPool_30_43'},
    18: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 5, 'OX': 5, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [17]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_31_PART_0_44_fuse_bias_add_ConvTranspose_31_45'},
    19: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 5, 'OX': 5, 'FY': 2, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [17]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_31_PART_0_44_fuse_bias_add_ConvTranspose_31_45'},
    20: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 5, 'OX': 5, 'FY': 1, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [17]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_31_PART_0_44_fuse_bias_add_ConvTranspose_31_45'},
    21: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 512, 'C': 512, 'OY': 5, 'OX': 5, 'FY': 1, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [17]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_31_PART_0_44_fuse_bias_add_ConvTranspose_31_45'},
    22: {   'operator_type': 'Contract',
            'equation': 'contract',
            'loop_dim_size': {'B': 1, 'G': 512, 'OY': 10, 'OX': 10},
            'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
            'operand_source': {'X0': [21], 'X1': [20], 'X2': [19], 'X3': [18]},
            'constant_operands': [],
            'core_allocation': 1,
            'memory_operand_links': {'O': 'O', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_31_PART_0_44_fuse_bias_add_ConvTranspose_31_45'},
    23: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 512, 'D': 1, 'OY': 10, 'OX': 10},
            'equation': 'O[b][g][d][oy][ox]=X[g][oy][ox]+Y[g][oy][ox]',
            'operand_source': {'X': [22], 'Y': [13]},
            'constant_operands': [],
            'operand_precision': {'O': 32, 'O_final': 8, 'X': 8, 'Y': 8},
            'memory_operand_links': {'O': 'O', 'X': 'I1', 'Y': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_Add_33_47'},
    24: {   'operator_type': 'Mul',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 512, 'D': 1, 'OY': 10, 'OX': 10},
            'equation': 'O[b][g][d][oy][ox]=W[g]*I[g][oy][ox]',
            'operand_source': {'I': [23], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'multiply_48'},
    25: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 512, 'D': 1, 'OY': 10, 'OX': 10},
            'equation': 'O[b][g][d][oy][ox]=W[g]+I[g][oy][ox]',
            'operand_source': {'I': [24], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_BatchNormalization_34_PART_0_49'},
    26: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 256, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [25]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_35_PART_0_50_fuse_bias_add_ConvTranspose_35_51'},
    27: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 256, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 2, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [25]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_35_PART_0_50_fuse_bias_add_ConvTranspose_35_51'},
    28: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 256, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 1, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [25]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_35_PART_0_50_fuse_bias_add_ConvTranspose_35_51'},
    29: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 256, 'C': 512, 'OY': 10, 'OX': 10, 'FY': 1, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [25]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_35_PART_0_50_fuse_bias_add_ConvTranspose_35_51'},
    30: {   'operator_type': 'Contract',
            'equation': 'contract',
            'loop_dim_size': {'B': 1, 'G': 256, 'OY': 20, 'OX': 20},
            'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
            'operand_source': {'X0': [29], 'X1': [28], 'X2': [27], 'X3': [26]},
            'constant_operands': [],
            'core_allocation': 1,
            'memory_operand_links': {'O': 'O', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_35_PART_0_50_fuse_bias_add_ConvTranspose_35_51'},
    31: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 256, 'D': 1, 'OY': 20, 'OX': 20},
            'equation': 'O[b][g][d][oy][ox]=X[g][oy][ox]+Y[g][oy][ox]',
            'operand_source': {'X': [30], 'Y': [9]},
            'constant_operands': [],
            'operand_precision': {'O': 32, 'O_final': 8, 'X': 8, 'Y': 8},
            'memory_operand_links': {'O': 'O', 'X': 'I1', 'Y': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_Add_37_53'},
    32: {   'operator_type': 'Mul',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 256, 'D': 1, 'OY': 20, 'OX': 20},
            'equation': 'O[b][g][d][oy][ox]=W[g]*I[g][oy][ox]',
            'operand_source': {'I': [31], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'multiply_54'},
    33: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 256, 'D': 1, 'OY': 20, 'OX': 20},
            'equation': 'O[b][g][d][oy][ox]=W[g]+I[g][oy][ox]',
            'operand_source': {'I': [32], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_BatchNormalization_38_PART_0_55'},
    34: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 128, 'C': 256, 'OY': 20, 'OX': 20, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [33]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_39_PART_0_56_fuse_bias_add_ConvTranspose_39_57'},
    35: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 128, 'C': 256, 'OY': 20, 'OX': 20, 'FY': 2, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [33]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_39_PART_0_56_fuse_bias_add_ConvTranspose_39_57'},
    36: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 128, 'C': 256, 'OY': 20, 'OX': 20, 'FY': 1, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [33]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_39_PART_0_56_fuse_bias_add_ConvTranspose_39_57'},
    37: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 128, 'C': 256, 'OY': 20, 'OX': 20, 'FY': 1, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [33]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_39_PART_0_56_fuse_bias_add_ConvTranspose_39_57'},
    38: {   'operator_type': 'Contract',
            'equation': 'contract',
            'loop_dim_size': {'B': 1, 'G': 128, 'OY': 40, 'OX': 40},
            'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
            'operand_source': {'X0': [37], 'X1': [36], 'X2': [35], 'X3': [34]},
            'constant_operands': [],
            'core_allocation': 1,
            'memory_operand_links': {'O': 'O', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_39_PART_0_56_fuse_bias_add_ConvTranspose_39_57'},
    39: {   'operator_type': 'Mul',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 128, 'D': 1, 'OY': 40, 'OX': 40},
            'equation': 'O[b][g][d][oy][ox]=W[g]*I[g][oy][ox]',
            'operand_source': {'I': [38], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'multiply_59'},
    40: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 128, 'D': 1, 'OY': 40, 'OX': 40},
            'equation': 'O[b][g][d][oy][ox]=W[g]+I[g][oy][ox]',
            'operand_source': {'I': [39], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_BatchNormalization_41_PART_0_60'},
    41: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 64, 'C': 128, 'OY': 40, 'OX': 40, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [40]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_42_PART_0_61_fuse_bias_add_ConvTranspose_42_62'},
    42: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 64, 'C': 128, 'OY': 40, 'OX': 40, 'FY': 2, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [40]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_42_PART_0_61_fuse_bias_add_ConvTranspose_42_62'},
    43: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 64, 'C': 128, 'OY': 40, 'OX': 40, 'FY': 1, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [40]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_42_PART_0_61_fuse_bias_add_ConvTranspose_42_62'},
    44: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 64, 'C': 128, 'OY': 40, 'OX': 40, 'FY': 1, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [40]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_42_PART_0_61_fuse_bias_add_ConvTranspose_42_62'},
    45: {   'operator_type': 'Contract',
            'equation': 'contract',
            'loop_dim_size': {'B': 1, 'G': 64, 'OY': 80, 'OX': 80},
            'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
            'operand_source': {'X0': [44], 'X1': [43], 'X2': [42], 'X3': [41]},
            'constant_operands': [],
            'core_allocation': 1,
            'memory_operand_links': {'O': 'O', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_42_PART_0_61_fuse_bias_add_ConvTranspose_42_62'},
    46: {   'operator_type': 'Mul',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 64, 'D': 1, 'OY': 80, 'OX': 80},
            'equation': 'O[b][g][d][oy][ox]=W[g]*I[g][oy][ox]',
            'operand_source': {'I': [45], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'multiply_64'},
    47: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 64, 'D': 1, 'OY': 80, 'OX': 80},
            'equation': 'O[b][g][d][oy][ox]=W[g]+I[g][oy][ox]',
            'operand_source': {'I': [46], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_BatchNormalization_44_PART_0_65'},
    48: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 32, 'C': 64, 'OY': 80, 'OX': 80, 'FY': 2, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [47]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_45_PART_0_66_fuse_bias_add_ConvTranspose_45_67'},
    49: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fy]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 32, 'C': 64, 'OY': 80, 'OX': 80, 'FY': 2, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [47]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_45_PART_0_66_fuse_bias_add_ConvTranspose_45_67'},
    50: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c][fx]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 32, 'C': 64, 'OY': 80, 'OX': 80, 'FY': 1, 'FX': 2},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [47]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_45_PART_0_66_fuse_bias_add_ConvTranspose_45_67'},
    51: {   'operator_type': 'Conv_from_deconv',
            'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 32, 'C': 64, 'OY': 80, 'OX': 80, 'FY': 1, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [47]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_45_PART_0_66_fuse_bias_add_ConvTranspose_45_67'},
    52: {   'operator_type': 'Contract',
            'equation': 'contract',
            'loop_dim_size': {'B': 1, 'G': 32, 'OY': 160, 'OX': 160},
            'operand_precision': {'O': 8, 'O_final': 8, 'I': 8},
            'operand_source': {'X0': [51], 'X1': [50], 'X2': [49], 'X3': [48]},
            'constant_operands': [],
            'core_allocation': 1,
            'memory_operand_links': {'O': 'O', 'I': 'I1'},
            'name': 'conv2d_transpose_ConvTranspose_45_PART_0_66_fuse_bias_add_ConvTranspose_45_67'},
    53: {   'operator_type': 'Mul',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 32, 'D': 1, 'OY': 160, 'OX': 160},
            'equation': 'O[b][g][d][oy][ox]=W[g]*I[g][oy][ox]',
            'operand_source': {'I': [52], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'multiply_69'},
    54: {   'operator_type': 'Add',
            'equation_relations': [],
            'loop_dim_size': {'B': 1, 'G': 32, 'D': 1, 'OY': 160, 'OX': 160},
            'equation': 'O[b][g][d][oy][ox]=W[g]+I[g][oy][ox]',
            'operand_source': {'I': [53], 'W': []},
            'constant_operands': ['W'],
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('G', 8)},
            'name': 'add_BatchNormalization_47_PART_0_70'},
    55: {   'operator_type': 'Conv',
            'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][ix][iy]',
            'equation_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
            'loop_dim_size': {'B': 1, 'K': 20, 'C': 32, 'OY': 160, 'OX': 160, 'FY': 1, 'FX': 1},
            'operand_precision': {'O': 32, 'O_final': 8, 'W': 8, 'I': 8},
            'operand_source': {'W': [], 'I': [54]},
            'constant_operands': ['W'],
            'core_allocation': 1,
            'spatial_mapping': {'D1': ('K', 8), 'D2': ('C', 32), 'D3': ('OX', 2), 'D4': ('OY', 2)},
            'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
            'name': 'conv2d_Conv_48_PART_0_71_fuse_bias_add_output@@Conv_48_72'}}
