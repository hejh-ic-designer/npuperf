mapping = {
    'DLA_name': 'Eyeriss_like',
    'partial_sum_precision': 24,
    'Conv': {
        'core_allocation': 1,
        'spatial_mapping': {
            'D1': ('K', 14),
            'D2': ('C', 12)
        },
        'memory_operand_links': {
            'O': 'O',
            'W': 'I2',
            'I': 'I1'
        }
    },
    'dw_Conv': {
        'core_allocation': 1,
        'spatial_mapping': {
            'D1': ('G', 14)
        },
        'memory_operand_links': {
            'O': 'O',
            'W': 'I2',
            'I': 'I1'
        }
    },
    'Pool': {
        'core_allocation': 1,
        'spatial_mapping': {
            'D1': ('G', 14)
        },
        'memory_operand_links': {
            'O': 'O',
            'W': 'I2',
            'I': 'I1'
        }
    },
    'Fc': {
        'core_allocation': 1,
        'spatial_mapping': {
            'D1': ('K', 14),
            'D2': ('C', 12)
        },
        'memory_operand_links': {
            'O': 'O',
            'W': 'I2',
            'I': 'I1'
        }
    },
    'Add': {
        'core_allocation': 1,
        'spatial_mapping': {
            'D1': ('G', 14)
        },
        'memory_operand_links': {
            'O': 'O',
            'X': 'I1',
            'Y': 'I1'
        }
    },
    'Mul': {
        'core_allocation': 1,
        'spatial_mapping': {
            'D1': ('G', 14)
        },
        'memory_operand_links': {
            'O': 'O',
            'X': 'I1',
            'Y': 'I1'
        }
    },
    'Input': {
        'core_allocation': 1,
        'spatial_mapping': None,
        'memory_operand_links': {
            'O': 'I1'
        }
    }
}
