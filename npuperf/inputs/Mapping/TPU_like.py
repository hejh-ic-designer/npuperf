mapping = {
    "Conv": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("K", 32),
            "D2": ("C", 32),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },

    "dw_Conv": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 32),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },

    "Add": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 32),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "Y": "I1"},
    },

    "Mul": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 32),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "Y": "I1"},
    },

    "Pool": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 32),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },

    "Fc": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("K", 32),
            "D2": ("C", 32),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },

    "Input": {
        "core_allocation": 1,
        "memory_operand_links": {"O": "I1"},
    },
}
