# Shared experiment configuration

EXPERIMENTS = [
    "solo_lead",
    "lead_preprocessing",
    "lead_augmentation",
    "lead_preprocessing_augmentation",
]

LEAD_CONFIGURATIONS = [
    [(12,)],
    [(12,), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,)],
    [(12,), (0, 1), (11, 6), (10, 8), (2, 7), (2, 9), (6, 8)],
    [(12,), (0, 1, 6), (10, 2, 8), (11, 7, 9), (1, 6, 8)],
]


def stringify_lead_configuration(lead_configuration):
    """Convert a list of tuples into a stable string name."""
    return "-".join(map(str, lead_configuration))
