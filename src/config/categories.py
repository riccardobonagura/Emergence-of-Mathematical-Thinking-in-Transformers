# categories.py — single source of truth for dataset category names.
# Import from here in every file that masks stimuli by category.

from typing import Tuple

# Arithmetic stimuli (contrastive pairs with meaningful sign/parity labels)
MATH_CATS: Tuple[str, ...] = ("CAT-SIGN", "CAT-PARITY")

# Linguistic control stimuli (labels are sentinels: sign=−1, parity=−1)
CTRL_CATS: Tuple[str, ...] = ("CTRL-NEU", "CTRL-NUM")

# All categories in the v5 dataset
ALL_CATS: Tuple[str, ...] = MATH_CATS + CTRL_CATS

# Label fields available for probing (v5: binary only)
PROBE_PROPERTIES: Tuple[str, ...] = ("sign", "parity")

# Sentinel value used in Labels for non-arithmetic stimuli
LABEL_SENTINEL: int = -1
