from NeuRec.util.cython.random_choice import batch_randint_choice

from .configurator import Configurator
from .data_iterator import DataIterator
from .logger import Logger
from .tool import argmax_top_k
from .tool import csr_to_user_dict
from .tool import inner_product
# from util.tool import batch_random_choice
from .tool import l2_loss
from .tool import log_loss
from .tool import pad_sequences
from .tool import randint_choice
from .tool import timer
from .tool import typeassert

# note: prevent from auto clean due to cpython reference problem.
type(batch_randint_choice)

__all__ = [
    "Configurator",
    "DataIterator",
    "randint_choice",
    "csr_to_user_dict",
    "typeassert",
    "argmax_top_k",
    "timer",
    "pad_sequences",
    "inner_product",
    "l2_loss",
    "logger",
    "log_loss",
    "batch_randint_choice",
    "Logger",
]
