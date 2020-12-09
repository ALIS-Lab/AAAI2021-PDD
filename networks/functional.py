"""Functional interface"""

from .parallel_dropout import *

# Activation functions  
def dropout_parallel(fcs, K, M, alpha, beta, training=False, inplace=False):
    return Parallel_Dropout.apply(fcs, K, M, alpha, beta, training, inplace)