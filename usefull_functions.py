#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:02:43 2024

@author: moenning
"""

import functools
import numpy as np
from scipy.sparse.linalg import expm

#Creates a decorator so the function is cached but the cache can be cleaned at any time
def resettable_cache(maxsize=128):
    def decorator(func):
        cache = {}
        cache_order = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function's arguments
            key = (args, frozenset(kwargs.items()))
            
            # Return cached result if present
            if key in cache:
                return cache[key]

            # Call the function and store the result in the cache
            result = func(*args, **kwargs)
            
            # Manage the cache size
            if len(cache_order) >= maxsize:
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]

            # Store the result in the cache
            cache[key] = result
            cache_order.append(key)
            
            return result

        # Add a method to clear the cache
        def reset_cache():
            nonlocal cache, cache_order
            cache = {}
            cache_order = []

        # Attach the reset method to the wrapper function
        wrapper.reset_cache = reset_cache
        
        return wrapper

    return decorator


#cachin the expm function
@functools.lru_cache()
def cached_expm(matrix_tuple):
    """Compute the matrix exponential and cache the result."""
    matrix = np.array(matrix_tuple)
    return expm(matrix)

def expm_wrapper(matrix):
    """Wrapper function for expm with caching."""
    
    matrix_tuple = tuple(map(tuple, matrix))
    return cached_expm(matrix_tuple)

    