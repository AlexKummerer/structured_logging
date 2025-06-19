#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix

from structured_logging.serializers import SerializationConfig, serialize_for_logging

# Create sparse matrix
data = np.array([1, 2, 3, 4])
row = np.array([0, 0, 1, 2])
col = np.array([0, 2, 1, 0])
sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))

print(f"Matrix shape: {sparse_matrix.shape}")
print(f"Matrix nnz: {sparse_matrix.nnz}")
print(f"Matrix size: {sparse_matrix.size}")
print(f"Expected density: {sparse_matrix.nnz / sparse_matrix.size}")
print("Matrix:")
print(sparse_matrix.toarray())

config = SerializationConfig(numpy_array_max_size=100)
result = serialize_for_logging(sparse_matrix, config)
print("\nSerialization result:")
print(result)
