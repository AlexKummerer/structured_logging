"""
SciPy sparse matrix serialization module
"""

from typing import Any, Dict

try:
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    scipy = None
    HAS_SCIPY = False

from .config import SerializationConfig


def serialize_scipy_sparse_matrix(
    sparse_matrix: Any, config: SerializationConfig
) -> Dict[str, Any]:
    """Serialize SciPy sparse matrices"""
    if not HAS_SCIPY:
        return {"error": "SciPy not available"}

    if not isinstance(sparse_matrix, scipy.sparse.spmatrix):
        return {"error": "Not a SciPy sparse matrix"}

    result = {
        "shape": list(sparse_matrix.shape),
        "dtype": str(sparse_matrix.dtype),
        "format": sparse_matrix.format,
        "nnz": int(sparse_matrix.nnz),
        "density": (
            float(
                sparse_matrix.nnz
                / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
            )
            if (sparse_matrix.shape[0] * sparse_matrix.shape[1]) > 0
            else 0.0
        ),
        "__scipy_type__": "sparse_matrix",
    }

    # Memory information
    result["memory_info"] = {
        "data_bytes": (
            sparse_matrix.data.nbytes if hasattr(sparse_matrix, "data") else 0
        ),
        "total_bytes": (
            sparse_matrix.data.nbytes
            + sparse_matrix.indices.nbytes
            + sparse_matrix.indptr.nbytes
            if hasattr(sparse_matrix, "indices")
            and hasattr(sparse_matrix, "indptr")
            else 0
        ),
    }

    # For very sparse matrices, include some sample data
    if sparse_matrix.nnz <= config.numpy_array_max_size and sparse_matrix.nnz > 0:
        try:
            # Convert to COO format to get coordinates and values
            coo = sparse_matrix.tocoo()
            result["sample_data"] = {
                "coordinates": list(
                    zip(coo.row[:10].tolist(), coo.col[:10].tolist())
                ),
                "values": coo.data[:10].tolist(),
            }
        except Exception as e:
            result["sample_error"] = str(e)

    return result