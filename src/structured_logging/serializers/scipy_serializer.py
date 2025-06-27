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


def _get_sparse_matrix_metadata(sparse_matrix: Any) -> Dict[str, Any]:
    """Get basic metadata for sparse matrix"""
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    density = float(sparse_matrix.nnz / total_elements) if total_elements > 0 else 0.0
    
    return {
        "shape": list(sparse_matrix.shape),
        "dtype": str(sparse_matrix.dtype),
        "format": sparse_matrix.format,
        "nnz": int(sparse_matrix.nnz),
        "density": density,
        "__scipy_type__": "sparse_matrix",
    }


def _get_sparse_matrix_memory_info(sparse_matrix: Any) -> Dict[str, Any]:
    """Get memory usage information for sparse matrix"""
    data_bytes = sparse_matrix.data.nbytes if hasattr(sparse_matrix, "data") else 0
    
    total_bytes = data_bytes
    if hasattr(sparse_matrix, "indices") and hasattr(sparse_matrix, "indptr"):
        total_bytes += sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes
    
    return {"data_bytes": data_bytes, "total_bytes": total_bytes}


def _add_sparse_matrix_sample(sparse_matrix: Any, config: SerializationConfig, result: Dict[str, Any]) -> None:
    """Add sample data for sparse matrix if appropriate"""
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


def serialize_scipy_sparse_matrix(
    sparse_matrix: Any, config: SerializationConfig
) -> Dict[str, Any]:
    """Serialize SciPy sparse matrices"""
    if not HAS_SCIPY:
        return {"error": "SciPy not available"}

    if not isinstance(sparse_matrix, scipy.sparse.spmatrix):
        return {"error": "Not a SciPy sparse matrix"}

    result = _get_sparse_matrix_metadata(sparse_matrix)
    result["memory_info"] = _get_sparse_matrix_memory_info(sparse_matrix)
    _add_sparse_matrix_sample(sparse_matrix, config, result)

    return result