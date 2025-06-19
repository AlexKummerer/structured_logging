#!/usr/bin/env python3

import numpy as np

from structured_logging.serializers import SerializationConfig, serialize_for_logging

config = SerializationConfig(numpy_include_metadata=True, numpy_array_precision=3)

# Test different scalar types
test_cases = [
    (np.int32(42), "integer"),
    (np.float64(3.14159), "floating"),
    (np.complex128(1 + 2j), "complex"),
    (np.bool_(True), "boolean"),
    (np.str_("test"), "string"),
]

for scalar, expected_type in test_cases:
    result = serialize_for_logging(scalar, config)
    print(f"\nScalar: {scalar} (type: {type(scalar)})")
    print(f"Expected type: {expected_type}")
    print(f"Result: {result}")
    print(
        f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
    )
    if isinstance(result, dict) and "__numpy_type__" in result:
        print(f"__numpy_type__: {result['__numpy_type__']}")
        print(f"Matches expected: {result['__numpy_type__'] == expected_type}")
