"""
Utility functions for framework integrations
"""

from typing import Dict, Set


def filter_sensitive_headers(
    headers: Dict[str, str], sensitive_headers: Set[str], mask_sensitive: bool = True
) -> Dict[str, str]:
    """Filter out sensitive headers"""
    if not mask_sensitive:
        return headers

    filtered = {}
    for key, value in headers.items():
        if key.lower() in sensitive_headers:
            filtered[key] = "[MASKED]"
        else:
            filtered[key] = value

    return filtered


def filter_sensitive_query_params(
    params: Dict[str, str], sensitive_params: Set[str], mask_sensitive: bool = True
) -> Dict[str, str]:
    """Filter out sensitive query parameters"""
    if not mask_sensitive:
        return params

    filtered = {}
    for key, value in params.items():
        if key.lower() in sensitive_params:
            filtered[key] = "[MASKED]"
        else:
            filtered[key] = value

    return filtered