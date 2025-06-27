"""
Enhanced JSON encoder with support for complex Python types
"""

import json
from typing import Any, Optional

from .config import SerializationConfig
from .lazy import LazyDict, LazySerializable
from .registry import TypeRegistry


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with support for complex Python types"""

    def __init__(self, config: Optional[SerializationConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SerializationConfig()
        self.type_registry = TypeRegistry()

    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable formats"""

        # Handle lazy serializable objects
        if isinstance(obj, LazySerializable):
            return obj.force_serialize()

        # Handle lazy dictionaries
        if isinstance(obj, LazyDict):
            return obj.force_serialize_all()

        # Get custom serializer
        serializer = self.type_registry.get_serializer(obj)
        if serializer:
            try:
                return serializer(obj, self.config)
            except Exception as e:
                # Fallback to safe representation
                return {
                    "__serialization_error__": str(e),
                    "__type__": type(obj).__name__,
                    "__repr__": repr(obj)[:200],  # Truncate repr
                }

        # Handle collections with size limits
        if isinstance(obj, (list, tuple)):
            if len(obj) > self.config.max_collection_size:
                truncated = list(obj[: self.config.max_collection_size])
                truncated.append(
                    f"... ({len(obj) - self.config.max_collection_size} more items)"
                )
                return truncated
            return list(obj)

        # Handle strings with truncation
        if isinstance(obj, str) and self.config.truncate_strings:
            if len(obj) > self.config.truncate_strings:
                return obj[: self.config.truncate_strings] + "..."

        # Fallback to default behavior
        try:
            return super().default(obj)
        except TypeError:
            # Last resort: safe string representation
            try:
                obj_repr = repr(obj)[:200]
            except Exception as e:
                obj_repr = f"<repr failed: {str(e)[:50]}>"

            return {"__unserializable__": type(obj).__name__, "__repr__": obj_repr}


def enhanced_json_dumps(
    obj: Any, config: Optional[SerializationConfig] = None, **kwargs
) -> str:
    """
    JSON dumps with enhanced serialization support

    Args:
        obj: Object to serialize
        config: Serialization configuration
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation
    """
    encoder = EnhancedJSONEncoder(config=config)
    return json.dumps(obj, cls=EnhancedJSONEncoder, **kwargs)