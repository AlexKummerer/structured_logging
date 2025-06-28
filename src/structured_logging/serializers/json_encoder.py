"""
Enhanced JSON encoder with support for complex Python types
"""

import json
from typing import Any, Dict, Optional

from .config import SerializationConfig
from .lazy import LazyDict, LazySerializable
from .registry import TypeRegistry


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with support for complex Python types"""

    def __init__(self, config: Optional[SerializationConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or SerializationConfig()
        self.type_registry = TypeRegistry()

    def _handle_lazy_objects(self, obj: Any) -> Any:
        """Handle lazy serializable objects"""
        if isinstance(obj, LazySerializable):
            return obj.force_serialize()
        if isinstance(obj, LazyDict):
            return obj.force_serialize_all()
        return None

    def _handle_custom_serializer(self, obj: Any) -> Any:
        """Handle objects with custom serializers"""
        serializer = self.type_registry.get_serializer(obj)
        if not serializer:
            return None
            
        try:
            return serializer(obj, self.config)
        except Exception as e:
            return {
                "__serialization_error__": str(e),
                "__type__": type(obj).__name__,
                "__repr__": repr(obj)[:200],  # Truncate repr
            }

    def _handle_collections(self, obj: Any) -> Any:
        """Handle collections with size limits"""
        if not isinstance(obj, (list, tuple)):
            return None
            
        if len(obj) > self.config.max_collection_size:
            truncated = list(obj[: self.config.max_collection_size])
            truncated.append(
                f"... ({len(obj) - self.config.max_collection_size} more items)"
            )
            return truncated
        return list(obj)

    def _handle_string_truncation(self, obj: Any) -> Any:
        """Handle string truncation"""
        if isinstance(obj, str) and self.config.truncate_strings:
            if len(obj) > self.config.truncate_strings:
                return obj[: self.config.truncate_strings] + "..."
        return None

    def _safe_repr(self, obj: Any) -> Dict[str, str]:
        """Create safe representation for unserializable objects"""
        try:
            obj_repr = repr(obj)[:200]
        except Exception as e:
            obj_repr = f"<repr failed: {str(e)[:50]}>"
        
        return {"__unserializable__": type(obj).__name__, "__repr__": obj_repr}

    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable formats"""
        # Try lazy objects
        result = self._handle_lazy_objects(obj)
        if result is not None:
            return result
        
        # Try custom serializer
        result = self._handle_custom_serializer(obj)
        if result is not None:
            return result
        
        # Try collections
        result = self._handle_collections(obj)
        if result is not None:
            return result
        
        # Try string truncation
        result = self._handle_string_truncation(obj)
        if result is not None:
            return result
        
        # Fallback to default behavior
        try:
            return super().default(obj)
        except TypeError:
            return self._safe_repr(obj)


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