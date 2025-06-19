#!/usr/bin/env python3
"""
Lazy Serialization Examples

Demonstrates the new lazy serialization features in Version 0.6.0:
- Deferred serialization for massive performance gains
- Smart threshold-based activation
- Memory efficiency for large data structures
- Performance optimization for filtered logs
- Conditional serialization strategies
"""

import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from structured_logging import (
    LazyDict,
    LazySerializable,
    LazySerializationManager,
    SerializationConfig,
    create_lazy_serializable,
    get_lazy_serialization_stats,
    get_logger,
    log_with_context,
    request_context,
    reset_lazy_serialization_stats,
    should_use_lazy_serialization,
)


@dataclass
class LargeDataStructure:
    """Example data structure that's expensive to serialize"""

    id: str
    timestamp: datetime
    data: dict
    metadata: dict


def example_basic_lazy_serialization():
    """Example: Basic lazy serialization concepts"""
    print("🚀 Basic Lazy Serialization Example")
    print("=" * 50)

    # Create expensive data to serialize
    expensive_data = {
        f"record_{i}": {
            "id": str(uuid4()),
            "timestamp": datetime.now(),
            "data": "x" * 1000,  # Large string
            "nested": {"deep": {"value": f"item_{i}"}},
        }
        for i in range(100)  # 100 records = expensive to serialize
    }

    print(f"📊 Created expensive data: {len(expensive_data)} records")

    # Create lazy wrapper - this is very fast
    start_time = time.perf_counter()
    lazy_obj = create_lazy_serializable(expensive_data)
    lazy_creation_time = time.perf_counter() - start_time

    print(f"⚡ Lazy wrapper created in: {lazy_creation_time*1000:.3f}ms")
    print(f"🔍 Is serialized yet? {lazy_obj.is_serialized()}")

    # Force serialization - this is expensive but only happens when needed
    start_time = time.perf_counter()
    serialized_data = lazy_obj.force_serialize()
    serialization_time = time.perf_counter() - start_time

    print(f"⏱️  Serialization took: {serialization_time*1000:.1f}ms")
    print(f"✅ Is serialized now? {lazy_obj.is_serialized()}")
    print(f"📈 Performance ratio: {serialization_time/lazy_creation_time:.0f}x slower")
    print()

    # Demonstrate the benefit: if log was filtered, serialization never happens
    filtered_lazy = create_lazy_serializable(expensive_data)
    print("🎯 Simulating filtered log (serialization never needed):")
    print(f"   - Lazy object created: {not filtered_lazy.is_serialized()}")
    print("   - Log filtered out, object garbage collected")
    print("   - Zero serialization cost! 🎉")
    print()


def example_threshold_based_activation():
    """Example: Threshold-based lazy serialization activation"""
    print("📏 Threshold-Based Activation Example")
    print("=" * 50)

    # Configure different thresholds
    small_threshold_config = SerializationConfig(
        lazy_threshold_items=5,  # Use lazy for >5 items
        lazy_threshold_bytes=100,  # Use lazy for >100 bytes
    )

    large_threshold_config = SerializationConfig(
        lazy_threshold_items=50,  # Use lazy for >50 items
        lazy_threshold_bytes=5000,  # Use lazy for >5000 bytes
    )

    test_data = [
        ("Small dict", {"a": 1, "b": 2, "c": 3}),
        ("Medium dict", {f"key_{i}": f"value_{i}" for i in range(10)}),
        ("Large dict", {f"item_{i}": f"data_{i}" * 10 for i in range(20)}),
        ("Small string", "short text"),
        ("Large string", "x" * 2000),
    ]

    print("🧮 Testing with SMALL thresholds (lazy more often):")
    for name, data in test_data:
        should_use = should_use_lazy_serialization(data, small_threshold_config)
        print(f"   {name:12}: {'✅ LAZY' if should_use else '❌ IMMEDIATE'}")

    print("\n🧮 Testing with LARGE thresholds (lazy less often):")
    for name, data in test_data:
        should_use = should_use_lazy_serialization(data, large_threshold_config)
        print(f"   {name:12}: {'✅ LAZY' if should_use else '❌ IMMEDIATE'}")

    print()


def example_performance_comparison():
    """Example: Performance comparison with real logging"""
    print("⚡ Performance Comparison Example")
    print("=" * 50)

    # Create realistic complex data
    complex_records = []
    for i in range(50):
        record = LargeDataStructure(
            id=str(uuid4()),
            timestamp=datetime.now(),
            data={
                "user_id": str(uuid4()),
                "session_data": {"actions": [f"action_{j}" for j in range(20)]},
                "payload": "x" * 500,  # 500 char payload
            },
            metadata={
                "created_by": "system",
                "version": "1.0",
                "tags": [f"tag_{k}" for k in range(10)],
            },
        )
        complex_records.append(record)

    print(f"📊 Created {len(complex_records)} complex records")

    # Test with immediate serialization
    config_immediate = SerializationConfig(enable_lazy_serialization=False)
    logger = get_logger("perf_immediate")

    start_time = time.perf_counter()
    with request_context(test_type="immediate"):
        for i, record in enumerate(complex_records):
            log_with_context(
                logger,
                "info",
                f"Processing record {i}",
                record_data=record,
                processing_time=time.perf_counter(),
            )
    immediate_time = time.perf_counter() - start_time

    # Test with lazy serialization
    config_lazy = SerializationConfig(enable_lazy_serialization=True)
    logger_lazy = get_logger("perf_lazy")

    # Override formatter config
    from structured_logging.formatter import StructuredFormatter

    for handler in logger_lazy.handlers:
        if isinstance(handler.formatter, StructuredFormatter):
            handler.formatter.serialization_config = config_lazy

    start_time = time.perf_counter()
    with request_context(test_type="lazy"):
        for i, record in enumerate(complex_records):
            log_with_context(
                logger_lazy,
                "info",
                f"Processing record {i}",
                record_data=record,
                processing_time=time.perf_counter(),
            )
    lazy_time = time.perf_counter() - start_time

    # Results
    performance_improvement = immediate_time / lazy_time
    print("\n📈 Performance Results:")
    print(f"   Immediate serialization: {immediate_time:.3f}s")
    print(f"   Lazy serialization:      {lazy_time:.3f}s")
    print(f"   🚀 Performance gain:      {performance_improvement:.1f}x faster")

    if performance_improvement > 5:
        print("   🎉 Excellent! Lazy serialization provides major speedup")
    elif performance_improvement > 2:
        print("   ✅ Good! Significant performance improvement")
    else:
        print("   ℹ️  Modest improvement (may vary by data complexity)")

    print()


def example_memory_efficiency():
    """Example: Memory efficiency with lazy serialization"""
    print("💾 Memory Efficiency Example")
    print("=" * 50)

    # Create many large objects
    num_objects = 1000
    large_objects = []

    print(f"📦 Creating {num_objects} large objects...")

    start_time = time.perf_counter()
    for i in range(num_objects):
        # Each object contains substantial data
        obj_data = {
            "id": i,
            "large_payload": "x" * 1000,  # 1KB per object
            "metadata": {
                "created": datetime.now(),
                "uuid": str(uuid4()),
                "nested": {"deep": {"data": list(range(50))}},
            },
        }

        # Wrap in lazy serialization
        lazy_obj = create_lazy_serializable(obj_data)
        large_objects.append(lazy_obj)

    creation_time = time.perf_counter() - start_time
    print(f"⚡ Created {num_objects} lazy objects in {creation_time:.3f}s")

    # Check how many are actually serialized
    serialized_count = sum(1 for obj in large_objects if obj.is_serialized())
    print(f"🎯 Objects serialized so far: {serialized_count}/{num_objects}")

    # Simulate processing only some objects (common in real scenarios)
    print("🔄 Processing only first 10% of objects...")

    start_time = time.perf_counter()
    processed_objects = []
    for i in range(num_objects // 10):  # Only process 10%
        serialized = large_objects[i].force_serialize()
        processed_objects.append(serialized)
    processing_time = time.perf_counter() - start_time

    # Final stats
    serialized_count = sum(1 for obj in large_objects if obj.is_serialized())
    efficiency = ((num_objects - serialized_count) / num_objects) * 100

    print(f"⏱️  Processing time: {processing_time:.3f}s")
    print("📊 Final serialization stats:")
    print(f"   - Total objects: {num_objects}")
    print(f"   - Actually serialized: {serialized_count}")
    print(f"   - Avoided serialization: {num_objects - serialized_count}")
    print(f"   - 💰 Efficiency gained: {efficiency:.1f}% work avoided")
    print()


def example_conditional_serialization():
    """Example: Conditional serialization based on log levels"""
    print("🎛️ Conditional Serialization Example")
    print("=" * 50)

    # Create different configs for different log levels
    debug_config = SerializationConfig(
        enable_lazy_serialization=True,
        lazy_threshold_items=1,  # Very aggressive - lazy for almost everything
        force_lazy_for_detection=True,
    )

    production_config = SerializationConfig(
        enable_lazy_serialization=True,
        lazy_threshold_items=20,  # Less aggressive - only for large objects
        lazy_threshold_bytes=2000,
        force_lazy_for_detection=False,
    )

    # Sample data of varying complexity
    test_scenarios = [
        ("Simple error", {"error": "Not found", "code": 404}),
        (
            "User action",
            {"user_id": str(uuid4()), "action": "login", "timestamp": datetime.now()},
        ),
        (
            "Complex transaction",
            {
                f"item_{i}": {
                    "id": str(uuid4()),
                    "data": "x" * 100,
                    "metadata": {"type": "purchase", "amount": Decimal(f"{i}.99")},
                }
                for i in range(25)  # 25 items = complex
            },
        ),
        (
            "Debug dump",
            {f"debug_info_{i}": f"detail_{i}" * 50 for i in range(100)},
        ),  # Very large
    ]

    print("🧪 Testing serialization decisions by scenario:")
    print("\nDEBUG environment (aggressive lazy):")
    for name, data in test_scenarios:
        should_lazy = should_use_lazy_serialization(data, debug_config)
        print(f"   {name:20}: {'🚀 LAZY' if should_lazy else '⚡ IMMEDIATE'}")

    print("\nPRODUCTION environment (conservative lazy):")
    for name, data in test_scenarios:
        should_lazy = should_use_lazy_serialization(data, production_config)
        print(f"   {name:20}: {'🚀 LAZY' if should_lazy else '⚡ IMMEDIATE'}")

    print()


def example_lazy_serialization_statistics():
    """Example: Monitoring lazy serialization statistics"""
    print("📊 Lazy Serialization Statistics Example")
    print("=" * 50)

    # Reset stats for clean slate
    reset_lazy_serialization_stats()

    # Simulate various operations
    print("🔄 Simulating application workload...")

    # Create various types of data
    small_data = [{"id": i, "value": f"small_{i}"} for i in range(20)]
    medium_data = [{f"field_{j}": f"data_{j}" for j in range(15)} for i in range(10)]
    large_data = [{f"item_{k}": "x" * 100 for k in range(30)} for i in range(5)]

    # Process with lazy serialization
    operations = [
        ("Small dataset", small_data),
        ("Medium dataset", medium_data),
        ("Large dataset", large_data),
    ]

    for name, dataset in operations:
        print(f"   Processing {name}...")
        for item in dataset:
            lazy_obj = create_lazy_serializable(item)

            # Simulate random access pattern (some serialized, some not)
            if hash(str(item)) % 3 == 0:  # Random 1/3 get serialized
                lazy_obj.force_serialize()

    # Get final statistics
    stats = get_lazy_serialization_stats()

    print("\n📈 Final Statistics:")
    print(f"   Objects created:     {stats['objects_created']:,}")
    print(f"   Objects serialized:  {stats['objects_serialized']:,}")
    print(f"   Objects skipped:     {stats['objects_skipped']:,}")
    print(f"   Efficiency:          {stats['efficiency_percent']:.1f}% work avoided")

    if stats["efficiency_percent"] > 60:
        print("   🎉 Excellent efficiency! Lazy serialization is very beneficial")
    elif stats["efficiency_percent"] > 30:
        print("   ✅ Good efficiency! Worthwhile performance gains")
    else:
        print("   ℹ️  Some efficiency gained, monitor usage patterns")

    print()


def example_advanced_lazy_patterns():
    """Example: Advanced lazy serialization patterns"""
    print("🎓 Advanced Lazy Patterns Example")
    print("=" * 50)

    # Pattern 1: Lazy Dictionary with mixed content
    print("📚 Pattern 1: LazyDict with mixed lazy/immediate content")

    lazy_dict = LazyDict()
    lazy_dict["immediate_data"] = "This is stored normally"
    lazy_dict["expensive_computation"] = create_lazy_serializable(
        {
            "result": [i**2 for i in range(1000)],  # Expensive computation
            "metadata": {"computed_at": datetime.now()},
        }
    )
    lazy_dict["another_lazy"] = create_lazy_serializable(
        {"large_text": "Lorem ipsum " * 1000}
    )

    print("   ✅ LazyDict created with mixed content")
    print(f"   📊 Items: {len(lazy_dict)}")

    # Access immediate data (fast)
    immediate = lazy_dict["immediate_data"]
    print(f"   ⚡ Immediate access: '{immediate[:20]}...'")

    # Serialize all when needed (triggers lazy evaluation)
    print("   🔄 Forcing full serialization...")
    start_time = time.perf_counter()
    fully_serialized = lazy_dict.force_serialize_all()
    serialize_time = time.perf_counter() - start_time
    print(f"   ⏱️  Full serialization: {serialize_time*1000:.1f}ms")

    # Pattern 2: Nested lazy structures
    print("\n🪆 Pattern 2: Nested lazy structures")

    nested_structure = {
        "level1": create_lazy_serializable(
            {"level2": create_lazy_serializable({"level3": {"deep_data": "x" * 1000}})}
        ),
        "parallel_branch": create_lazy_serializable({"data": list(range(500))}),
    }

    print("   ✅ Nested lazy structure created")

    # Pattern 3: Conditional lazy wrapping based on content
    print("\n🤖 Pattern 3: Smart conditional wrapping")

    manager = LazySerializationManager()

    sample_data = [
        "short string",
        "x" * 2000,  # Long string
        {"small": "dict"},
        {f"large_dict_{i}": f"value_{i}" for i in range(50)},  # Large dict
        list(range(5)),  # Small list
        list(range(100)),  # Large list
    ]

    for i, data in enumerate(sample_data):
        wrapped = manager.wrap_if_beneficial(data)
        is_lazy = isinstance(wrapped, LazySerializable)
        data_type = type(data).__name__
        size_info = (
            f"len={len(data)}"
            if hasattr(data, "__len__")
            else f"str_len={len(str(data))}"
        )

        print(
            f"   Item {i} ({data_type}, {size_info}): {'🚀 LAZY' if is_lazy else '⚡ IMMEDIATE'}"
        )

    print()


def main():
    """Run all lazy serialization examples"""
    print("🚀 Structured Logging v0.6.0 - Lazy Serialization Examples")
    print("=" * 70)
    print()

    print("📋 Available Examples:")
    print("1. Basic lazy serialization concepts")
    print("2. Threshold-based activation strategies")
    print("3. Performance comparison with real logging")
    print("4. Memory efficiency demonstrations")
    print("5. Conditional serialization by environment")
    print("6. Statistics and monitoring")
    print("7. Advanced patterns and techniques")
    print()

    try:
        example_basic_lazy_serialization()
        example_threshold_based_activation()
        example_performance_comparison()
        example_memory_efficiency()
        example_conditional_serialization()
        example_lazy_serialization_statistics()
        example_advanced_lazy_patterns()

        print("🎉 All lazy serialization examples completed!")
        print()
        print("💡 Key Benefits of Lazy Serialization:")
        print("  ⚡ Massive performance gains for complex data")
        print("  💾 Reduced memory usage and CPU utilization")
        print("  🎯 Zero cost for filtered/unused log entries")
        print("  📊 Configurable thresholds for optimal performance")
        print("  🧠 Smart activation based on data characteristics")
        print("  📈 Detailed statistics for performance monitoring")
        print("  🔧 Advanced patterns for complex use cases")
        print()
        print("🚀 Your logs are now incredibly efficient and fast!")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
