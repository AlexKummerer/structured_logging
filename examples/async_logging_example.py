#!/usr/bin/env python3
"""
Example demonstrating async structured logging capabilities
"""

import asyncio

from structured_logging import (
    AsyncLoggerConfig,
    LoggerConfig,
    alog_with_context,
    async_request_context,
    get_async_logger,
    shutdown_all_async_loggers,
)


async def simulate_payment_processing():
    """Simulate async payment processing with structured logging"""

    # Configure async logger for high performance
    async_config = AsyncLoggerConfig(batch_size=10, flush_interval=0.5, queue_size=1000)

    logger_config = LoggerConfig(
        formatter_type="json",
        include_timestamp=True,
        include_request_id=True,
        include_user_context=True,
    )

    # Create async logger
    payment_logger = get_async_logger("payment_service", logger_config, async_config)

    async with async_request_context(
        user_id="user_12345",
        tenant_id="merchant_abc",
        service="payment_api",
        version="2.1.0",
    ):
        await payment_logger.ainfo("Payment processing started")

        # Simulate validation step
        await asyncio.sleep(0.01)  # Simulate async work
        await alog_with_context(
            payment_logger,
            "debug",
            "Payment validation completed",
            payment_method="credit_card",
            amount=99.99,
            currency="EUR",
        )

        # Simulate external API call
        await asyncio.sleep(0.02)  # Simulate async API call
        await payment_logger.ainfo("External payment gateway contacted")

        # Simulate processing
        await asyncio.sleep(0.01)
        await alog_with_context(
            payment_logger,
            "info",
            "Payment processed successfully",
            transaction_id="txn_67890",
            processing_time_ms=40,
            gateway_response="approved",
        )

    # Ensure all logs are processed
    await payment_logger.flush()
    return payment_logger


async def simulate_concurrent_users():
    """Simulate multiple concurrent users making payments"""

    logger = get_async_logger("concurrent_demo")

    async def process_user_payment(user_id: int):
        async with async_request_context(user_id=f"user_{user_id}"):
            await logger.ainfo(f"User {user_id} payment started")
            await asyncio.sleep(0.01)  # Simulate processing
            await alog_with_context(
                logger,
                "info",
                f"User {user_id} payment completed",
                amount=user_id * 10.0,
                status="success",
            )

    # Process 5 users concurrently
    await asyncio.gather(*[process_user_payment(i) for i in range(1, 6)])

    await logger.flush()
    return logger


async def demonstrate_error_handling():
    """Demonstrate async error handling and logging"""

    logger = get_async_logger("error_demo")

    async with async_request_context(user_id="error_user", operation="risky_operation"):
        try:
            await logger.ainfo("Starting risky operation")

            # Simulate an error
            await asyncio.sleep(0.005)
            raise ValueError("Simulated payment failure")

        except ValueError as e:
            await alog_with_context(
                logger,
                "error",
                "Operation failed",
                error_type=type(e).__name__,
                error_message=str(e),
                recovery_action="retry_scheduled",
            )

            # Log recovery attempt
            await logger.awarning("Scheduling retry for failed operation")

    await logger.flush()
    return logger


async def main():
    """Main async demo function"""
    print("🚀 Async Structured Logging Demo")
    print("=" * 40)

    # Demo 1: Payment processing
    print("\n1. Payment Processing Demo:")
    payment_logger = await simulate_payment_processing()

    # Demo 2: Concurrent users
    print("\n2. Concurrent Users Demo:")
    concurrent_logger = await simulate_concurrent_users()

    # Demo 3: Error handling
    print("\n3. Error Handling Demo:")
    error_logger = await demonstrate_error_handling()

    # Cleanup all async loggers
    print("\n4. Graceful Shutdown:")
    await shutdown_all_async_loggers(timeout=2.0)
    print("✅ All async loggers shut down gracefully")


if __name__ == "__main__":
    asyncio.run(main())
