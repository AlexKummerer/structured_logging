"""
Example: Celery integration with structured logging

This example demonstrates how to use structured logging with Celery
for distributed task processing with comprehensive logging.
"""

from celery import Celery
from celery.schedules import crontab

from structured_logging.integrations.celery import (
    CeleryLoggingConfig,
    StructuredLoggingTask,
    get_task_logger,
    log_task,
    log_task_chain,
    setup_celery_logging,
)

# Create Celery app
app = Celery("myapp", broker="redis://localhost:6379")

# Configure Celery logging
config = CeleryLoggingConfig(
    log_task_arguments=True,  # Log task arguments
    log_task_result=True,  # Log task results
    log_worker_events=True,  # Log worker lifecycle
    include_task_runtime=True,  # Include execution time
    propagate_correlation_id=True,  # Propagate trace IDs
)

# Setup structured logging for Celery
setup_celery_logging(app, config, logger_name="myapp.celery")


# Example 1: Basic task with automatic logging
@app.task
def process_order(order_id: str, amount: float) -> dict:
    """Process an order - automatically logged"""
    logger = get_task_logger()

    logger.info(f"Processing order {order_id}")

    # Simulate processing
    import time

    time.sleep(2)

    result = {
        "order_id": order_id,
        "amount": amount,
        "status": "completed",
        "processed_at": time.time(),
    }

    logger.info("Order processed successfully", extra={"result": result})
    return result


# Example 2: Task with custom logging decorator
@app.task
@log_task(operation="data_export", customer="acme")
def export_data(dataset_id: str, format: str = "json") -> str:
    """Export data with additional logging context"""
    logger = get_task_logger()

    logger.info(f"Starting export of dataset {dataset_id}")

    # Simulate export
    import time

    time.sleep(3)

    export_path = f"/exports/{dataset_id}.{format}"
    logger.info(f"Data exported to {export_path}")

    return export_path


# Example 3: Task with retry handling
@app.task(bind=True, max_retries=3)
def fetch_external_data(self, url: str) -> dict:
    """Fetch data with retry logic - retries are logged automatically"""
    logger = get_task_logger()

    try:
        logger.info(f"Fetching data from {url}")

        # Simulate API call that might fail
        import random

        if random.random() > 0.7:  # 30% chance of failure
            raise ConnectionError("API temporarily unavailable")

        return {"data": "success", "url": url}

    except ConnectionError as exc:
        logger.warning(
            f"Connection failed, retrying... (attempt {self.request.retries})"
        )

        # Exponential backoff
        raise self.retry(exc=exc, countdown=2**self.request.retries)


# Example 4: Task chain with correlation ID
def create_processing_chain(file_id: str, correlation_id: str):
    """Create a chain of tasks with correlation ID propagation"""

    # Log the chain creation
    log_task_chain(
        chain_id=correlation_id,
        tasks=["download_file", "process_file", "upload_results"],
    )

    # Create chain with correlation ID in headers
    chain = (
        download_file.si(file_id).set(headers={"correlation_id": correlation_id})
        | process_file.si().set(headers={"correlation_id": correlation_id})
        | upload_results.si().set(headers={"correlation_id": correlation_id})
    )

    return chain


@app.task
def download_file(file_id: str) -> str:
    """Download file - correlation ID is automatically included in logs"""
    logger = get_task_logger()
    logger.info(f"Downloading file {file_id}")

    # Simulate download
    import time

    time.sleep(1)

    local_path = f"/tmp/{file_id}"
    logger.info(f"File downloaded to {local_path}")
    return local_path


@app.task
def process_file(local_path: str) -> dict:
    """Process file - correlation ID is propagated"""
    logger = get_task_logger()
    logger.info(f"Processing file at {local_path}")

    # Simulate processing
    import time

    time.sleep(2)

    result = {"path": local_path, "lines": 1000, "size": 50000}

    logger.info("File processed", extra={"stats": result})
    return result


@app.task
def upload_results(processing_result: dict) -> str:
    """Upload results - correlation ID maintained throughout chain"""
    logger = get_task_logger()
    logger.info("Uploading results", extra={"result_summary": processing_result})

    # Simulate upload
    import time

    time.sleep(1)

    upload_url = "s3://bucket/results/output.json"
    logger.info(f"Results uploaded to {upload_url}")
    return upload_url


# Example 5: Periodic task with beat scheduler
app.conf.beat_schedule = {
    "cleanup-old-data": {
        "task": "myapp.cleanup_old_data",
        "schedule": crontab(hour=2, minute=0),  # Run at 2 AM daily
    },
}


@app.task
def cleanup_old_data():
    """Periodic cleanup task - beat events are logged"""
    logger = get_task_logger()

    logger.info("Starting daily cleanup")

    # Simulate cleanup
    import time

    time.sleep(5)

    deleted_count = 42
    logger.info(f"Cleanup completed, deleted {deleted_count} old records")

    return {"deleted": deleted_count}


# Example 6: Task with structured error handling
@app.task
def risky_operation(operation_type: str) -> dict:
    """Task that might fail with different error types"""
    logger = get_task_logger()

    logger.info(f"Starting risky operation: {operation_type}")

    try:
        if operation_type == "timeout":
            import time

            time.sleep(60)  # Will timeout
        elif operation_type == "error":
            raise ValueError("Invalid operation parameters")
        elif operation_type == "crash":
            raise SystemError("Critical system error")
        else:
            return {"status": "success", "operation": operation_type}

    except ValueError as e:
        # Business logic error - log and return error result
        logger.warning(f"Operation failed: {e}", extra={"error_type": "validation"})
        return {"status": "failed", "error": str(e)}

    except SystemError as e:
        # System error - log and re-raise
        logger.error(f"Critical error: {e}", extra={"error_type": "system"})
        raise


# Example 7: Running tasks programmatically
if __name__ == "__main__":
    # Example of calling tasks with structured logging

    # Simple task execution
    result = process_order.delay("ORDER-123", 99.99)
    print(f"Task ID: {result.id}")

    # Task with correlation ID
    correlation_id = "TRACE-456"
    result = export_data.apply_async(
        args=["DATASET-789"],
        kwargs={"format": "csv"},
        headers={"correlation_id": correlation_id},
    )

    # Create and execute a chain
    chain = create_processing_chain("FILE-001", "CHAIN-789")
    chain_result = chain.apply_async()

    # Monitor task execution
    while not result.ready():
        print(f"Task state: {result.state}")
        import time

        time.sleep(1)

    print(f"Final result: {result.get()}")


# Example 8: Custom task class with enhanced logging
class DataProcessingTask(StructuredLoggingTask):
    """Custom task class with data processing specific logging"""

    def on_success(self, retval, task_id, args, kwargs):
        """Log successful completion with metrics"""
        logger = get_task_logger()

        # Extract metrics from result
        if isinstance(retval, dict) and "metrics" in retval:
            logger.info(
                "Task completed with metrics",
                extra={
                    "task_id": task_id,
                    "metrics": retval["metrics"],
                    "status": "success",
                },
            )

        return super().on_success(retval, task_id, args, kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log failure with detailed context"""
        logger = get_task_logger()

        logger.error(
            "Task failed with exception",
            extra={
                "task_id": task_id,
                "args": args[:2] if len(args) > 2 else args,  # Limit logged args
                "exception_type": type(exc).__name__,
                "error_details": str(exc),
            },
        )

        return super().on_failure(exc, task_id, args, kwargs, einfo)


# Use custom task class
@app.task(base=DataProcessingTask)
def analyze_dataset(dataset_id: str, algorithm: str) -> dict:
    """Analyze dataset with custom logging behavior"""
    logger = get_task_logger()

    logger.info(f"Analyzing dataset {dataset_id} with {algorithm}")

    # Simulate analysis
    import random
    import time

    time.sleep(random.uniform(1, 5))

    metrics = {
        "rows_processed": random.randint(1000, 10000),
        "anomalies_found": random.randint(0, 100),
        "processing_time": random.uniform(1.0, 5.0),
    }

    return {"dataset_id": dataset_id, "algorithm": algorithm, "metrics": metrics}
