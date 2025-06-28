"""
Example: SQLAlchemy integration with structured logging

This example demonstrates how to use structured logging with SQLAlchemy
for comprehensive database operation logging.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from structured_logging.integrations.sqlalchemy import (
    DatabaseOperation,
    SQLAlchemyLoggingConfig,
    get_query_logger,
    log_query,
    setup_sqlalchemy_logging,
)

# Create database models
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)

    # Relationship
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(5000))
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship
    author = relationship("User", back_populates="posts")


# Example 1: Basic setup with SQLAlchemy logging
def setup_database_with_logging():
    """Setup database with comprehensive logging"""

    # Create engine (using SQLite for example)
    engine = create_engine(
        "sqlite:///example.db", echo=False  # Disable SQLAlchemy's built-in logging
    )

    # Configure logging
    config = SQLAlchemyLoggingConfig(
        log_query_parameters=True,  # Log query parameters
        slow_query_threshold=0.1,  # Warn on queries > 100ms
        log_connection_pool=True,  # Log pool events
        log_transactions=True,  # Log transaction lifecycle
        log_orm_events=True,  # Log ORM flush/bulk operations
    )

    # Create session factory
    session_factory = sessionmaker(bind=engine)

    # Setup structured logging
    setup_sqlalchemy_logging(engine, config, session_factory)

    # Create tables
    Base.metadata.create_all(engine)

    return engine, session_factory


# Example 2: Query logging with decorator
@log_query(operation="user_search", purpose="authentication")
def find_user_by_email(session, email: str):
    """Find user by email with automatic logging"""
    return session.query(User).filter_by(email=email, is_active=True).first()


@log_query(operation="user_posts", include_count=True)
def get_user_posts(session, user_id: int, limit: int = 10):
    """Get recent posts for a user"""
    posts = (
        session.query(Post)
        .filter_by(author_id=user_id)
        .order_by(Post.created_at.desc())
        .limit(limit)
        .all()
    )
    return posts


# Example 3: Bulk operations with logging
def import_users_batch(session, user_data_list):
    """Import multiple users with bulk operation logging"""
    logger = get_query_logger("user_import")

    with DatabaseOperation("bulk_user_import", count=len(user_data_list)) as op:
        try:
            # Create user objects
            users = []
            for data in user_data_list:
                user = User(
                    username=data["username"],
                    email=data["email"],
                    is_active=data.get("is_active", True),
                )
                users.append(user)

            # Bulk insert
            session.bulk_save_objects(users)
            session.commit()

            # Set operation result
            op.set_result({"imported": len(users), "status": "success"})

            logger.info(f"Successfully imported {len(users)} users")

        except Exception as e:
            session.rollback()
            logger.error(f"User import failed: {e}")
            raise


# Example 4: Complex queries with performance tracking
def analyze_user_activity(session, days_back: int = 30):
    """Analyze user activity with detailed logging"""
    logger = get_query_logger("analytics")

    with DatabaseOperation("user_activity_analysis", days=days_back) as op:
        # Get active users
        cutoff_date = func.datetime("now", f"-{days_back} days")

        # This will be logged with timing
        active_users = (
            session.query(User)
            .join(Post)
            .filter(Post.created_at >= cutoff_date)
            .distinct()
            .count()
        )

        # Get post statistics - will trigger slow query warning if > threshold
        post_stats = session.execute(
            text(
                """
                SELECT
                    u.username,
                    COUNT(p.id) as post_count,
                    MAX(p.created_at) as last_post
                FROM users u
                JOIN posts p ON u.id = p.author_id
                WHERE p.created_at >= datetime('now', :days_back)
                GROUP BY u.id
                ORDER BY post_count DESC
                LIMIT 10
            """
            ),
            {"days_back": f"-{days_back} days"},
        ).fetchall()

        result = {
            "active_users": active_users,
            "top_posters": [
                {"username": row[0], "post_count": row[1], "last_post": row[2]}
                for row in post_stats
            ],
        }

        op.set_result(result)
        logger.info("Activity analysis completed", extra={"result_summary": result})

        return result


# Example 5: Transaction management with logging
def transfer_posts(session, from_user_id: int, to_user_id: int):
    """Transfer all posts from one user to another with transaction logging"""
    logger = get_query_logger("post_transfer")

    with DatabaseOperation(
        "post_ownership_transfer", from_user=from_user_id, to_user=to_user_id
    ) as op:
        # Start transaction (automatically logged)
        try:
            # Get users
            from_user = session.query(User).get(from_user_id)
            to_user = session.query(User).get(to_user_id)

            if not from_user or not to_user:
                raise ValueError("User not found")

            # Count posts to transfer
            post_count = session.query(Post).filter_by(author_id=from_user_id).count()

            # Update all posts (logged as bulk update)
            session.query(Post).filter_by(author_id=from_user_id).update(
                {"author_id": to_user_id, "updated_at": datetime.utcnow()}
            )

            # Commit transaction (automatically logged with duration)
            session.commit()

            op.set_result(
                {
                    "posts_transferred": post_count,
                    "from_user": from_user.username,
                    "to_user": to_user.username,
                }
            )

            logger.info(
                (f"Transferred {post_count} posts from "
                 f"{from_user.username} to {to_user.username}")
            )

        except Exception as e:
            # Rollback (automatically logged)
            session.rollback()
            logger.error(f"Post transfer failed: {e}")
            raise


# Example 6: Connection pool monitoring
def monitor_connection_pool(engine):
    """Monitor database connection pool events"""
    logger = get_query_logger("pool_monitor")

    # Pool events are automatically logged by the integration
    # This function demonstrates additional custom monitoring

    pool = engine.pool
    logger.info(
        "Connection pool status",
        extra={
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
        },
    )


# Example 7: Handling slow queries
def find_inactive_users_with_posts(session):
    """Find inactive users who have posts (potentially slow query)"""
    logger = get_query_logger("maintenance")

    # This query might be slow and will trigger a warning
    with DatabaseOperation("inactive_user_cleanup") as op:
        # Complex join that might be slow
        inactive_with_posts = (
            session.query(User)
            .join(Post)
            .filter(User.is_active.is_(False))
            .group_by(User.id)
            .having(func.count(Post.id) > 0)
            .all()
        )

        result = {
            "inactive_users_with_posts": len(inactive_with_posts),
            "users": [
                {
                    "id": user.id,
                    "username": user.username,
                    "post_count": len(user.posts),
                }
                for user in inactive_with_posts
            ],
        }

        op.set_result(result)

        if inactive_with_posts:
            logger.warning(
                f"Found {len(inactive_with_posts)} inactive users with posts",
                extra={"users": [u.username for u in inactive_with_posts]},
            )

        return result


# Example 8: Using raw SQL with parameter logging
def execute_custom_report(session, min_posts: int = 5):
    """Execute custom SQL report with parameter logging"""
    logger = get_query_logger("reports")

    # Raw SQL will be logged with parameters (safely sanitized)
    query = text(
        """
        SELECT
            u.username,
            u.email,
            COUNT(p.id) as total_posts,
            SUM(CASE WHEN p.created_at > datetime('now', '-7 days')
                THEN 1 ELSE 0 END) as recent_posts
        FROM users u
        LEFT JOIN posts p ON u.id = p.author_id
        WHERE u.is_active = :is_active
        GROUP BY u.id
        HAVING COUNT(p.id) >= :min_posts
        ORDER BY total_posts DESC
    """
    )

    with DatabaseOperation("user_activity_report", min_posts=min_posts) as op:
        result = session.execute(
            query, {"is_active": True, "min_posts": min_posts}
        ).fetchall()

        report_data = [
            {
                "username": row[0],
                "email": row[1],
                "total_posts": row[2],
                "recent_posts": row[3],
            }
            for row in result
        ]

        op.set_result({"user_count": len(report_data)})
        logger.info(f"Generated report for {len(report_data)} active users")

        return report_data


# Example usage
if __name__ == "__main__":
    # Setup database with logging
    engine, session_factory = setup_database_with_logging()

    # Create a session
    session = session_factory()

    try:
        # Create sample data
        print("Creating sample data...")

        # Users
        users = [
            User(username="alice", email="alice@example.com"),
            User(username="bob", email="bob@example.com"),
            User(username="charlie", email="charlie@example.com", is_active=False),
        ]
        session.add_all(users)
        session.commit()

        # Posts
        for user in users[:2]:  # Only active users
            for i in range(3):
                post = Post(
                    title=f"{user.username}'s post {i+1}",
                    content=f"Content from {user.username}",
                    author_id=user.id,
                )
                session.add(post)
        session.commit()

        print("\nTesting various logging scenarios...")

        # Test query logging
        user = find_user_by_email(session, "alice@example.com")
        print(f"Found user: {user.username if user else 'Not found'}")

        # Test bulk import
        new_users = [
            {"username": "david", "email": "david@example.com"},
            {"username": "eve", "email": "eve@example.com"},
        ]
        import_users_batch(session, new_users)

        # Test analytics
        activity = analyze_user_activity(session, 30)
        print(f"Active users: {activity['active_users']}")

        # Test slow query detection
        inactive = find_inactive_users_with_posts(session)
        print(f"Inactive users with posts: {inactive['inactive_users_with_posts']}")

        # Monitor connection pool
        monitor_connection_pool(engine)

        # Generate report
        report = execute_custom_report(session, min_posts=2)
        print(f"Report generated for {len(report)} users")

    finally:
        session.close()
        engine.dispose()

        # Clean up
        import os

        if os.path.exists("example.db"):
            os.remove("example.db")
