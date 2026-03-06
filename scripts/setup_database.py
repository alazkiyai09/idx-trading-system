#!/usr/bin/env python3
"""
Database setup and initialization script.

Creates database tables and verifies the database connection.

Usage:
    python scripts/setup_database.py
    python scripts/setup_database.py --database sqlite:///data/trading.db
    python scripts/setup_database.py --verify
    python scripts/setup_database.py --reset
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import inspect

from config.logging_config import setup_logging
from core.data.database import Base, DatabaseManager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Setup and manage IDX Trading System database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create all tables
    python scripts/setup_database.py

    # Use custom database
    python scripts/setup_database.py --database sqlite:///custom.db

    # Verify database structure
    python scripts/setup_database.py --verify

    # Reset database (drop and recreate all tables)
    python scripts/setup_database.py --reset

    # Show database info
    python scripts/setup_database.py --info
        """,
    )

    parser.add_argument(
        "--database",
        type=str,
        default="sqlite:///data/trading.db",
        help="Database URL (default: sqlite:///data/trading.db)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify database structure",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables (WARNING: destroys data)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show database information",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def create_tables(db_manager: DatabaseManager) -> None:
    """Create all database tables.

    Args:
        db_manager: Database manager instance.
    """
    print("Creating database tables...")
    db_manager.create_tables()
    print("Tables created successfully.")


def verify_database(db_manager: DatabaseManager) -> bool:
    """Verify database structure.

    Args:
        db_manager: Database manager instance.

    Returns:
        True if verification passes, False otherwise.
    """
    print("\nVerifying database structure...")
    print("-" * 40)

    inspector = inspect(db_manager.engine)
    tables = inspector.get_table_names()

    expected_tables = [
        "price_history",
        "foreign_flow_history",
        "trade_history",
        "open_positions",
        "calibration_surface",
    ]

    all_present = True
    for table in expected_tables:
        if table in tables:
            columns = inspector.get_columns(table)
            print(f"✓ {table}: {len(columns)} columns")
        else:
            print(f"✗ {table}: MISSING")
            all_present = False

    print("-" * 40)

    if all_present:
        print("Verification PASSED: All tables present")
        return True
    else:
        print("Verification FAILED: Some tables missing")
        return False


def reset_database(db_manager: DatabaseManager) -> None:
    """Reset database by dropping and recreating all tables.

    Args:
        db_manager: Database manager instance.
    """
    print("\n" + "=" * 40)
    print("WARNING: This will DESTROY ALL DATA!")
    print("=" * 40)

    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() != "yes":
        print("Reset cancelled.")
        return

    print("\nDropping all tables...")
    Base.metadata.drop_all(db_manager.engine)
    print("Creating all tables...")
    Base.metadata.create_all(db_manager.engine)
    print("Database reset complete.")


def show_database_info(db_manager: DatabaseManager) -> None:
    """Show database information.

    Args:
        db_manager: Database manager instance.
    """
    print("\n" + "=" * 60)
    print("DATABASE INFORMATION")
    print("=" * 60)

    print(f"\nDatabase URL: {db_manager.database_url}")

    inspector = inspect(db_manager.engine)
    tables = inspector.get_table_names()

    print(f"\nTables ({len(tables)}):")
    print("-" * 40)

    total_records = 0

    for table in sorted(tables):
        columns = inspector.get_columns(table)
        session = db_manager.get_session()

        try:
            # Get row count
            from sqlalchemy import text

            result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar() or 0
            total_records += count
            print(f"  {table}:")
            print(f"    - Columns: {len(columns)}")
            print(f"    - Records: {count:,}")
        except Exception as e:
            print(f"  {table}: Error getting info - {e}")
        finally:
            session.close()

    print("-" * 40)
    print(f"Total records: {total_records:,}")

    # Show indexes
    print("\nIndexes:")
    for table in sorted(tables):
        indexes = inspector.get_indexes(table)
        if indexes:
            for idx in indexes:
                print(f"  {table}.{idx['name']}: {idx['column_names']}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    try:
        # Create database manager
        db_manager = DatabaseManager(args.database)

        # Handle different modes
        if args.reset:
            reset_database(db_manager)
        elif args.verify:
            success = verify_database(db_manager)
            return 0 if success else 1
        elif args.info:
            show_database_info(db_manager)
        else:
            # Default: create tables
            create_tables(db_manager)
            print("\nDatabase setup complete.")
            show_database_info(db_manager)

        return 0

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
