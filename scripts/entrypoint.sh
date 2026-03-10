#!/bin/bash
set -e

# Docker entrypoint for IDX Trading System

echo "=== IDX Trading System ==="
echo "Starting at $(date)"

# Validate required environment
if [ -z "$DATABASE_URL" ]; then
    echo "WARNING: DATABASE_URL not set, using default SQLite"
fi

# Ensure data directories exist
mkdir -p data/market data/trades data/backtest data/fundamental logs

# Run database migrations if needed
echo "Checking database..."
python -c "
from config.settings import settings
settings.ensure_directories()
print('Directories ready.')
" 2>/dev/null || echo "Settings check skipped."

# Execute the passed command
echo "Executing: $@"
exec "$@"
