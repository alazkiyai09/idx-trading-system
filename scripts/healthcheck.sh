#!/bin/bash
# Container health check script

HEALTH_URL="http://localhost:8000/health"

response=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null)

if [ "$response" == "200" ]; then
    exit 0
else
    echo "Health check failed: HTTP $response"
    exit 1
fi
