#!/bin/bash

# Run the update script using Poetry
while true; do
    poetry run python scripts/scheduled_update.py
    # Wait for 1 hour before next update
    sleep 3600
done