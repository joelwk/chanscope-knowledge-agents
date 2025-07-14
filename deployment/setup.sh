#!/bin/bash
set -e

# This script is a lightweight wrapper for running the main application.
# It ensures that the application starts with the correct user and environment.

echo "Starting application setup..."

# It's good practice to run as a non-root user.
# The user 'nobody' is created by default in many base images.
if [ "$(id -u)" = "0" ]; then
    echo "Running as root, switching to user 'nobody'"
    # Execute the rest of the script as the 'nobody' user
    exec su-exec nobody "$0" "$@"
fi

echo "Running application as user: $(whoami)"

# Execute the main application command passed to this script
exec "$@"