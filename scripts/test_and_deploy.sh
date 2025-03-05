#!/bin/bash
set -e

# Script to run tests and then deploy the application
# This script uses run_tests.sh to run tests and then deploys if tests pass

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --separate       Run tests and deployment as separate steps (default)"
    echo "  --integrated     Use integrated test and deploy workflow"
    echo "  --env=<environment>  Specify environment for testing: local, docker, or replit"
    echo "  --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --integrated"
    echo "  $0 --env=docker"
}

# Default values
WORKFLOW="separate"
ENVIRONMENT="docker"  # Default to Docker for deployment

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --separate)
            WORKFLOW="separate"
            shift
            ;;
        --integrated)
            WORKFLOW="integrated"
            shift
            ;;
        --env=*)
            ENVIRONMENT="${1#*=}"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Starting Knowledge Agent test and deploy workflow...${NC}"
echo -e "${YELLOW}Workflow mode: ${WORKFLOW}${NC}"
echo -e "${YELLOW}Test environment: ${ENVIRONMENT}${NC}"

if [ "$WORKFLOW" = "separate" ]; then
    # Separate test and deploy workflow
    echo -e "${YELLOW}Step 1: Running tests...${NC}"
    
    # Run tests using the new run_tests.sh script
    if [ "$ENVIRONMENT" = "docker" ]; then
        # For Docker, we need to build and run the tests
        "$SCRIPT_DIR/run_tests.sh" --env=docker
    else
        # For other environments, just run the tests
        "$SCRIPT_DIR/run_tests.sh" --env="$ENVIRONMENT"
    fi
    
    # Check if tests passed
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}Tests passed successfully!${NC}"
        echo -e "${YELLOW}Step 2: Deploying application...${NC}"
        
        # Build and deploy
        docker-compose -f deployment/docker-compose.yml build
        docker-compose -f deployment/docker-compose.yml up -d
        
        echo -e "${GREEN}Application deployed successfully!${NC}"
    else
        echo -e "${RED}Tests failed with exit code: $TEST_EXIT_CODE${NC}"
        echo -e "${RED}Deployment aborted.${NC}"
        exit 1
    fi
else
    # Integrated test and deploy workflow
    echo -e "${YELLOW}Running integrated test and deploy workflow...${NC}"
    
    # Build and deploy with tests enabled
    docker-compose -f deployment/docker-compose.yml build
    docker-compose -f deployment/docker-compose.yml up -d -e RUN_TESTS_ON_STARTUP=true
    
    echo -e "${YELLOW}Application starting with tests enabled...${NC}"
    echo -e "${YELLOW}Check logs to see test results:${NC}"
    echo -e "${YELLOW}docker-compose -f deployment/docker-compose.yml logs -f${NC}"
fi

echo -e "${GREEN}Workflow completed!${NC}"
exit 0 
