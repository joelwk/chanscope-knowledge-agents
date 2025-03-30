#!/bin/bash
set -e

# Unified test and deploy script for Chanscope
# Supports both Docker and Replit environments with a consistent interface

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
    echo "  --env=<environment>  Specify environment for testing/deployment: docker or replit"
    echo "  --skip-tests     Skip tests and go straight to deployment"
    echo "  --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --integrated"
    echo "  $0 --env=replit"
}

# Default values
WORKFLOW="separate"
ENVIRONMENT="auto"
SKIP_TESTS="false"

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
        --skip-tests)
            SKIP_TESTS="true"
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

# Auto-detect environment if not specified
if [ "$ENVIRONMENT" = "auto" ]; then
    # Check for Docker environment
    if [ -f "/.dockerenv" ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        ENVIRONMENT="docker"
        echo -e "${YELLOW}Auto-detected Docker environment${NC}"
    # Check for Replit environment - enhanced detection
    elif [ -n "$REPL_ID" ] || [ -n "$REPL_SLUG" ] || [ -n "$REPL_OWNER" ] || [ "$REPLIT_ENV" = "replit" ] || [ -d "/home/runner" ]; then
        ENVIRONMENT="replit"
        echo -e "${YELLOW}Auto-detected Replit environment${NC}"
    else
        echo -e "${YELLOW}No specific environment detected, defaulting to Docker${NC}"
        ENVIRONMENT="docker"
    fi
fi

# Validate environment
if [ "$ENVIRONMENT" != "docker" ] && [ "$ENVIRONMENT" != "replit" ]; then
    echo -e "${RED}Error: Invalid environment specified: $ENVIRONMENT${NC}"
    echo -e "${YELLOW}Valid environments are: docker, replit${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting Knowledge Agent test and deploy workflow...${NC}"
echo -e "${YELLOW}Workflow mode: ${WORKFLOW}${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests in ${ENVIRONMENT} environment...${NC}"
    
    if [ "$ENVIRONMENT" = "docker" ]; then
        bash "$SCRIPT_DIR/run_tests.sh" --env=docker --all
    else
        bash "$SCRIPT_DIR/run_tests.sh" --env=replit --all
    fi
    
    return $?
}

# Function to deploy Docker
deploy_docker() {
    echo -e "${YELLOW}Deploying application in Docker environment...${NC}"
    
    # Build and deploy Docker container
    docker-compose -f deployment/docker-compose.yml build
    docker-compose -f deployment/docker-compose.yml up -d
    
    # Check if deployment was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Docker deployment successful!${NC}"
        
        # Get container status and health
        echo -e "${YELLOW}Container status:${NC}"
        docker-compose -f deployment/docker-compose.yml ps
        
        # Print logs for initial health check
        echo -e "${YELLOW}Container logs:${NC}"
        docker-compose -f deployment/docker-compose.yml logs --tail=20
        
        echo -e "${YELLOW}Application deployed and running at:${NC}"
        echo -e "${GREEN}http://localhost:${API_PORT:-80}${NC}"
        
        return 0
    else
        echo -e "${RED}Docker deployment failed!${NC}"
        return 1
    fi
}

# Function to deploy Replit
deploy_replit() {
    echo -e "${YELLOW}Deploying application in Replit environment...${NC}"
    
    # Ensure .replit file has correct configuration
    if [ -f ".replit" ]; then
        echo -e "${YELLOW}Verifying .replit configuration...${NC}"
        
        # Check if ports are properly configured
        if ! grep -q '^\[\[ports\]\]' .replit; then
            echo -e "${YELLOW}Adding port configuration to .replit...${NC}"
            cat >> .replit << EOF

[[ports]]
localPort = 80
externalPort = 80
exposeLocalhost = true
EOF
        fi
        
        # Check if deployment target is set
        if ! grep -q "deploymentTarget" .replit; then
            echo -e "${YELLOW}Adding deployment target to .replit...${NC}"
            cat >> .replit << EOF

[deployment]
run = ["python3", "main.py"]
deploymentTarget = "cloudrun"
EOF
        fi
    else
        echo -e "${RED}Error: .replit file not found!${NC}"
        return 1
    fi
    
    # Run Replit setup script
    if [ -f "$SCRIPT_DIR/replit_setup.sh" ]; then
        echo -e "${YELLOW}Running Replit setup...${NC}"
        bash "$SCRIPT_DIR/replit_setup.sh"
    fi
    
    echo -e "${GREEN}Replit deployment configuration complete${NC}"
    echo -e "${YELLOW}To complete deployment:${NC}"
    echo -e "1. Click the 'Run' button in the Replit interface"
    echo -e "2. For permanent deployment, use the Replit Deployments tab to create a deployment"
    
    return 0
}

# Deploy the application based on environment
deploy_application() {
    if [ "$ENVIRONMENT" = "docker" ]; then
        deploy_docker
    else
        deploy_replit
    fi
    
    return $?
}

# Run the appropriate workflow
if [ "$WORKFLOW" = "separate" ]; then
    # Separate test and deploy workflow
    if [ "$SKIP_TESTS" = "false" ]; then
        echo -e "${YELLOW}Step 1: Running tests...${NC}"
        run_tests
        
        # Check if tests passed
        TEST_EXIT_CODE=$?
        
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}Tests passed successfully!${NC}"
            echo -e "${YELLOW}Step 2: Deploying application...${NC}"
            deploy_application
        else
            echo -e "${RED}Tests failed with exit code: $TEST_EXIT_CODE${NC}"
            echo -e "${RED}Deployment aborted.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Skipping tests as requested...${NC}"
        deploy_application
    fi
else
    # Integrated test and deploy workflow
    echo -e "${YELLOW}Running integrated test and deploy workflow...${NC}"
    
    if [ "$ENVIRONMENT" = "docker" ]; then
        # For Docker, use RUN_TESTS_ON_STARTUP
        docker-compose -f deployment/docker-compose.yml build
        docker-compose -f deployment/docker-compose.yml up -d -e RUN_TESTS_ON_STARTUP=true
        
        echo -e "${YELLOW}Application starting with tests enabled...${NC}"
        echo -e "${YELLOW}Check logs to see test results:${NC}"
        echo -e "${YELLOW}docker-compose -f deployment/docker-compose.yml logs -f${NC}"
    else
        # For Replit, we first run tests and then deploy if they pass
        if [ "$SKIP_TESTS" = "false" ]; then
            run_tests
            TEST_EXIT_CODE=$?
            
            if [ $TEST_EXIT_CODE -eq 0 ]; then
                echo -e "${GREEN}Tests passed successfully!${NC}"
                deploy_replit
            else
                echo -e "${RED}Tests failed with exit code: $TEST_EXIT_CODE${NC}"
                echo -e "${RED}Deployment aborted.${NC}"
                exit 1
            fi
        else
            echo -e "${YELLOW}Skipping tests as requested...${NC}"
            deploy_replit
        fi
    fi
fi

echo -e "${GREEN}Workflow completed!${NC}"
exit 0 
