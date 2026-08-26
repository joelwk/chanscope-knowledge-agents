#!/usr/bin/env bash
set -euo pipefail

# Deployment helper for ChanScope web frontends.
# Mirrors production defaults (docker container on port 80) while allowing
# a convenience local mode that brings up the same FastAPI app with:
# - refresh dashboard mounted at /refresh
# - query console mounted at /query-ui

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENVIRONMENT="local"    # local | docker | replit
PORT="80"
RELOAD="false"
SKIP_INSTALL="false"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --env=local|docker|replit   Deployment target (default: local)
  --port=<port>                HTTP port for local mode (default: 80)
  --reload                     Enable uvicorn reload when running locally
  --skip-install               Skip Python dependency installation check
  --help                       Show this help message

Examples:
  $(basename "$0") --env=docker
  $(basename "$0") --port=8080 --reload
EOF
}

log() {
    printf "%b\n" "$*" >&2
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "${RED}Missing required command: $1${NC}"
        exit 1
    fi
}

ensure_libstdcxx() {
    # Verify libstdc++.so.6 is reachable; install when possible.
    if command -v ldconfig >/dev/null 2>&1; then
        if ldconfig -p 2>/dev/null | grep -q 'libstdc\+\+\.so\.6'; then
            return 0
        fi
    elif [[ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]]; then
        return 0
    fi

    log "${YELLOW}libstdc++.so.6 not detected; attempting to install dependency...${NC}"

    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y libstdc++6
    elif command -v nix-env >/dev/null 2>&1; then
        nix-env -iA nixpkgs.gcc
    else
        log "${RED}Unable to install libstdc++ automatically. Please install libstdc++6 for your platform and re-run.${NC}"
        exit 1
    fi
}

ensure_virtualenv() {
    if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        return 0
    fi

    log "${YELLOW}Creating Python virtual environment...${NC}"
    require_command python3
    python3 -m venv "${PROJECT_ROOT}/.venv"
}

install_requirements() {
    if [[ "${SKIP_INSTALL}" == "true" ]]; then
        return 0
    fi

    log "${YELLOW}Installing Python dependencies...${NC}"
    "${PROJECT_ROOT}/.venv/bin/pip" install --upgrade pip
    "${PROJECT_ROOT}/.venv/bin/pip" install -r "${PROJECT_ROOT}/requirements.txt"
}

verify_frontend_routes() {
    local base_url="$1"
    local max_attempts="${2:-15}"
    local delay_seconds="${3:-2}"

    if ! command -v curl >/dev/null 2>&1; then
        log "${YELLOW}curl not found; skipping frontend route verification.${NC}"
        return 0
    fi

    local path
    for path in "/refresh" "/query-ui"; do
        local attempt=1
        local ok="false"
        while [[ "${attempt}" -le "${max_attempts}" ]]; do
            if curl -fsS -o /dev/null "${base_url}${path}"; then
                ok="true"
                break
            fi
            sleep "${delay_seconds}"
            attempt=$((attempt + 1))
        done

        if [[ "${ok}" == "true" ]]; then
            log "${GREEN}Verified ${base_url}${path}${NC}"
        else
            log "${RED}Failed to verify ${base_url}${path} after ${max_attempts} attempts${NC}"
            return 1
        fi
    done
}

start_local_server() {
    ensure_libstdcxx
    ensure_virtualenv
    install_requirements

    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"

    export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/.pythonlibs/lib/python3.11/site-packages:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${PROJECT_ROOT}/.venv/lib/python3.11/site-packages"

    log "${YELLOW}Starting uvicorn on port ${PORT} (reload=${RELOAD})...${NC}"
    log "${GREEN}Frontend URLs:${NC}"
    log "  - Refresh dashboard: http://localhost:${PORT}/refresh"
    log "  - Query console:     http://localhost:${PORT}/query-ui"
    if [[ "${RELOAD}" == "true" ]]; then
        uvicorn api.app:app --host 0.0.0.0 --port "${PORT}" --reload
    else
        uvicorn api.app:app --host 0.0.0.0 --port "${PORT}"
    fi
}

start_docker_stack() {
    require_command docker
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE="docker-compose"
    else
        require_command docker
        COMPOSE="docker compose"
    fi

    ensure_libstdcxx

    log "${YELLOW}Building production container...${NC}"
    (cd "${PROJECT_ROOT}" && ${COMPOSE} -f deployment/docker-compose.yml build app)

    log "${YELLOW}Starting frontend-enabled API service...${NC}"
    (cd "${PROJECT_ROOT}" && API_PORT="${PORT}" ${COMPOSE} -f deployment/docker-compose.yml up -d app)

    log "${GREEN}Frontend URLs:${NC}"
    log "  - Refresh dashboard: http://localhost:${PORT}/refresh"
    log "  - Query console:     http://localhost:${PORT}/query-ui"
    verify_frontend_routes "http://localhost:${PORT}"
}

print_replit_guidance() {
    log "${YELLOW}Replit deployment helper${NC}"
    log "  1. Ensure the Nix environment has re-built after updating replit.nix."
    log "  2. Run scripts/replit_setup.sh to validate configuration."
    log "  3. Use scripts/replit_startup.sh during deployment, or start the Replit run command."
    log "  4. Frontend URLs after deploy:"
    log "     - Refresh dashboard: https://<your-repl>.replit.app/refresh"
    log "     - Query console:     https://<your-repl>.replit.app/query-ui"
}

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --env=*)
            ENVIRONMENT="${arg#*=}"
            ;;
        --port=*)
            PORT="${arg#*=}"
            ;;
        --reload)
            RELOAD="true"
            ;;
        --skip-install)
            SKIP_INSTALL="true"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log "${RED}Unknown option: $arg${NC}"
            usage
            exit 1
            ;;
    esac
done

case "${ENVIRONMENT}" in
    local)
        start_local_server
        ;;
    docker)
        start_docker_stack
        ;;
    replit)
        print_replit_guidance
        ;;
    *)
        log "${RED}Invalid environment: ${ENVIRONMENT}${NC}"
        usage
        exit 1
        ;;
esac
