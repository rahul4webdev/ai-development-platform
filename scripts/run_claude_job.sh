#!/bin/bash
# =============================================================================
# Claude CLI Secure Execution Wrapper
# Phase 14.2: Secure Execution Wrapper
#
# This script is the ONLY way Claude CLI should be invoked.
# It enforces security boundaries, loads required documents, and captures output.
#
# Usage: ./run_claude_job.sh <job_id> <working_dir> <task_file>
#
# Exit Codes:
#   0 - Success
#   1 - Missing arguments
#   2 - Working directory does not exist
#   3 - Task file does not exist
#   4 - Required document missing
#   5 - API key not configured
#   6 - Claude CLI execution failed
#   7 - Timeout exceeded
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="/etc/claude-cli.env"
DOCS_DIR="/home/aitesting.mybd.in/public_html/docs"
LOG_DIR="/var/log/claude-jobs"

# Load environment
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# Required documents that must be copied to job workspace
REQUIRED_DOCS=(
    "AI_POLICY.md"
    "ARCHITECTURE.md"
    "CURRENT_STATE.md"
    "DEPLOYMENT.md"
    "PROJECT_CONTEXT.md"
    "PROJECT_MANIFEST.yaml"
    "TESTING_STRATEGY.md"
)

# Timeout for Claude execution (default: 10 minutes)
CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-600}"
CLAUDE_MAX_TURNS="${CLAUDE_MAX_TURNS:-50}"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "[$timestamp] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_error() { log "ERROR" "$@"; }
log_warn() { log "WARN" "$@"; }

cleanup() {
    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - START_TIME))

    log_info "Job $JOB_ID completed with exit code $exit_code in ${duration}s"

    # Write execution summary
    cat > "$WORKING_DIR/EXECUTION_SUMMARY.yaml" << EOF
job_id: $JOB_ID
exit_code: $exit_code
start_time: $(date -u -d "@$START_TIME" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -r "$START_TIME" -u +"%Y-%m-%dT%H:%M:%SZ")
end_time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
duration_seconds: $duration
status: $([ $exit_code -eq 0 ] && echo "success" || echo "failed")
EOF

    exit $exit_code
}

# -----------------------------------------------------------------------------
# Argument Validation
# -----------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <job_id> <working_dir> <task_file>"
    echo "  job_id      - Unique job identifier (UUID)"
    echo "  working_dir - Job workspace directory"
    echo "  task_file   - Path to TASK.md file with instructions"
    exit 1
fi

JOB_ID="$1"
WORKING_DIR="$2"
TASK_FILE="$3"

# Initialize logging
START_TIME=$(date +%s)
LOG_FILE="$LOG_DIR/job-${JOB_ID}.log"
mkdir -p "$LOG_DIR"

trap cleanup EXIT

log_info "Starting job $JOB_ID"
log_info "Working directory: $WORKING_DIR"
log_info "Task file: $TASK_FILE"

# -----------------------------------------------------------------------------
# Pre-execution Checks
# -----------------------------------------------------------------------------

# Check working directory exists
if [[ ! -d "$WORKING_DIR" ]]; then
    log_error "Working directory does not exist: $WORKING_DIR"
    exit 2
fi

# Check task file exists
if [[ ! -f "$TASK_FILE" ]]; then
    log_error "Task file does not exist: $TASK_FILE"
    exit 3
fi

# Check API key is configured
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    log_error "ANTHROPIC_API_KEY is not configured"
    log_error "Please add your API key to $ENV_FILE"
    exit 5
fi

# Check and copy required documents
log_info "Copying required documents to job workspace..."
for doc in "${REQUIRED_DOCS[@]}"; do
    src="$DOCS_DIR/$doc"
    dst="$WORKING_DIR/$doc"

    if [[ ! -f "$src" ]]; then
        log_error "Required document missing: $src"
        exit 4
    fi

    cp "$src" "$dst"
    log_info "  Copied: $doc"
done

# Copy task file to workspace if not already there
if [[ "$TASK_FILE" != "$WORKING_DIR/TASK.md" ]]; then
    cp "$TASK_FILE" "$WORKING_DIR/TASK.md"
fi

# Create logs directory in workspace
mkdir -p "$WORKING_DIR/logs"

# -----------------------------------------------------------------------------
# Build Claude CLI Command
# -----------------------------------------------------------------------------

# Construct the prompt that instructs Claude
SYSTEM_PROMPT="You are working in job workspace: $WORKING_DIR

MANDATORY: Before taking ANY action, you MUST read and comply with these files in order:
1. AI_POLICY.md - Non-negotiable rules
2. ARCHITECTURE.md - Technical constraints
3. PROJECT_CONTEXT.md - Business context
4. PROJECT_MANIFEST.yaml - Project configuration
5. CURRENT_STATE.md - Current system state
6. DEPLOYMENT.md - Deployment rules
7. TESTING_STRATEGY.md - Testing requirements

Your task is defined in: TASK.md

CONSTRAINTS:
- You may ONLY modify files within this workspace
- You MUST update CURRENT_STATE.md after completing any changes
- You may NOT modify AI_POLICY.md or ARCHITECTURE.md
- You may NOT access external networks except GitHub
- You may NOT execute destructive commands
- You MUST create atomic, well-documented commits

After completing your task:
1. Update CURRENT_STATE.md with what was done
2. Summarize your actions in logs/EXECUTION_LOG.md
3. List any blocking issues in logs/BLOCKERS.md (if any)

Proceed with the task defined in TASK.md."

# -----------------------------------------------------------------------------
# Execute Claude CLI
# -----------------------------------------------------------------------------
log_info "Executing Claude CLI..."
log_info "Timeout: ${CLAUDE_TIMEOUT}s"
log_info "Max turns: ${CLAUDE_MAX_TURNS}"

cd "$WORKING_DIR"

# Capture stdout and stderr
STDOUT_FILE="$WORKING_DIR/logs/stdout.log"
STDERR_FILE="$WORKING_DIR/logs/stderr.log"

# Execute with timeout
set +e
timeout "$CLAUDE_TIMEOUT" claude \
    --print \
    --output-format json \
    --max-turns "$CLAUDE_MAX_TURNS" \
    --dangerously-skip-permissions \
    --append-system-prompt "$SYSTEM_PROMPT" \
    "Read TASK.md and execute the task following all policy documents." \
    > "$STDOUT_FILE" 2> "$STDERR_FILE"

CLAUDE_EXIT_CODE=$?
set -e

# Check exit code
if [[ $CLAUDE_EXIT_CODE -eq 124 ]]; then
    log_error "Claude CLI execution timed out after ${CLAUDE_TIMEOUT}s"
    exit 7
fi

if [[ $CLAUDE_EXIT_CODE -ne 0 ]]; then
    log_error "Claude CLI execution failed with exit code $CLAUDE_EXIT_CODE"
    log_error "Stderr: $(cat "$STDERR_FILE")"
    exit 6
fi

log_info "Claude CLI execution completed successfully"

# -----------------------------------------------------------------------------
# Post-execution
# -----------------------------------------------------------------------------

# Parse output if JSON
if [[ -f "$STDOUT_FILE" ]] && command -v jq &> /dev/null; then
    if jq -e . "$STDOUT_FILE" > /dev/null 2>&1; then
        log_info "Parsing JSON output..."
        jq -r '.result // .message // .content // "No result field"' "$STDOUT_FILE" > "$WORKING_DIR/logs/result.txt" 2>/dev/null || true
    fi
fi

log_info "Job $JOB_ID completed successfully"
exit 0
