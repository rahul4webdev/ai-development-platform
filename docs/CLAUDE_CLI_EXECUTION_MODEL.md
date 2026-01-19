# Claude CLI Execution Model

## Overview

This document describes how the AI Development Platform integrates with Claude CLI for autonomous task execution. Understanding these authentication states is critical for reliable operation.

## Authentication States

Claude CLI has **4 distinct authentication states**:

| State | Description | `--version` | `--print` prompt |
|-------|-------------|-------------|------------------|
| **NOT_INSTALLED** | Claude CLI binary not found | ❌ Fails | ❌ Fails |
| **INSTALLED_NOT_AUTHENTICATED** | CLI installed, no valid auth | ✅ Works | ❌ "Invalid API key" |
| **AUTHENTICATED_INTERACTIVE** | OAuth session active (interactive only) | ✅ Works | ❌ May fail in automation |
| **AUTHENTICATED_FOR_AUTOMATION** | API key or setup-token configured | ✅ Works | ✅ Works |

### State Detection Logic

```python
def detect_claude_cli_state():
    # 1. Check if binary exists
    if not shutil.which("claude"):
        return "NOT_INSTALLED"

    # 2. Check if version command works
    result = subprocess.run(["claude", "--version"], capture_output=True)
    if result.returncode != 0:
        return "NOT_INSTALLED"  # Binary broken

    # 3. Try a real execution with --print
    result = subprocess.run(
        ["claude", "--print", "respond with exactly: PING"],
        capture_output=True,
        timeout=30
    )

    if "Invalid API key" in result.stderr or result.returncode != 0:
        return "INSTALLED_NOT_AUTHENTICATED"

    if "PING" in result.stdout:
        return "AUTHENTICATED_FOR_AUTOMATION"

    return "INSTALLED_NOT_AUTHENTICATED"
```

## Authentication Methods

### Method 1: ANTHROPIC_API_KEY Environment Variable (Recommended for Automation)

The simplest method for service/automation use:

```bash
# Set in environment
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Or in systemd service file
Environment=ANTHROPIC_API_KEY=sk-ant-api03-...

# Verify
echo "respond with OK" | claude --print
```

**Pros:**
- Works immediately
- No interactive setup required
- Clear and explicit

**Cons:**
- Requires API key (separate from Claude subscription)
- Usage billed directly

### Method 2: setup-token (For Claude Subscription Users)

For users with Claude Pro/Team subscriptions who want to use their subscription quota:

```bash
# Interactive setup (requires TTY)
claude setup-token

# This will prompt you to visit a URL and enter a code
# Creates a long-lived token in ~/.claude/
```

**Pros:**
- Uses existing subscription
- Long-lived token

**Cons:**
- Requires interactive setup
- Requires Claude subscription
- Token may expire periodically

### Method 3: OAuth Session (Interactive Use Only)

Default when running `claude` interactively for the first time:

```bash
# Opens browser for authentication
claude
```

**Warning:** OAuth sessions are NOT reliable for automation:
- Session may expire
- May require browser interaction to refresh
- The `oauthAccount` field in `~/.claude.json` is just metadata, not an active session

## Platform Integration

### ClaudeBackend Execution Flow

```
1. Job submitted to scheduler
2. Worker picks up job
3. ClaudeBackend.execute_job() called
4. ExecutionGate checks permissions
5. check_claude_availability() verifies auth state
6. If AUTHENTICATED_FOR_AUTOMATION:
   - Execute: claude --print "{prompt}"
   - Stream output to job workspace
   - Parse result
7. If NOT authenticated:
   - Job fails with clear error message
   - Scheduler can retry or mark failed
```

### Health Check Implementation

```python
async def check_claude_availability() -> dict:
    """
    Returns:
        {
            "installed": bool,
            "version": str or None,
            "authenticated": bool,
            "can_execute": bool,  # THE CRITICAL CHECK
            "auth_method": str or None,  # "api_key" | "setup_token" | None
            "error": str or None
        }
    """
```

### Required for Production

For the platform to execute Claude jobs autonomously, ONE of these must be true:

1. `ANTHROPIC_API_KEY` environment variable is set with a valid key
2. `claude setup-token` has been run successfully

## VPS Setup Checklist

```bash
# 1. Verify Claude CLI installed
claude --version

# 2. Choose authentication method:

# Option A: API Key (recommended)
export ANTHROPIC_API_KEY="your-key-here"
# Add to /etc/environment or systemd service for persistence

# Option B: Setup Token (if you have Claude subscription)
claude setup-token
# Follow the interactive prompts

# 3. Verify execution works
echo "respond with exactly: SUCCESS" | claude --print
# Should output: SUCCESS

# 4. Start platform services
systemctl start ai-controller
systemctl start telegram-bot
```

## Troubleshooting

### "Invalid API key" Error

**Symptoms:**
- `claude --version` works
- `echo "test" | claude --print` fails with "Invalid API key"

**Causes:**
1. No `ANTHROPIC_API_KEY` set
2. `setup-token` not configured
3. OAuth session expired (not valid for automation)

**Solutions:**
```bash
# Check current environment
echo $ANTHROPIC_API_KEY

# Set API key
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Or run setup-token
claude setup-token
```

### "Raw mode is not supported" Error

**Symptoms:**
```
ERROR Raw mode is not supported on the current process.stdin
```

**Cause:** Running interactive `claude` command in a non-TTY context

**Solution:** Always use `--print` flag for automation:
```bash
# Wrong (tries to be interactive)
claude "your prompt"

# Correct (non-interactive)
echo "your prompt" | claude --print
# or
claude --print "your prompt"
```

### Permissions Issues

**Symptoms:** Jobs fail even though auth works

**Check:** ExecutionGate may be blocking the action based on:
- Job lifecycle state
- User role
- Action type

**Debug:**
```python
from controller.execution_gate import ExecutionGate
gate = ExecutionGate()
result = gate.check_permission(job, user, action)
print(result)  # Shows allowed/denied and reason
```

## Security Considerations

1. **Never commit API keys** to version control
2. **Use environment variables** or secure secret management
3. **Audit trail** - All Claude executions are logged
4. **ExecutionGate** - Enforces lifecycle-based permissions
5. **DEPLOY_PROD always blocked** - Production deployment requires human approval

## Version History

- **0.15.7**: Added execution model documentation, fixed auth detection
- **0.15.6**: ExecutionGate model for permission enforcement
- **0.15.5**: Claude CLI session-based auth support
- **0.14.x**: Initial Claude CLI integration
