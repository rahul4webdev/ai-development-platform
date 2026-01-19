# System Verification Audit Report

**Audit Date**: 2026-01-19
**Auditor**: Claude Code (Automated Verification)
**Platform Version**: 0.15.4 (codebase), 0.14.0 (running service)
**Overall Status**: CONDITIONAL GO (with remediation required)

---

## Executive Summary

This audit verifies the operational readiness of the AI Development Platform for Phase A execution (Operational Foundation). The platform infrastructure is **partially operational** with several issues that must be addressed before Claude CLI execution can proceed.

### Critical Findings

| Finding | Severity | Status |
|---------|----------|--------|
| Claude CLI not accessible to controller service | HIGH | REMEDIATION REQUIRED |
| Service running stale code (0.14.0 vs 0.15.4) | MEDIUM | SERVICE RESTART REQUIRED |
| ANTHROPIC_API_KEY not configured | BLOCKER | HUMAN ACTION REQUIRED |
| Telegram bot token not created | BLOCKER | HUMAN ACTION REQUIRED |

---

## 1. Service & Version Consistency Check

### Local Codebase
| Component | Version | File |
|-----------|---------|------|
| Controller | 0.15.4 | controller/__init__.py |
| Telegram Bot | 0.13.0 | telegram_bot_v2/__init__.py |
| Current Phase | 15.4 | "Roadmap Intelligence" |

### VPS Codebase
| Component | Version | File |
|-----------|---------|------|
| Controller Module | 0.15.4 | controller/__init__.py |
| Telegram Bot Module | 0.13.0 | telegram_bot_v2/__init__.py |

### Running Services
| Service | Reported Version | Actual Code Version | Status |
|---------|------------------|---------------------|--------|
| ai-testing-controller | 0.14.0 | 0.15.4 | **STALE - RESTART REQUIRED** |
| ai-telegram-bot | 0.14.0 | 0.13.0/0.15.4 | **STALE - RESTART REQUIRED** |

### Version Mismatch Analysis
- **Root Cause**: Services started on 2026-01-18 15:35 UTC and have not been restarted since code updates
- **Impact**: New endpoints (lifecycle, ingestion, scheduler) not available
- **Remediation**: `systemctl restart ai-testing-controller ai-telegram-bot`

---

## 2. Claude CLI Binary Verification

### Binary Location
| Check | Result | Notes |
|-------|--------|-------|
| Claude CLI installed | YES | `/root/.local/bin/claude` |
| Version | 2.1.12 | Matches expected version |
| Binary type | Symlink | Points to `/root/.local/share/claude/versions/2.1.12` |
| System-wide access | NO | Not in `/usr/local/bin` |

### Path Accessibility
| User | Can Access Claude | Notes |
|------|-------------------|-------|
| root | YES | Via PATH in .bashrc |
| ai-controller | **NO** | Claude not in service PATH |

### Issue Detail
The `ai-controller` user (uid=984) runs the controller service but cannot execute `claude` because:
1. Claude CLI is installed in `/root/.local/bin/` (root's home directory)
2. `ai-controller` is a system user with shell `/sbin/nologin`
3. No symlink or PATH propagation exists for the service user

### Remediation Options
1. **Symlink** (Recommended): `ln -s /root/.local/bin/claude /usr/local/bin/claude`
2. **Environment in systemd**: Add `Environment="PATH=/root/.local/bin:$PATH"` to service
3. **Copy binary**: Copy claude binary to accessible location

---

## 3. Environment Propagation Check

### /etc/claude-cli.env
| Variable | Configured | Value |
|----------|------------|-------|
| ANTHROPIC_API_KEY | **NO** | Commented out (placeholder only) |
| CLAUDE_JOBS_DIR | YES | /home/aitesting.mybd.in/jobs |
| CLAUDE_DOCS_DIR | YES | /home/aitesting.mybd.in/public_html/docs |
| CLAUDE_MAX_TURNS | YES | 50 |
| CLAUDE_TIMEOUT | YES | 600 |
| CLAUDE_LOG_DIR | YES | /var/log/claude-jobs |

### Directory Status
| Directory | Exists | Writable | Contents |
|-----------|--------|----------|----------|
| /home/aitesting.mybd.in/jobs | YES | YES | Empty |
| /home/aitesting.mybd.in/public_html/docs | YES | YES | 11 files (all governance docs) |
| /var/log/claude-jobs | YES | YES | Empty |

### Required Governance Documents
| Document | Present | Owner |
|----------|---------|-------|
| AI_POLICY.md | YES | ai-controller |
| ARCHITECTURE.md | YES | ai-controller |
| CURRENT_STATE.md | YES | root |
| DEPLOYMENT.md | YES | ai-controller |
| EPICS.yaml | YES | root |
| MILESTONES.yaml | YES | root |
| PROJECT_CONTEXT.md | YES | ai-controller |
| PROJECT_MANIFEST.yaml | YES | ai-controller |
| ROADMAP.md | YES | root |
| TESTING_STRATEGY.md | YES | ai-controller |

---

## 4. Controller → Claude Backend Wiring Inspection

### API Endpoints (from controller code)
| Endpoint | Purpose | Status on Running Service |
|----------|---------|---------------------------|
| POST /claude/job | Create Claude job | Available |
| GET /claude/job/{id} | Get job status | Available |
| GET /claude/jobs | List jobs | Available |
| GET /claude/queue | Queue status | Available |
| GET /claude/status | Claude availability | Available |
| GET /claude/scheduler | Multi-worker status | **NOT FOUND (stale code)** |
| POST /lifecycle | Create lifecycle | **NOT FOUND (stale code)** |
| POST /ingestion | Create ingestion | **NOT FOUND (stale code)** |

### Claude Status Response (from running service)
```json
{
    "available": false,
    "cli": {
        "available": false,
        "version": null,
        "api_key_configured": false,
        "wrapper_exists": true,
        "error": "Claude CLI not installed"
    }
}
```

### Analysis
1. **Wrapper script exists**: `/home/aitesting.mybd.in/public_html/scripts/run_claude_job.sh` (7550 bytes, executable)
2. **Claude detection fails**: The controller runs as `ai-controller` user which cannot find `claude` in PATH
3. **API key not configured**: ANTHROPIC_API_KEY is commented out in `/etc/claude-cli.env`

### Wiring Readiness
| Component | Status | Issue |
|-----------|--------|-------|
| Wrapper Script | READY | Exists and executable |
| Claude CLI Detection | FAIL | PATH issue for ai-controller user |
| API Key | FAIL | Not configured |
| Job Queue | READY | Code present (needs restart) |
| Multi-worker Scheduler | READY | Code present (needs restart) |

---

## 5. Scheduler Dry-Run Validation

### Scheduler Configuration (from code)
| Setting | Value | Purpose |
|---------|-------|---------|
| MAX_CONCURRENT_JOBS | 3 | Maximum parallel Claude jobs |
| WORKER_NICE_VALUE | 10 | CPU priority reduction |
| WORKER_MEMORY_LIMIT_MB | 2048 | Per-worker memory cap |
| JOB_CLEANUP_AFTER_HOURS | 24 | Auto-cleanup threshold |
| STARVATION_THRESHOLD_MINUTES | 30 | Priority escalation trigger |
| PRIORITY_ESCALATION_CAP | 75 | Max priority via escalation |

### Scheduler Readiness
| Component | Status | Notes |
|-----------|--------|-------|
| MultiWorkerScheduler class | PRESENT | In claude_backend.py |
| PersistentJobStore | PRESENT | JSON-based persistence |
| ClaudeWorker | PRESENT | Process isolation ready |
| Priority Queue | PRESENT | EMERGENCY > HIGH > NORMAL > LOW |
| Starvation Prevention | PRESENT | Auto-escalation after 30 min |

### Cannot Perform Dry-Run
A scheduler dry-run cannot be executed because:
1. Service is running stale code (scheduler endpoint returns 404)
2. Claude CLI is not accessible to the service user
3. ANTHROPIC_API_KEY is not configured

---

## 6. Observability & Logging Check

### Health Monitoring
| Component | Status | Notes |
|-----------|--------|-------|
| Watchdog Service | ACTIVE | ai-controller-watchdog.service |
| Watchdog Timer | ACTIVE | Runs every minute |
| Health Endpoint | HEALTHY | /health returns 200 |
| Watchdog Log | CLEAN | All "Health OK" entries |

### Log Files
| Log | Location | Status |
|-----|----------|--------|
| Controller Log | /home/aitesting.mybd.in/public_html/uvicorn.log | Active, health checks visible |
| Telegram Bot Log | journald | Active |
| Watchdog Log | /var/log/ai-controller-watchdog.log | Active, all healthy |
| Claude Job Logs | /var/log/claude-jobs/ | Empty (no jobs run) |

### Service Health (as of audit time)
| Service | Memory | CPU | Uptime |
|---------|--------|-----|--------|
| ai-testing-controller | 39.2M / 500M max | 1m 40s total | 15+ hours |
| ai-telegram-bot | 34.6M / 350M max | 15s total | 15+ hours |

---

## 7. Go/No-Go Decision

### GO/NO-GO Matrix

| Criterion | Status | Verdict |
|-----------|--------|---------|
| Services running | YES | GO |
| Health checks passing | YES | GO |
| Codebase synced to VPS | YES | GO |
| Services running latest code | **NO** | NO-GO |
| Claude CLI accessible | **NO** | NO-GO |
| ANTHROPIC_API_KEY configured | **NO** | NO-GO |
| Telegram bot token | **NO** | NO-GO |
| Required docs present | YES | GO |
| Scheduler code ready | YES | GO |
| Watchdog operational | YES | GO |

### Overall Verdict: **CONDITIONAL GO**

The platform is structurally ready but requires remediation before Phase A/B execution.

---

## 8. Remediation Steps

### Immediate Actions (Before Next Session)

#### 1. Restart Services to Load New Code
```bash
ssh root@82.25.110.109
systemctl restart ai-testing-controller ai-telegram-bot
```
**Expected Result**: Services report version 0.15.4, new endpoints available

#### 2. Make Claude CLI Accessible System-Wide
```bash
ssh root@82.25.110.109
ln -s /root/.local/bin/claude /usr/local/bin/claude
chmod 755 /usr/local/bin/claude
```
**Expected Result**: `sudo -u ai-controller which claude` returns `/usr/local/bin/claude`

#### 3. Verify Claude CLI Access
```bash
ssh root@82.25.110.109
sudo -u ai-controller claude --version
```
**Expected Result**: Returns "2.1.12 (Claude Code)"

### Blocker Actions (Require Human)

#### 4. Create Telegram Bot Token
- Go to Telegram, message @BotFather
- Create new bot with `/newbot`
- Save token
- Update `/etc/ai-telegram-bot.env` with token

#### 5. Configure ANTHROPIC_API_KEY
```bash
ssh root@82.25.110.109
vi /etc/claude-cli.env
# Uncomment and set: ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

### Post-Remediation Verification
After completing steps 1-3, verify:
```bash
curl http://127.0.0.1:8001/health  # Should show version 0.15.4
curl http://127.0.0.1:8001/claude/status  # Should show available: true
curl http://127.0.0.1:8001/lifecycle  # Should return empty list, not 404
```

---

## 9. Summary

### What Works
- VPS infrastructure operational
- Systemd services running and monitored
- Codebase fully synced (0.15.4)
- All governance documents present
- Watchdog health monitoring active
- OpenLiteSpeed reverse proxy configured

### What Needs Fix
1. **Service restart** - Simple fix, no risk
2. **Claude CLI path** - Simple symlink, no risk
3. **API key** - Human action required
4. **Bot token** - Human action required

### Timeline to Full Readiness
- Steps 1-3: Immediate (5 minutes)
- Steps 4-5: Dependent on human availability
- Full verification: 10 minutes after all steps

---

*Report generated by Phase 15.5 System Verification Audit*
*This is a READ-ONLY audit - no system modifications were made*
