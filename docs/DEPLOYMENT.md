# Deployment Configuration

This file defines environments, domains, deployment commands, and rollback strategies.
The AI agent MUST follow this exactly when deploying.

---

## Environments

| Environment | Purpose                        | Deployment Trigger      | Human Approval Required |
|-------------|--------------------------------|-------------------------|-------------------------|
| Development | Local/dev server testing       | Automatic on push       | No                      |
| Testing     | Human validation environment   | Automatic on PR merge   | No                      |
| Production  | Live production environment    | Manual trigger only     | Yes                     |

---

## Environment Configuration

### Development
- **Domain**: `localhost`
- **Purpose**: Rapid iteration and debugging
- **Data**: Mock/seed data only
- **Secrets**: Local `.env` file (not committed)

### Testing (CONFIRMED)
- **Domain**: `aitesting.mybd.in`
- **Purpose**: Human validation before production
- **Data**: Sanitized copy of production or realistic test data
- **Secrets**: Environment variables in CI/CD

### Production (CONFIRMED)
- **Domain**: `ai.mybd.in`
- **Purpose**: Live user-facing environment
- **Data**: Real production data
- **Secrets**: Secure secret manager (GitHub Secrets)

---

## Hosting Infrastructure (CONFIRMED)

| Component        | Value                               |
|------------------|-------------------------------------|
| VPS OS           | AlmaLinux 9                         |
| Control Panel    | CyberPanel                          |
| Web Server       | OpenLiteSpeed                       |
| Repository       | github.com/rahul4webdev/ai-development-platform |

**Note**: Server path and deployment credentials will be provided per project.

---

## Deployment Commands

### Development Deployment
```bash
# Run locally
cd projects/{project-name}/repo
./scripts/deploy-dev.sh
# OR
docker-compose -f docker-compose.dev.yml up --build
```

### Testing Deployment
```bash
# Triggered via CI/CD pipeline
# Manual command if needed:
cd projects/{project-name}/repo
./scripts/deploy-test.sh
```

### Production Deployment
```bash
# MUST be triggered via explicit workflow dispatch
# Command reference (executed by CI/CD, not manually):
cd projects/{project-name}/repo
./scripts/deploy-prod.sh
```

---

## CI/CD Pipeline Requirements

### Pre-Deployment Checks (All Environments)
1. All tests pass
2. Linting passes
3. Coverage threshold met (minimum 70% [ASSUMPTION])
4. No security vulnerabilities in dependencies
5. Policy compliance validated

### Testing Environment Additional Checks
- Build succeeds
- Integration tests pass
- Notify chat bot of deployment URL

### Production Environment Additional Checks
- Manual approval via workflow dispatch
- Testing environment validation confirmed
- Version tag created
- Changelog updated

---

## Rollback Strategy

### Automatic Rollback Triggers
- Health check failures post-deployment
- Error rate exceeds threshold (5% [ASSUMPTION])
- Response time exceeds threshold (2s [ASSUMPTION])

### Manual Rollback Command
```bash
# Rollback to previous version
./scripts/rollback.sh {environment} {version}

# Example:
./scripts/rollback.sh production v1.2.3
```

### Rollback Procedure
1. Identify failing version
2. Fetch previous known-good version tag
3. Deploy previous version
4. Verify health checks pass
5. Update CURRENT_STATE.md with rollback details
6. Notify via chat bot
7. Create incident report

---

## Environment Variables

### Required for All Environments
| Variable            | Description                     | Source              |
|---------------------|---------------------------------|---------------------|
| `NODE_ENV`          | Environment name                | CI/CD               |
| `LOG_LEVEL`         | Logging verbosity               | CI/CD               |
| `API_BASE_URL`      | Base URL for API                | CI/CD               |

### Sensitive Variables (Never in Code)
| Variable            | Description                     | Storage             |
|---------------------|---------------------------------|---------------------|
| `DATABASE_URL`      | Database connection string      | Secret Manager      |
| `API_SECRET_KEY`    | API authentication key          | Secret Manager      |
| `TELEGRAM_BOT_TOKEN`| Telegram bot token              | Secret Manager      |

---

## Deployment Notifications

On successful deployment, notify via chat:
```
✅ Deployment Complete
Environment: {environment}
Version: {version}
URL: {deployment_url}
Time: {timestamp}
```

On failed deployment:
```
❌ Deployment Failed
Environment: {environment}
Version: {version}
Error: {error_summary}
Action Required: Review logs at {log_url}
```

---

## Confirmed Configuration

- CI/CD platform: GitHub Actions
- Container-based deployments: Docker
- Testing domain: aitesting.mybd.in
- Production domain: ai.mybd.in
- Hosting: AlmaLinux 9 + CyberPanel + OpenLiteSpeed
- Coverage threshold: 70%
- Error rate threshold: 5%
- Response time threshold: 2 seconds

## Pending Items

- Server path and credentials (to be provided)
- Telegram bot token (to be created)

---

*This file governs all deployments. The AI agent must follow this exactly.*
