# AI Policy (Authority File)

This file defines non-negotiable rules that govern AI agent behavior on this platform.
The AI agent MUST read and comply with this file at the start of every session.

---

## Core Principles

1. **Policy Over Permission**: Rules are enforced by files and CI/CD. The agent does not ask for approvals repeatedly.
2. **Autonomy with Guardrails**: Full autonomy in development and testing; controlled autonomy in production.
3. **Stateless Agent, Stateful System**: The agent can restart at any time; system state remains intact via persistent files.

---

## Mandatory Rules

### Deployment Rules
- **NEVER** deploy to production without an explicit human trigger
- **ALWAYS** deploy to testing environment before production
- **ALWAYS** verify deployment success before marking task complete
- **ALWAYS** follow rollback procedures defined in DEPLOYMENT.md if deployment fails

### Testing Rules
- **ALWAYS** run unit tests before any deployment
- **ALWAYS** ensure tests pass before deploying to testing environment
- **NEVER** skip test execution to save time
- **ALWAYS** update test coverage when adding new features

### Data Rules
- **NEVER** delete production data
- **NEVER** access production database directly
- **NEVER** store secrets in the repository
- **ALWAYS** use environment variables for sensitive configuration

### State Management Rules
- **ALWAYS** update CURRENT_STATE.md after completing any task
- **ALWAYS** read PROJECT_CONTEXT.md at session start
- **ALWAYS** follow architecture defined in ARCHITECTURE.md
- **NEVER** make changes that contradict PROJECT_MANIFEST.yaml

### Communication Rules
- **ALWAYS** provide structured, summarized outputs (not raw logs)
- **ALWAYS** notify when human validation is required
- **ALWAYS** report blockers immediately
- **NEVER** proceed with ambiguous requirements; ask clarifying questions once, then stop

### Security Rules
- **ALWAYS** follow security best practices (OWASP guidelines)
- **NEVER** expose API keys or credentials in code or logs
- **ALWAYS** validate external inputs
- **ALWAYS** log all significant actions for audit

---

## Autonomy Mode Behaviors

The agent's behavior adapts based on the current autonomy mode (stored in PROJECT_MANIFEST.yaml):

| Mode        | Allowed Actions                                      | Restricted Actions                |
|-------------|------------------------------------------------------|-----------------------------------|
| bootstrap   | Scaffold project, create files, ask questions        | Deploy, modify production         |
| development | Implement features, write tests, deploy to dev/test  | Deploy to production              |
| hardening   | Fix bugs, improve stability, enhance tests           | Add new features                  |
| release     | Prepare release, deploy to production (with trigger) | Major refactoring                 |
| maintenance | Bug fixes, minor updates, monitoring                 | Large feature additions           |

---

## Enforcement

These policies are enforced via:
- CI/CD pipeline checks
- Pre-commit hooks
- Deployment gating
- Automated policy compliance scans

Violations will block deployments and trigger alerts.

---

## Assumptions

- [ASSUMPTION] CI/CD platform is GitHub Actions (can be changed)
- [ASSUMPTION] Production deployment requires manual workflow dispatch trigger
- [ASSUMPTION] Secrets are managed via environment variables or external secret manager

---

*This file replaces repeated human approvals. Update only with explicit authorization.*
