# Projects Directory

This directory contains managed project directories. Each project is:

- Fully isolated
- Independently deployable
- Independently trackable

## Project Structure

Each project directory should contain:

```
project-name/
├── PROJECT_MANIFEST.yaml    # Single source of truth
├── AI_POLICY.md             # Project-specific AI policy (inherits from platform)
├── PROJECT_CONTEXT.md       # Business memory
├── ARCHITECTURE.md          # Technical architecture
├── DEPLOYMENT.md            # Deployment configuration
├── TESTING_STRATEGY.md      # Testing approach
├── CURRENT_STATE.md         # Living system state
└── repo/                    # Actual project code (Git repository)
```

## Creating a New Project

Projects are created via the Task Controller during the bootstrap phase:

1. Upload handoff document via Telegram
2. Claude agent parses document and creates project structure
3. Project is registered in this directory

## Current Projects

(None registered yet - platform is in bootstrap phase)
