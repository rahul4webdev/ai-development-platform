# AI-Driven Autonomous Development Platform

An AI-driven platform enabling near-fully autonomous software development where an AI agent (Claude CLI) performs development, testing, deployment, and iteration with minimal human involvement.

## Project Status

**Current Phase**: Bootstrap / Phase 1 Skeleton

This repository contains the platform infrastructure skeleton. Feature implementation is pending.

## Architecture

```
User (Mobile / Desktop Chat)
        ↓
Telegram Bot (bots/)
        ↓
Task Controller (controller/)
        ↓
Claude CLI Agent (future)
        ↓
Git Repository + Filesystem
        ↓
CI/CD Pipeline (GitHub Actions)
        ↓
Testing → Production Environments
```

## Repository Structure

```
ai-development-platform/
├── controller/          # Task Controller (FastAPI)
│   ├── __init__.py
│   └── main.py
├── bots/                # Chat bot implementations
│   ├── __init__.py
│   └── telegram_bot.py
├── projects/            # Managed project directories
│   └── README.md
├── workflows/           # GitHub Actions CI/CD
│   └── ci.yml
├── docs/                # Documentation
│   ├── AI_POLICY.md
│   ├── PROJECT_CONTEXT.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── TESTING_STRATEGY.md
│   ├── PROJECT_MANIFEST.yaml
│   └── CURRENT_STATE.md
├── utils/               # Shared utilities
│   └── README.md
├── tests/               # Test suite
│   ├── test_controller.py
│   └── test_telegram_bot.py
├── requirements.txt     # Python dependencies
├── pytest.ini           # Test configuration
└── README.md            # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/rahul4webdev/ai-development-platform.git
cd ai-development-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Task Controller

```bash
# Development mode
uvicorn controller.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# http://localhost:8000/docs
```

### Running Tests

```bash
pytest tests/ -v
```

### Running the Telegram Bot (requires token)

```bash
export TELEGRAM_BOT_TOKEN="your-token-here"
python -m bots.telegram_bot
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | For bot only |
| `CONTROLLER_URL` | Task Controller URL | Default: localhost:8000 |

## Environments

| Environment | Domain | Purpose |
|-------------|--------|---------|
| Development | localhost | Local development |
| Testing | aitesting.mybd.in | Human validation |
| Production | ai.mybd.in | Live environment |

## Documentation

See the `docs/` directory for detailed documentation:

- [AI_POLICY.md](docs/AI_POLICY.md) - AI agent rules and constraints
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment configuration
- [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) - Testing approach
- [CURRENT_STATE.md](docs/CURRENT_STATE.md) - Current system state

## Contributing

This is an AI-driven autonomous development platform. Human contributions should focus on:

1. Policy updates (AI_POLICY.md)
2. Architecture decisions (ARCHITECTURE.md)
3. High-level feature requests
4. Validation and approval

## License

[TBD]
