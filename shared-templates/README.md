# Shared Project Templates

This directory contains project scaffolding templates for the AI Development Platform.
Templates include CI/CD configurations, deployment workflows, and best practices that
prevent common issues identified during project development.

## Available Templates

### 1. fastapi-react/
Full-stack template with FastAPI backend and React frontend.
- Pre-configured CI/CD with known-good dependencies
- Correct bcrypt/passlib pinning for compatibility
- npm install with --legacy-peer-deps for React
- useCallback patterns in React contexts
- CyberPanel proxy configuration templates

### 2. nextjs-app/
Next.js SSR application template.
- Proper Node.js version pinning
- SSR proxy configuration for OpenLiteSpeed

### 3. python-api/
Python-only API template.
- FastAPI with best-practice dependencies
- pytest configuration with coverage

### 4. react-spa/
React Single Page Application (no backend).
- Optimized build configuration
- Static deployment setup

## Template Variables

Templates use `{{VARIABLE}}` placeholders that are replaced during project creation:

- `{{PROJECT_NAME}}` - Project name (e.g., health-tracker-app)
- `{{PROJECT_SLUG}}` - URL-safe project slug
- `{{API_DOMAIN}}` - API domain (e.g., healthapi.gahfaudio.in)
- `{{WEB_DOMAIN}}` - Frontend domain (e.g., health.gahfaudio.in)
- `{{DB_NAME}}` - Database name
- `{{GITHUB_USER}}` - GitHub username for repository

## Usage

Templates are automatically selected based on the project's tech stack defined
in the CHD (Change Handoff Document). The template engine in
`controller/template_engine.py` handles variable substitution and file generation.

## CI/CD Fixes Built-In

All templates include fixes for common CI issues:
1. `bcrypt>=4.0.0,<4.2.0` - Prevents passlib compatibility issues
2. `npm install --legacy-peer-deps` - Prevents peer dependency conflicts
3. `useCallback` patterns - Prevents ESLint exhaustive-deps warnings
4. Proper flake8 configuration - Prevents unused import errors
5. Node 18 pinning - Consistent with GitHub Actions runners
