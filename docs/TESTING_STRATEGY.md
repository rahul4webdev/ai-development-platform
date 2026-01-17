# Testing Strategy

This file defines the multi-layer testing approach for the platform.
All testing requirements here are mandatory for the AI agent.

---

## Testing Philosophy

1. **Tests are mandatory**: No deployment without passing tests
2. **Automated first**: Maximize automated testing, minimize manual testing
3. **Fail fast**: Catch issues early in the pipeline
4. **Human validation**: Final gate before production is human testing on testing domain

---

## Testing Layers

```
┌─────────────────────────────────────────┐
│     Layer 4: Human Validation           │  ← Testing domain
│     (Manual testing before production)  │
├─────────────────────────────────────────┤
│     Layer 3: Integration Tests          │  ← CI/CD
│     (API, component integration)        │
├─────────────────────────────────────────┤
│     Layer 2: Dev/Test Verification      │  ← Automated
│     (Environment smoke tests)           │
├─────────────────────────────────────────┤
│     Layer 1: Unit Tests                 │  ← Mandatory
│     (Functions, modules, components)    │
└─────────────────────────────────────────┘
```

---

## Layer 1: Unit Tests (Mandatory)

### Requirements
- **Coverage Threshold**: Minimum 70% [ASSUMPTION]
- **Execution**: On every commit
- **Blocking**: Merge blocked if tests fail

### What to Test
- Individual functions and methods
- Business logic
- Data transformations
- Edge cases and error handling

### What NOT to Test
- External API calls (mock these)
- Database queries (use test fixtures)
- UI rendering details (test behavior, not implementation)

### Frameworks
| Language     | Framework          |
|--------------|--------------------|
| Python       | pytest             |
| JavaScript   | Jest               |
| TypeScript   | Jest / Vitest      |

---

## Layer 2: Dev/Test Environment Verification

### Requirements
- **Execution**: After deployment to dev/test
- **Scope**: Smoke tests to verify environment health

### Checks
- [ ] Application starts successfully
- [ ] Health endpoint responds
- [ ] Database connection works
- [ ] External service connections verified
- [ ] Critical user flows functional

### Automation
```bash
# Smoke test script example
./scripts/smoke-test.sh {environment}
```

---

## Layer 3: Integration Tests

### Requirements
- **Execution**: On PR merge, before testing deployment
- **Scope**: Component interactions, API contracts

### What to Test
- API endpoint responses
- Service-to-service communication
- Database operations with test database
- Authentication/authorization flows

### Test Data
- Use fixtures or factories
- Never use production data
- Reset test database before each run

---

## Layer 4: Human Validation

### When Required
- Feature deployed to testing environment
- Bug fix deployed to testing environment
- Before any production promotion

### Process
1. AI agent deploys to testing domain
2. AI agent notifies via chat: "Testing ready at {url}"
3. Human tests in browser/app
4. Human responds:
   - "Approved → promote to production"
   - "Issue found: {description}"

### Human Testing Checklist (Example)
- [ ] Feature works as expected
- [ ] No visual regressions
- [ ] Performance acceptable
- [ ] No console errors
- [ ] Mobile responsive (if applicable)

---

## Test Execution in CI/CD

### Pull Request Pipeline
```yaml
steps:
  - name: Unit Tests
    run: npm test  # or pytest

  - name: Lint
    run: npm run lint

  - name: Coverage Check
    run: npm run coverage
    threshold: 70%

  - name: Integration Tests
    run: npm run test:integration
```

### Deployment Pipeline
```yaml
steps:
  - name: Run All Tests
    run: npm run test:all

  - name: Deploy to Testing
    run: ./scripts/deploy-test.sh

  - name: Smoke Tests
    run: ./scripts/smoke-test.sh test

  - name: Notify Chat
    run: ./scripts/notify-testing-ready.sh
```

---

## Test Failure Protocol

### If Unit Tests Fail
1. AI agent identifies failing tests
2. AI agent fixes code or test
3. Re-run tests
4. Repeat until passing
5. Update CURRENT_STATE.md

### If Integration Tests Fail
1. Check for environment issues
2. Review recent changes
3. Fix integration points
4. Re-run full test suite

### If Human Validation Fails
1. AI agent receives feedback via chat
2. Creates bug fix task
3. Implements fix
4. Re-deploys to testing
5. Requests re-validation

---

## Test Documentation

Each project should maintain:
- `tests/README.md` - How to run tests locally
- Test naming convention: `test_{feature}_{scenario}`
- Clear test descriptions

---

## Assumptions

- [ASSUMPTION] Coverage threshold is 70%
- [ASSUMPTION] Jest for JavaScript/TypeScript, pytest for Python
- [ASSUMPTION] Test database is separate from dev/prod databases
- [ASSUMPTION] CI/CD handles test environment setup

---

*All testing requirements are mandatory. The AI agent cannot skip tests.*
