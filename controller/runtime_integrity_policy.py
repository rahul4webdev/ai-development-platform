"""
Phase 26.6: Runtime Integrity Policy

Defines the enforcement policy for runtime integrity checks.
This is the ONLY place that defines policy semantics.

| RuntimeIntegrityStatus | WARN_ONLY   | BLOCK_CRITICAL (DEFAULT) | STRICT      |
| ---------------------- | ----------- | ------------------------ | ----------- |
| HEALTHY                | Allow       | Allow                    | Allow       |
| DEGRADED               | Allow + log | Allow + log              | Allow + log |
| FATAL                  | Allow + log | BLOCK                    | BLOCK       |
"""

from enum import Enum


class RuntimeIntegrityPolicy(str, Enum):
    """Runtime integrity enforcement policy. Default: BLOCK_CRITICAL."""
    WARN_ONLY = "warn_only"
    BLOCK_CRITICAL = "block_critical"
    STRICT = "strict"


# Default policy — MUST be BLOCK_CRITICAL
RUNTIME_INTEGRITY_POLICY: RuntimeIntegrityPolicy = RuntimeIntegrityPolicy.BLOCK_CRITICAL


def get_policy() -> RuntimeIntegrityPolicy:
    """Get the current runtime integrity policy."""
    return RUNTIME_INTEGRITY_POLICY


def set_policy(policy: RuntimeIntegrityPolicy) -> None:
    """Set the runtime integrity policy (owner/admin only at API layer)."""
    global RUNTIME_INTEGRITY_POLICY
    RUNTIME_INTEGRITY_POLICY = policy
