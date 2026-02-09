"""
Phase 26.5: Platform Runtime Integrity Layer (PRIL)

Verifies the platform's own runtime environment before any meaningful work.
Checks: venv, required packages, Playwright, required binaries, filesystem permissions.

Status derivation (deterministic):
- FATAL if venv_ok/playwright_ok/fs_permissions_ok is False OR missing_packages non-empty
- DEGRADED if missing_binaries non-empty
- HEALTHY otherwise
"""

import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

logger = logging.getLogger("runtime_integrity")

# -----------------------------------------------------------------------------
# Constants (LOCKED — do not change without dual approval)
# -----------------------------------------------------------------------------
EXPECTED_VENV = Path("/home/aitesting.mybd.in/public_html/venv")

REQUIRED_PACKAGES = ("aiohttp", "requests", "playwright", "pydantic", "fastapi")

REQUIRED_BINARIES = ("curl", "git", "rsync", "ssh", "node", "npm")

REQUIRED_WRITABLE_PATHS = (
    Path("/home/aitesting.mybd.in/jobs/registry"),
    Path("/home/aitesting.mybd.in/public_html/data"),
)


# -----------------------------------------------------------------------------
# RuntimeIntegrityReport (frozen dataclass)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeIntegrityReport:
    """Immutable snapshot of the platform's runtime integrity."""

    timestamp: str
    python_executable: str
    venv_ok: bool
    missing_packages: tuple  # Tuple[str, ...]
    playwright_ok: bool
    missing_binaries: tuple  # Tuple[str, ...]
    fs_permissions_ok: bool
    status: Literal["HEALTHY", "DEGRADED", "FATAL"]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "python_executable": self.python_executable,
            "venv_ok": self.venv_ok,
            "missing_packages": list(self.missing_packages),
            "playwright_ok": self.playwright_ok,
            "missing_binaries": list(self.missing_binaries),
            "fs_permissions_ok": self.fs_permissions_ok,
            "status": self.status,
        }


# -----------------------------------------------------------------------------
# RuntimeIntegrityChecker
# -----------------------------------------------------------------------------
class RuntimeIntegrityChecker:
    """
    Performs 5 sequential checks on the platform runtime.

    Check order:
    1. Venv existence and Python executable
    2. Required packages importable
    3. Playwright installed
    4. Required system binaries on PATH
    5. Filesystem write permissions
    """

    def run(self) -> RuntimeIntegrityReport:
        """Execute all checks in strict order, return frozen report."""
        timestamp = datetime.now(timezone.utc).isoformat()
        python_executable = sys.executable

        # 1. Venv check
        venv_ok = self._check_venv()

        # 2. Required packages
        missing_packages = self._check_packages()

        # 3. Playwright
        playwright_ok = self._check_playwright()

        # 4. Required binaries
        missing_binaries = self._check_binaries()

        # 5. Filesystem permissions
        fs_permissions_ok = self._check_fs_permissions()

        # Derive status
        status = self._derive_status(
            venv_ok=venv_ok,
            missing_packages=missing_packages,
            playwright_ok=playwright_ok,
            missing_binaries=missing_binaries,
            fs_permissions_ok=fs_permissions_ok,
        )

        report = RuntimeIntegrityReport(
            timestamp=timestamp,
            python_executable=python_executable,
            venv_ok=venv_ok,
            missing_packages=tuple(missing_packages),
            playwright_ok=playwright_ok,
            missing_binaries=tuple(missing_binaries),
            fs_permissions_ok=fs_permissions_ok,
            status=status,
        )

        logger.info(
            f"Runtime integrity check: status={report.status}, "
            f"venv_ok={venv_ok}, missing_pkg={missing_packages}, "
            f"playwright_ok={playwright_ok}, missing_bin={missing_binaries}, "
            f"fs_ok={fs_permissions_ok}"
        )

        return report

    # ----- Check 1: Venv -----
    def _check_venv(self) -> bool:
        """Verify we are running inside the expected venv."""
        try:
            if not EXPECTED_VENV.exists():
                logger.warning(f"Expected venv does not exist: {EXPECTED_VENV}")
                return False

            venv_python = EXPECTED_VENV / "bin" / "python"
            if not venv_python.exists():
                logger.warning(f"Venv python not found: {venv_python}")
                return False

            # Check current executable is inside the venv
            current = Path(sys.executable).resolve()
            expected_prefix = EXPECTED_VENV.resolve()
            if str(current).startswith(str(expected_prefix)):
                return True

            # Fallback: accept if VIRTUAL_ENV env var matches
            virtual_env = os.environ.get("VIRTUAL_ENV", "")
            if virtual_env and Path(virtual_env).resolve() == expected_prefix:
                return True

            logger.warning(
                f"Running outside expected venv: "
                f"current={current}, expected_prefix={expected_prefix}"
            )
            return False
        except Exception as e:
            logger.error(f"Venv check error: {e}")
            return False

    # ----- Check 2: Required packages -----
    def _check_packages(self) -> List[str]:
        """Check that all required packages are importable."""
        missing = []
        for pkg in REQUIRED_PACKAGES:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)
                logger.warning(f"Required package not importable: {pkg}")
        return missing

    # ----- Check 3: Playwright -----
    def _check_playwright(self) -> bool:
        """Check that Playwright is installed and has at least one browser."""
        try:
            import playwright  # noqa: F401
        except ImportError:
            logger.warning("Playwright package not installed")
            return False

        try:
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "--dry-run"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            # If dry-run succeeds or mentions browsers, it's installed
            if result.returncode == 0:
                return True
            # Fallback: check if chromium binary exists in common location
            playwright_path = Path.home() / ".cache" / "ms-playwright"
            if playwright_path.exists() and any(playwright_path.iterdir()):
                return True
            logger.warning(f"Playwright dry-run failed: {result.stderr[:200]}")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Playwright check timed out")
            return False
        except Exception as e:
            logger.warning(f"Playwright check error: {e}")
            return False

    # ----- Check 4: Required binaries -----
    def _check_binaries(self) -> List[str]:
        """Check that all required system binaries are on PATH."""
        missing = []
        for binary in REQUIRED_BINARIES:
            if shutil.which(binary) is None:
                missing.append(binary)
                logger.warning(f"Required binary not found: {binary}")
        return missing

    # ----- Check 5: Filesystem permissions -----
    def _check_fs_permissions(self) -> bool:
        """Verify write access to required paths."""
        all_ok = True
        for path in REQUIRED_WRITABLE_PATHS:
            if not path.exists():
                logger.warning(f"Required writable path does not exist: {path}")
                all_ok = False
                continue
            try:
                # Attempt to create and immediately remove a temp file
                test_file = path / f".pril_check_{os.getpid()}"
                test_file.write_text("integrity_check")
                test_file.unlink()
            except (PermissionError, OSError) as e:
                logger.warning(f"No write access to {path}: {e}")
                all_ok = False
        return all_ok

    # ----- Status derivation -----
    @staticmethod
    def _derive_status(
        venv_ok: bool,
        missing_packages: List[str],
        playwright_ok: bool,
        missing_binaries: List[str],
        fs_permissions_ok: bool,
    ) -> Literal["HEALTHY", "DEGRADED", "FATAL"]:
        """
        Deterministic status derivation.

        FATAL if: venv_ok=False OR playwright_ok=False OR fs_permissions_ok=False
                  OR missing_packages non-empty
        DEGRADED if: missing_binaries non-empty
        HEALTHY otherwise
        """
        if not venv_ok:
            return "FATAL"
        if missing_packages:
            return "FATAL"
        if not playwright_ok:
            return "FATAL"
        if not fs_permissions_ok:
            return "FATAL"
        if missing_binaries:
            return "DEGRADED"
        return "HEALTHY"


# -----------------------------------------------------------------------------
# Singleton
# -----------------------------------------------------------------------------
_checker: Optional[RuntimeIntegrityChecker] = None
_last_report: Optional[RuntimeIntegrityReport] = None


def get_runtime_integrity_checker() -> RuntimeIntegrityChecker:
    """Get the singleton RuntimeIntegrityChecker."""
    global _checker
    if _checker is None:
        _checker = RuntimeIntegrityChecker()
    return _checker


def get_last_integrity_report() -> Optional[RuntimeIntegrityReport]:
    """Get the last cached integrity report (may be None if never run)."""
    return _last_report


def run_integrity_check() -> RuntimeIntegrityReport:
    """Run an integrity check and cache the result."""
    global _last_report
    checker = get_runtime_integrity_checker()
    _last_report = checker.run()
    return _last_report
