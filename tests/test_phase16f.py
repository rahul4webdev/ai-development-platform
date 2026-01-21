"""
Phase 16F: Intent Drift, Regression & Contract Enforcement Tests

Tests for:
- IntentBaselineManager: Immutable baseline storage
- IntentDriftEngine: Drift detection and classification
- IntentContractEnforcer: Contract enforcement model
- ExecutionGate integration: Drift checks in execution flow

Requirements: 25+ comprehensive tests
"""

import json
import os
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Import the modules under test
from controller.intent_baseline import (
    IntentBaselineManager,
    IntentBaseline,
    RebaselineRequest,
    BaselineStatus,
    RebaselineReason,
    create_initial_baseline,
    get_active_baseline,
    request_rebaseline,
    approve_rebaseline,
)
from controller.intent_drift_engine import (
    IntentDriftEngine,
    DriftLevel,
    DriftDimension,
    DriftAnalysisResult,
    DimensionDrift,
    DRIFT_THRESHOLDS,
    DIMENSION_WEIGHTS,
    analyze_drift,
    check_drift_blocks_execution,
)
from controller.intent_contract import (
    IntentContractEnforcer,
    ContractType,
    ViolationType,
    EnforcementAction,
    ContractViolation,
    ContractEvaluationResult,
    PendingConfirmation,
    evaluate_contract,
    check_can_execute,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_baseline_dir():
    """Create a temporary directory for baseline tests."""
    temp_dir = tempfile.mkdtemp(prefix="baseline_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_contract_dir():
    """Create a temporary directory for contract tests."""
    temp_dir = tempfile.mkdtemp(prefix="contract_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def baseline_manager(temp_baseline_dir):
    """Create a baseline manager with temp directory."""
    return IntentBaselineManager(baseline_dir=temp_baseline_dir)


@pytest.fixture
def drift_engine():
    """Create a drift engine instance."""
    return IntentDriftEngine()


@pytest.fixture
def contract_enforcer(temp_contract_dir):
    """Create a contract enforcer with temp directory."""
    return IntentContractEnforcer(contract_dir=temp_contract_dir)


@pytest.fixture
def sample_baseline_intent() -> Dict[str, Any]:
    """Sample normalized intent for baseline."""
    return {
        "purpose_keywords": ["task", "management", "todo"],
        "functional_modules": ["auth", "tasks", "notifications"],
        "architecture_class": "monolith",
        "database_type": "postgresql",
        "domain_topology": ["users", "tasks", "projects"],
        "target_users": ["developers", "teams"],
    }


@pytest.fixture
def sample_current_intent() -> Dict[str, Any]:
    """Sample normalized intent (current state, no drift)."""
    return {
        "purpose_keywords": ["task", "management", "todo"],
        "functional_modules": ["auth", "tasks", "notifications"],
        "architecture_class": "monolith",
        "database_type": "postgresql",
        "domain_topology": ["users", "tasks", "projects"],
        "target_users": ["developers", "teams"],
    }


@pytest.fixture
def drifted_intent_low() -> Dict[str, Any]:
    """Intent with LOW drift (only one minor addition)."""
    return {
        "purpose_keywords": ["task", "management", "todo"],
        "functional_modules": ["auth", "tasks", "notifications"],
        "architecture_class": "monolith",
        "database_type": "postgresql",
        "domain_topology": ["users", "tasks", "projects"],
        "target_users": ["developers", "teams", "admins"],  # Only minor user type addition
    }


@pytest.fixture
def drifted_intent_medium() -> Dict[str, Any]:
    """Intent with MEDIUM drift (moderate purpose expansion)."""
    return {
        "purpose_keywords": ["task", "management", "todo", "reporting"],  # Small expansion
        "functional_modules": ["auth", "tasks", "notifications"],
        "architecture_class": "monolith",
        "database_type": "postgresql",
        "domain_topology": ["users", "tasks", "projects", "reports"],  # One domain added
        "target_users": ["developers", "teams"],
    }


@pytest.fixture
def drifted_intent_high() -> Dict[str, Any]:
    """Intent with HIGH drift (architecture change only)."""
    return {
        "purpose_keywords": ["task", "management", "todo"],
        "functional_modules": ["auth", "tasks", "notifications"],
        "architecture_class": "microservices",  # Breaking change!
        "database_type": "postgresql",
        "domain_topology": ["users", "tasks", "projects"],
        "target_users": ["developers", "teams"],
    }


@pytest.fixture
def drifted_intent_critical() -> Dict[str, Any]:
    """Intent with CRITICAL drift (complete purpose change + architecture)."""
    return {
        "purpose_keywords": ["e-commerce", "shopping", "payments"],  # Completely different
        "functional_modules": ["cart", "checkout", "payments", "inventory"],
        "architecture_class": "microservices",
        "database_type": "mongodb",  # Also changed
        "domain_topology": ["products", "orders", "customers", "payments"],
        "target_users": ["shoppers", "merchants"],
    }


# =============================================================================
# IntentBaselineManager Tests (8 tests)
# =============================================================================

class TestIntentBaselineManager:
    """Tests for IntentBaselineManager."""

    def test_create_initial_baseline(self, baseline_manager, sample_baseline_intent):
        """Test creating an initial baseline for a new project."""
        success, message, baseline = baseline_manager.create_initial_baseline(
            project_id="proj-001",
            project_name="Test Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-abc123",
            created_by="user-001",
        )

        assert success is True
        assert baseline is not None
        assert baseline.version == 1
        assert baseline.status == BaselineStatus.ACTIVE.value
        assert baseline.normalized_intent == sample_baseline_intent
        assert baseline.creation_reason == "initial"

    def test_baseline_immutability(self, baseline_manager, sample_baseline_intent):
        """Test that baseline cannot be modified after creation."""
        success, _, baseline = baseline_manager.create_initial_baseline(
            project_id="proj-002",
            project_name="Immutable Test",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-def456",
            created_by="user-001",
        )

        # Attempting to create another initial baseline should fail
        success2, message2, _ = baseline_manager.create_initial_baseline(
            project_id="proj-002",
            project_name="Immutable Test",
            normalized_intent={"different": "intent"},
            fingerprint="fp-ghi789",
            created_by="user-001",
        )

        assert success2 is False
        assert "already has an active baseline" in message2

    def test_get_active_baseline(self, baseline_manager, sample_baseline_intent):
        """Test retrieving the active baseline for a project."""
        baseline_manager.create_initial_baseline(
            project_id="proj-003",
            project_name="Retrieval Test",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-jkl012",
            created_by="user-001",
        )

        active = baseline_manager.get_active_baseline("proj-003")
        assert active is not None
        assert active.is_active()
        assert active.project_id == "proj-003"

    def test_get_baseline_nonexistent(self, baseline_manager):
        """Test getting baseline for non-existent project returns None."""
        active = baseline_manager.get_active_baseline("nonexistent-project")
        assert active is None

    def test_request_rebaseline(self, baseline_manager, sample_baseline_intent, drifted_intent_medium):
        """Test requesting a rebaseline."""
        baseline_manager.create_initial_baseline(
            project_id="proj-004",
            project_name="Rebaseline Test",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-mno345",
            created_by="user-001",
        )

        success, message, request = baseline_manager.request_rebaseline(
            project_id="proj-004",
            project_name="Rebaseline Test",
            proposed_intent=drifted_intent_medium,
            proposed_fingerprint="fp-pqr678",
            reason=RebaselineReason.SCOPE_EXPANSION.value,
            justification="Adding analytics features as requested",
            requested_by="user-001",
        )

        assert success is True
        assert request is not None
        assert request.status == "pending"

    def test_approve_rebaseline(self, baseline_manager, sample_baseline_intent, drifted_intent_medium):
        """Test approving a rebaseline request."""
        baseline_manager.create_initial_baseline(
            project_id="proj-005",
            project_name="Approve Test",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-stu901",
            created_by="user-001",
        )

        _, _, request = baseline_manager.request_rebaseline(
            project_id="proj-005",
            project_name="Approve Test",
            proposed_intent=drifted_intent_medium,
            proposed_fingerprint="fp-vwx234",
            reason=RebaselineReason.SCOPE_EXPANSION.value,
            justification="Approved expansion",
            requested_by="user-001",
        )

        success, message, new_baseline = baseline_manager.approve_rebaseline(
            request_id=request.request_id,
            approved_by="admin-001",
            notes="Looks good",
        )

        assert success is True
        assert new_baseline is not None
        assert new_baseline.version == 2
        assert new_baseline.status == BaselineStatus.ACTIVE.value

        # Old baseline should be superseded
        old = baseline_manager.get_baseline_by_id(request.current_baseline_id)
        assert old.status == BaselineStatus.SUPERSEDED.value

    def test_reject_rebaseline(self, baseline_manager, sample_baseline_intent, drifted_intent_high):
        """Test rejecting a rebaseline request."""
        baseline_manager.create_initial_baseline(
            project_id="proj-006",
            project_name="Reject Test",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-yza567",
            created_by="user-001",
        )

        _, _, request = baseline_manager.request_rebaseline(
            project_id="proj-006",
            project_name="Reject Test",
            proposed_intent=drifted_intent_high,
            proposed_fingerprint="fp-bcd890",
            reason=RebaselineReason.ARCHITECTURE_UPGRADE.value,
            justification="Want to move to microservices",
            requested_by="user-001",
        )

        success, message = baseline_manager.reject_rebaseline(
            request_id=request.request_id,
            rejected_by="admin-001",
            notes="Not approved - too risky",
        )

        assert success is True

        # Active baseline should still be version 1
        active = baseline_manager.get_active_baseline("proj-006")
        assert active.version == 1

    def test_baseline_history(self, baseline_manager, sample_baseline_intent, drifted_intent_low):
        """Test getting baseline history."""
        baseline_manager.create_initial_baseline(
            project_id="proj-007",
            project_name="History Test",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-efg123",
            created_by="user-001",
        )

        _, _, request = baseline_manager.request_rebaseline(
            project_id="proj-007",
            project_name="History Test",
            proposed_intent=drifted_intent_low,
            proposed_fingerprint="fp-hij456",
            reason=RebaselineReason.REQUIREMENTS_CLARIFICATION.value,
            justification="Adding reports module",
            requested_by="user-001",
        )

        baseline_manager.approve_rebaseline(
            request_id=request.request_id,
            approved_by="admin-001",
        )

        history = baseline_manager.get_baseline_history("proj-007")
        assert len(history) == 2
        assert history[0].version == 1
        assert history[1].version == 2


# =============================================================================
# IntentDriftEngine Tests (10 tests)
# =============================================================================

class TestIntentDriftEngine:
    """Tests for IntentDriftEngine."""

    def test_no_drift(self, drift_engine, sample_baseline_intent, sample_current_intent):
        """Test detection of no drift (identical intents)."""
        result = drift_engine.analyze_drift(
            project_id="proj-001",
            baseline_id="baseline-001",
            baseline_intent=sample_baseline_intent,
            current_intent=sample_current_intent,
        )

        assert result.overall_level == DriftLevel.NONE.value
        assert result.overall_score <= DRIFT_THRESHOLDS[DriftLevel.NONE]
        assert result.blocks_execution is False
        assert result.requires_confirmation is False

    def test_low_drift(self, drift_engine, sample_baseline_intent):
        """Test detection of LOW drift (minor change)."""
        # Create a minimal change that produces LOW drift
        low_drift_intent = {
            "purpose_keywords": ["task", "management", "todo"],
            "functional_modules": ["auth", "tasks", "notifications", "logging"],  # Add one module
            "architecture_class": "monolith",
            "database_type": "postgresql",
            "domain_topology": ["users", "tasks", "projects"],
            "target_users": ["developers", "teams"],
        }
        result = drift_engine.analyze_drift(
            project_id="proj-002",
            baseline_id="baseline-002",
            baseline_intent=sample_baseline_intent,
            current_intent=low_drift_intent,
        )

        # LOW drift should not block or require confirmation
        assert result.overall_level in [DriftLevel.NONE.value, DriftLevel.LOW.value]
        assert result.blocks_execution is False
        assert result.requires_confirmation is False

    def test_medium_drift(self, drift_engine, sample_baseline_intent):
        """Test detection of MEDIUM drift (purpose expansion)."""
        # Create a moderate change - adding several purpose keywords
        medium_drift_intent = {
            "purpose_keywords": ["task", "management", "todo", "workflow", "automation", "scheduling"],  # Significant purpose expansion
            "functional_modules": ["auth", "tasks", "notifications", "scheduler", "workflows"],
            "architecture_class": "monolith",
            "database_type": "postgresql",
            "domain_topology": ["users", "tasks", "projects", "schedules", "workflows"],
            "target_users": ["developers", "teams", "managers"],
        }
        result = drift_engine.analyze_drift(
            project_id="proj-003",
            baseline_id="baseline-003",
            baseline_intent=sample_baseline_intent,
            current_intent=medium_drift_intent,
        )

        # Medium+ drift should require some action
        assert result.overall_score > DRIFT_THRESHOLDS[DriftLevel.LOW]
        assert result.requires_action is True

    def test_high_drift_architecture(self, drift_engine, sample_baseline_intent):
        """Test detection of HIGH drift from architecture change."""
        # Architecture change is a breaking change - expect HIGH or CRITICAL
        result = drift_engine.analyze_drift(
            project_id="proj-004",
            baseline_id="baseline-004",
            baseline_intent=sample_baseline_intent,
            current_intent={
                "purpose_keywords": ["task", "management", "todo"],
                "functional_modules": ["auth", "tasks", "notifications"],
                "architecture_class": "microservices",  # Breaking change!
                "database_type": "postgresql",
                "domain_topology": ["users", "tasks", "projects"],
                "target_users": ["developers", "teams"],
            },
        )

        # Breaking architecture change should block execution
        assert result.overall_level in [DriftLevel.HIGH.value, DriftLevel.CRITICAL.value]
        assert result.blocks_execution is True

    def test_critical_drift(self, drift_engine, sample_baseline_intent, drifted_intent_critical):
        """Test detection of CRITICAL drift."""
        result = drift_engine.analyze_drift(
            project_id="proj-005",
            baseline_id="baseline-005",
            baseline_intent=sample_baseline_intent,
            current_intent=drifted_intent_critical,
        )

        assert result.overall_level == DriftLevel.CRITICAL.value
        assert result.blocks_execution is True
        assert result.requires_rebaseline is True

    def test_purpose_drift_detection(self, drift_engine, sample_baseline_intent):
        """Test specific purpose drift detection."""
        current = {**sample_baseline_intent}
        current["purpose_keywords"] = ["task", "management", "todo", "crm", "sales"]  # Major expansion

        result = drift_engine.analyze_drift(
            project_id="proj-006",
            baseline_id="baseline-006",
            baseline_intent=sample_baseline_intent,
            current_intent=current,
        )

        purpose_drift = next(
            d for d in result.dimension_drifts
            if d.dimension == DriftDimension.PURPOSE.value
        )
        assert purpose_drift.score > 0
        assert "crm" in purpose_drift.added
        assert "sales" in purpose_drift.added

    def test_database_drift_detection(self, drift_engine, sample_baseline_intent):
        """Test database change drift detection."""
        current = {**sample_baseline_intent}
        current["database_type"] = "mongodb"  # Breaking change

        result = drift_engine.analyze_drift(
            project_id="proj-007",
            baseline_id="baseline-007",
            baseline_intent=sample_baseline_intent,
            current_intent=current,
        )

        db_drift = next(
            d for d in result.dimension_drifts
            if d.dimension == DriftDimension.DATABASE.value
        )
        assert db_drift.score > 50  # Breaking change has high score
        assert "BREAKING" in db_drift.explanation

    def test_module_drift_removal(self, drift_engine, sample_baseline_intent):
        """Test module removal drift detection."""
        current = {**sample_baseline_intent}
        current["functional_modules"] = ["auth", "tasks"]  # Removed notifications

        result = drift_engine.analyze_drift(
            project_id="proj-008",
            baseline_id="baseline-008",
            baseline_intent=sample_baseline_intent,
            current_intent=current,
        )

        module_drift = next(
            d for d in result.dimension_drifts
            if d.dimension == DriftDimension.MODULE.value
        )
        assert "notifications" in module_drift.removed

    def test_weighted_score_calculation(self, drift_engine, sample_baseline_intent):
        """Test that weighted score calculation is correct."""
        # Change only architecture (weight 0.25, score ~60 for non-breaking)
        current = {**sample_baseline_intent}
        current["architecture_class"] = "api_only"  # Non-breaking change

        result = drift_engine.analyze_drift(
            project_id="proj-009",
            baseline_id="baseline-009",
            baseline_intent=sample_baseline_intent,
            current_intent=current,
        )

        # With only architecture changed (weight 0.25, score 60), overall should be ~15
        arch_drift = next(
            d for d in result.dimension_drifts
            if d.dimension == DriftDimension.ARCHITECTURE.value
        )
        expected_contribution = arch_drift.score * DIMENSION_WEIGHTS[DriftDimension.ARCHITECTURE]
        assert abs(result.overall_score - expected_contribution) < 1  # Allow small rounding

    def test_check_drift_blocks_execution(self, sample_baseline_intent, drifted_intent_high):
        """Test the convenience function for blocking check."""
        blocks, reason, analysis = check_drift_blocks_execution(
            project_id="proj-010",
            baseline_intent=sample_baseline_intent,
            current_intent=drifted_intent_high,
        )

        assert blocks is True
        assert analysis is not None


# =============================================================================
# IntentContractEnforcer Tests (10 tests)
# =============================================================================

class TestIntentContractEnforcer:
    """Tests for IntentContractEnforcer."""

    def test_evaluate_no_baseline(self, contract_enforcer):
        """Test evaluation when no baseline exists."""
        result = contract_enforcer.evaluate_contract(
            project_id="new-project",
            project_name="New Project",
            current_intent={"purpose_keywords": ["test"]},
        )

        assert result.action == EnforcementAction.ALLOW.value
        assert result.can_proceed is True
        assert len(result.violations) == 0

    def test_evaluate_no_drift(
        self, contract_enforcer, baseline_manager,
        sample_baseline_intent, sample_current_intent,
        temp_baseline_dir
    ):
        """Test evaluation with no drift."""
        # Create baseline first
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)
        mgr.create_initial_baseline(
            project_id="proj-clean",
            project_name="Clean Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-clean",
            created_by="user-001",
        )

        # Use same temp dir for enforcer
        enforcer = IntentContractEnforcer(contract_dir=temp_baseline_dir)

        # Patch the baseline manager's directory
        from controller import intent_baseline
        original_manager = intent_baseline._manager
        intent_baseline._manager = mgr

        try:
            result = enforcer.evaluate_contract(
                project_id="proj-clean",
                project_name="Clean Project",
                current_intent=sample_current_intent,
            )

            assert result.action == EnforcementAction.ALLOW.value
            assert result.can_proceed is True
        finally:
            intent_baseline._manager = original_manager

    def test_evaluate_high_drift_blocks(
        self, temp_baseline_dir,
        sample_baseline_intent
    ):
        """Test that HIGH drift blocks execution."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)
        mgr.create_initial_baseline(
            project_id="proj-block",
            project_name="Block Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-block",
            created_by="user-001",
        )

        # Use a breaking architecture change
        high_drift_intent = {
            "purpose_keywords": ["task", "management", "todo"],
            "functional_modules": ["auth", "tasks", "notifications"],
            "architecture_class": "microservices",  # Breaking change!
            "database_type": "postgresql",
            "domain_topology": ["users", "tasks", "projects"],
            "target_users": ["developers", "teams"],
        }

        from controller import intent_baseline
        original_manager = intent_baseline._manager
        intent_baseline._manager = mgr

        try:
            enforcer = IntentContractEnforcer(contract_dir=temp_baseline_dir)
            result = enforcer.evaluate_contract(
                project_id="proj-block",
                project_name="Block Project",
                current_intent=high_drift_intent,
            )

            # Should either BLOCK or FREEZE - both prevent execution
            assert result.action in [EnforcementAction.BLOCK.value, EnforcementAction.FREEZE.value]
            assert result.can_proceed is False
        finally:
            intent_baseline._manager = original_manager

    def test_evaluate_medium_drift_requires_confirmation(
        self, temp_baseline_dir,
        sample_baseline_intent, drifted_intent_medium
    ):
        """Test that MEDIUM drift requires confirmation."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)
        mgr.create_initial_baseline(
            project_id="proj-confirm",
            project_name="Confirm Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-confirm",
            created_by="user-001",
        )

        from controller import intent_baseline
        original_manager = intent_baseline._manager
        intent_baseline._manager = mgr

        try:
            enforcer = IntentContractEnforcer(contract_dir=temp_baseline_dir)
            result = enforcer.evaluate_contract(
                project_id="proj-confirm",
                project_name="Confirm Project",
                current_intent=drifted_intent_medium,
            )

            assert result.action == EnforcementAction.CONFIRM.value
            assert result.can_proceed is False
            assert result.requires_confirmation is True
            assert result.user_prompt is not None
        finally:
            intent_baseline._manager = original_manager

    def test_create_confirmation_request(self, contract_enforcer):
        """Test creating a pending confirmation request."""
        # Create a mock evaluation result
        mock_result = ContractEvaluationResult(
            project_id="proj-test",
            project_name="Test Project",
            baseline_id="baseline-test",
            evaluation_timestamp=datetime.utcnow().isoformat(),
            action=EnforcementAction.CONFIRM.value,
            can_proceed=False,
            requires_confirmation=True,
            drift_analysis=None,
            violations=[],
            summary="Test confirmation",
            user_prompt="Do you approve?",
            block_reason=None,
        )

        confirmation = contract_enforcer.create_confirmation_request(
            evaluation_result=mock_result,
            timeout_hours=24,
        )

        assert confirmation is not None
        assert confirmation.status == "pending"
        assert confirmation.project_id == "proj-test"

    def test_approve_confirmation(self, contract_enforcer):
        """Test approving a confirmation request."""
        mock_result = ContractEvaluationResult(
            project_id="proj-approve",
            project_name="Approve Project",
            baseline_id="baseline-approve",
            evaluation_timestamp=datetime.utcnow().isoformat(),
            action=EnforcementAction.CONFIRM.value,
            can_proceed=False,
            requires_confirmation=True,
            drift_analysis=None,
            violations=[],
            summary="Test approval",
            user_prompt="Approve?",
            block_reason=None,
        )

        confirmation = contract_enforcer.create_confirmation_request(mock_result)

        success, message = contract_enforcer.approve_confirmation(
            confirmation_id=confirmation.confirmation_id,
            approved_by="admin-001",
            notes="Approved",
        )

        assert success is True

        # Verify status changed
        pending = contract_enforcer.get_pending_confirmations("proj-approve")
        assert len(pending) == 0  # No longer pending

    def test_reject_confirmation(self, contract_enforcer):
        """Test rejecting a confirmation request."""
        mock_result = ContractEvaluationResult(
            project_id="proj-reject",
            project_name="Reject Project",
            baseline_id="baseline-reject",
            evaluation_timestamp=datetime.utcnow().isoformat(),
            action=EnforcementAction.CONFIRM.value,
            can_proceed=False,
            requires_confirmation=True,
            drift_analysis=None,
            violations=[],
            summary="Test rejection",
            user_prompt="Approve?",
            block_reason=None,
        )

        confirmation = contract_enforcer.create_confirmation_request(mock_result)

        success, message = contract_enforcer.reject_confirmation(
            confirmation_id=confirmation.confirmation_id,
            rejected_by="admin-001",
            notes="Not approved",
        )

        assert success is True

    def test_confirmation_not_found(self, contract_enforcer):
        """Test handling of non-existent confirmation."""
        success, message = contract_enforcer.approve_confirmation(
            confirmation_id="nonexistent",
            approved_by="admin-001",
        )

        assert success is False
        assert "not found" in message.lower()

    def test_get_pending_confirmations(self, contract_enforcer):
        """Test getting pending confirmations."""
        # Create two confirmations
        for i in range(2):
            mock_result = ContractEvaluationResult(
                project_id=f"proj-pending-{i}",
                project_name=f"Pending Project {i}",
                baseline_id=f"baseline-pending-{i}",
                evaluation_timestamp=datetime.utcnow().isoformat(),
                action=EnforcementAction.CONFIRM.value,
                can_proceed=False,
                requires_confirmation=True,
                drift_analysis=None,
                violations=[],
                summary="Test",
                user_prompt="Approve?",
                block_reason=None,
            )
            contract_enforcer.create_confirmation_request(mock_result)

        pending = contract_enforcer.get_pending_confirmations()
        assert len(pending) >= 2

    def test_check_can_execute(self, temp_baseline_dir, sample_baseline_intent):
        """Test the convenience check_can_execute function."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)
        mgr.create_initial_baseline(
            project_id="proj-exec",
            project_name="Exec Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-exec",
            created_by="user-001",
        )

        from controller import intent_baseline
        original_manager = intent_baseline._manager
        intent_baseline._manager = mgr

        try:
            from controller.intent_contract import get_contract_enforcer
            # Reset global enforcer
            import controller.intent_contract
            controller.intent_contract._enforcer = IntentContractEnforcer(contract_dir=temp_baseline_dir)

            can_proceed, reason, result = check_can_execute(
                project_id="proj-exec",
                project_name="Exec Project",
                current_intent=sample_baseline_intent,
            )

            assert can_proceed is True
        finally:
            intent_baseline._manager = original_manager
            controller.intent_contract._enforcer = None


# =============================================================================
# Integration Tests (5 tests)
# =============================================================================

class TestIntegration:
    """Integration tests for the complete Phase 16F flow."""

    def test_full_baseline_drift_contract_flow(self, temp_baseline_dir, sample_baseline_intent):
        """Test complete flow: baseline -> drift -> contract evaluation."""
        # 1. Create baseline
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)
        success, _, baseline = mgr.create_initial_baseline(
            project_id="proj-full",
            project_name="Full Flow Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-full",
            created_by="user-001",
        )
        assert success is True

        # 2. Analyze drift (no drift)
        engine = IntentDriftEngine()
        drift_result = engine.analyze_drift(
            project_id="proj-full",
            baseline_id=baseline.baseline_id,
            baseline_intent=sample_baseline_intent,
            current_intent=sample_baseline_intent,
        )
        assert drift_result.overall_level == DriftLevel.NONE.value

        # 3. Evaluate contract (should allow)
        from controller import intent_baseline
        original_manager = intent_baseline._manager
        intent_baseline._manager = mgr

        try:
            enforcer = IntentContractEnforcer(contract_dir=temp_baseline_dir)
            contract_result = enforcer.evaluate_contract(
                project_id="proj-full",
                project_name="Full Flow Project",
                current_intent=sample_baseline_intent,
            )
            assert contract_result.can_proceed is True
        finally:
            intent_baseline._manager = original_manager

    def test_rebaseline_resolves_drift(
        self, temp_baseline_dir,
        sample_baseline_intent, drifted_intent_high
    ):
        """Test that approving rebaseline resolves drift blocking."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)

        # Create initial baseline
        mgr.create_initial_baseline(
            project_id="proj-resolve",
            project_name="Resolve Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-resolve",
            created_by="user-001",
        )

        # Request rebaseline with high-drift intent
        _, _, request = mgr.request_rebaseline(
            project_id="proj-resolve",
            project_name="Resolve Project",
            proposed_intent=drifted_intent_high,
            proposed_fingerprint="fp-resolve-new",
            reason=RebaselineReason.ARCHITECTURE_UPGRADE.value,
            justification="Migrating to microservices",
            requested_by="user-001",
        )

        # Approve rebaseline
        mgr.approve_rebaseline(
            request_id=request.request_id,
            approved_by="admin-001",
        )

        # Now drift analysis should show no drift against new baseline
        active = mgr.get_active_baseline("proj-resolve")
        engine = IntentDriftEngine()

        drift_result = engine.analyze_drift(
            project_id="proj-resolve",
            baseline_id=active.baseline_id,
            baseline_intent=active.normalized_intent,
            current_intent=drifted_intent_high,
        )

        assert drift_result.overall_level == DriftLevel.NONE.value

    def test_audit_trail_completeness(self, temp_baseline_dir, sample_baseline_intent):
        """Test that all operations are logged to audit trail."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)

        # Perform operations
        mgr.create_initial_baseline(
            project_id="proj-audit",
            project_name="Audit Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-audit",
            created_by="user-001",
        )

        # Check audit log exists and contains entry
        audit_file = temp_baseline_dir / "baseline_audit.log"
        assert audit_file.exists()

        with open(audit_file) as f:
            content = f.read()
            assert "BASELINE_CREATED" in content
            assert "proj-audit" in content

    def test_violation_detection_accuracy(
        self, temp_baseline_dir,
        sample_baseline_intent, drifted_intent_high
    ):
        """Test that violations are accurately detected and categorized."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)
        mgr.create_initial_baseline(
            project_id="proj-viol",
            project_name="Violation Project",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-viol",
            created_by="user-001",
        )

        from controller import intent_baseline
        original_manager = intent_baseline._manager
        intent_baseline._manager = mgr

        try:
            enforcer = IntentContractEnforcer(contract_dir=temp_baseline_dir)
            result = enforcer.evaluate_contract(
                project_id="proj-viol",
                project_name="Violation Project",
                current_intent=drifted_intent_high,
            )

            # Should have architecture violation
            arch_violations = [
                v for v in result.violations
                if v.violation_type == ViolationType.ARCHITECTURE_CHANGE.value
            ]
            assert len(arch_violations) > 0
            assert arch_violations[0].contract_type == ContractType.HARD_BLOCK.value
        finally:
            intent_baseline._manager = original_manager

    def test_concurrent_baseline_operations(self, temp_baseline_dir, sample_baseline_intent):
        """Test handling of concurrent baseline operations."""
        mgr = IntentBaselineManager(baseline_dir=temp_baseline_dir)

        # Create baseline for two different projects
        mgr.create_initial_baseline(
            project_id="proj-a",
            project_name="Project A",
            normalized_intent=sample_baseline_intent,
            fingerprint="fp-a",
            created_by="user-001",
        )

        mgr.create_initial_baseline(
            project_id="proj-b",
            project_name="Project B",
            normalized_intent={**sample_baseline_intent, "purpose_keywords": ["different"]},
            fingerprint="fp-b",
            created_by="user-002",
        )

        # Both should have separate baselines
        a_baseline = mgr.get_active_baseline("proj-a")
        b_baseline = mgr.get_active_baseline("proj-b")

        assert a_baseline is not None
        assert b_baseline is not None
        assert a_baseline.baseline_id != b_baseline.baseline_id
        assert a_baseline.project_id == "proj-a"
        assert b_baseline.project_id == "proj-b"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
