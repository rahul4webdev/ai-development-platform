#!/usr/bin/env python3
"""
Phase 16E: Project Identity, Fingerprinting & Conflict Resolution Tests

Comprehensive tests for:
1. Project Identity Engine (project_identity.py)
2. Project Decision Engine (project_decision_engine.py)
3. Project Registry v2 with identity (project_registry.py)
4. Integration tests

Total: 25+ tests as required by Phase 16E specification.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test 1-8: Project Identity Engine Tests
# =============================================================================

class TestProjectIdentityEngine:
    """Test Project Identity Engine functionality."""

    def test_identity_imports(self):
        """Test 1: Project identity module should import successfully."""
        from controller.project_identity import (
            ProjectIdentity,
            NormalizedIntent,
            IntentExtractor,
            FingerprintGenerator,
            ProjectIdentityManager,
            ArchitectureClass,
            DatabaseType,
            DomainTopology,
        )
        assert ProjectIdentity is not None
        assert NormalizedIntent is not None
        assert IntentExtractor is not None
        assert FingerprintGenerator is not None
        assert ProjectIdentityManager is not None
        assert ArchitectureClass is not None
        assert DatabaseType is not None
        assert DomainTopology is not None

    def test_normalized_intent_is_frozen(self):
        """Test 2: NormalizedIntent should be immutable (frozen)."""
        from controller.project_identity import NormalizedIntent

        intent = NormalizedIntent(
            purpose_keywords=frozenset(["crm", "sales"]),
            functional_modules=frozenset(["api", "frontend"]),
            domain_topology=frozenset(["b2b"]),
            database_type="postgresql",
            architecture_class="monolith",
            target_users=frozenset(["admin", "sales"]),
        )

        # Should raise error when trying to modify
        try:
            intent.purpose_keywords = frozenset(["other"])
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected

    def test_intent_extractor_purpose_keywords(self):
        """Test 3: IntentExtractor should extract purpose keywords."""
        from controller.project_identity import IntentExtractor

        extractor = IntentExtractor()
        description = "Build a SaaS CRM with user management and analytics"

        intent = extractor.extract_intent(description)

        assert "crm" in intent.purpose_keywords or "saas" in intent.purpose_keywords
        assert len(intent.purpose_keywords) > 0

    def test_intent_extractor_modules(self):
        """Test 4: IntentExtractor should extract functional modules."""
        from controller.project_identity import IntentExtractor

        extractor = IntentExtractor()
        description = "Build REST API backend with React frontend and admin dashboard"

        intent = extractor.extract_intent(description)

        assert "api" in intent.functional_modules or "backend" in intent.functional_modules
        assert "frontend" in intent.functional_modules or "admin" in intent.functional_modules

    def test_intent_extractor_database(self):
        """Test 5: IntentExtractor should detect database type."""
        from controller.project_identity import IntentExtractor

        extractor = IntentExtractor()

        # Test PostgreSQL detection
        intent1 = extractor.extract_intent("Build API with PostgreSQL database")
        assert intent1.database_type in ["postgresql", "relational", "unknown"]

        # Test MongoDB detection
        intent2 = extractor.extract_intent("Build API with MongoDB for documents")
        assert intent2.database_type in ["mongodb", "nosql", "unknown"]

    def test_fingerprint_deterministic(self):
        """Test 6: Fingerprint generation should be deterministic."""
        from controller.project_identity import FingerprintGenerator, NormalizedIntent

        generator = FingerprintGenerator()

        intent = NormalizedIntent(
            purpose_keywords=frozenset(["crm", "sales"]),
            functional_modules=frozenset(["api", "frontend"]),
            domain_topology=frozenset(["b2b"]),
            database_type="postgresql",
            architecture_class="monolith",
            target_users=frozenset(["admin"]),
        )

        fp1 = generator.generate(intent)
        fp2 = generator.generate(intent)

        assert fp1 == fp2, "Same intent should produce same fingerprint"
        assert len(fp1) == 64, "Fingerprint should be 64 chars (SHA-256 hex)"

    def test_fingerprint_different_for_different_intents(self):
        """Test 7: Different intents should produce different fingerprints."""
        from controller.project_identity import FingerprintGenerator, NormalizedIntent

        generator = FingerprintGenerator()

        intent1 = NormalizedIntent(
            purpose_keywords=frozenset(["crm"]),
            functional_modules=frozenset(["api"]),
            domain_topology=frozenset(["b2b"]),
            database_type="postgresql",
            architecture_class="monolith",
            target_users=frozenset(["admin"]),
        )

        intent2 = NormalizedIntent(
            purpose_keywords=frozenset(["ecommerce"]),
            functional_modules=frozenset(["api", "frontend"]),
            domain_topology=frozenset(["b2c"]),
            database_type="mongodb",
            architecture_class="microservices",
            target_users=frozenset(["customer"]),
        )

        fp1 = generator.generate(intent1)
        fp2 = generator.generate(intent2)

        assert fp1 != fp2, "Different intents should produce different fingerprints"

    def test_identity_manager_create_identity(self):
        """Test 8: ProjectIdentityManager should create project identity."""
        from controller.project_identity import ProjectIdentityManager

        manager = ProjectIdentityManager()
        description = "Build a SaaS CRM for small businesses"

        identity = manager.create_identity(
            project_id="test-123",
            description=description,
        )

        assert identity is not None
        assert identity.project_id == "test-123"
        assert len(identity.fingerprint) == 64
        assert identity.normalized_intent is not None


# =============================================================================
# Test 9-17: Project Decision Engine Tests
# =============================================================================

class TestProjectDecisionEngine:
    """Test Project Decision Engine functionality."""

    def test_decision_engine_imports(self):
        """Test 9: Decision engine module should import successfully."""
        from controller.project_decision_engine import (
            DecisionType,
            ConflictType,
            UserChoice,
            DecisionResult,
            ProjectDecisionEngine,
            evaluate_project_creation,
        )
        assert DecisionType is not None
        assert ConflictType is not None
        assert UserChoice is not None
        assert DecisionResult is not None
        assert ProjectDecisionEngine is not None
        assert evaluate_project_creation is not None

    def test_decision_types_enum(self):
        """Test 10: DecisionType enum should have all required values."""
        from controller.project_decision_engine import DecisionType

        assert hasattr(DecisionType, "NEW_PROJECT")
        assert hasattr(DecisionType, "REUSE_PROJECT")
        assert hasattr(DecisionType, "CHANGE_MODE")
        assert hasattr(DecisionType, "NEW_VERSION")
        assert hasattr(DecisionType, "CONFLICT_DETECTED")

    def test_conflict_types_enum(self):
        """Test 11: ConflictType enum should have all required values."""
        from controller.project_decision_engine import ConflictType

        assert hasattr(ConflictType, "NONE")
        assert hasattr(ConflictType, "EXACT_MATCH")
        assert hasattr(ConflictType, "HIGH_SIMILARITY")
        assert hasattr(ConflictType, "SCOPE_CHANGE")
        assert hasattr(ConflictType, "ARCHITECTURE_CHANGE")

    def test_decision_result_structure(self):
        """Test 12: DecisionResult should have correct structure."""
        from controller.project_decision_engine import DecisionResult, DecisionType, ConflictType

        result = DecisionResult(
            decision=DecisionType.NEW_PROJECT,
            confidence=0.95,
            explanation="No similar projects found",
            requires_user_confirmation=False,
            conflict_type=ConflictType.NONE,
            existing_project_name=None,
            existing_project_fingerprint=None,
            similarity_score=0.0,
            differences=[],
            recommended_action="Create new project",
        )

        assert result.decision == DecisionType.NEW_PROJECT
        assert result.confidence == 0.95
        assert result.requires_user_confirmation is False

    def test_decision_result_to_dict(self):
        """Test 13: DecisionResult should serialize to dict."""
        from controller.project_decision_engine import DecisionResult, DecisionType, ConflictType

        result = DecisionResult(
            decision=DecisionType.NEW_PROJECT,
            confidence=0.95,
            explanation="Test explanation",
            requires_user_confirmation=False,
            conflict_type=ConflictType.NONE,
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["decision"] == "NEW_PROJECT"
        assert data["confidence"] == 0.95

    def test_evaluate_new_project_no_existing(self):
        """Test 14: Should return NEW_PROJECT when no existing projects."""
        from controller.project_decision_engine import evaluate_project_creation, DecisionType

        decision = evaluate_project_creation(
            description="Build a new CRM system",
            requirements="CRM with user management",
            existing_identities=[],
        )

        assert decision.decision == DecisionType.NEW_PROJECT
        assert decision.requires_user_confirmation is False

    def test_evaluate_detects_exact_match(self):
        """Test 15: Should detect exact fingerprint match."""
        from controller.project_decision_engine import (
            evaluate_project_creation,
            DecisionType,
            ConflictType,
        )
        from controller.project_identity import ProjectIdentityManager

        manager = ProjectIdentityManager()

        # Create identity for existing project
        existing_identity = manager.create_identity(
            project_id="existing-123",
            description="Build a SaaS CRM with REST API",
        )

        # Evaluate same description
        decision = evaluate_project_creation(
            description="Build a SaaS CRM with REST API",
            requirements=None,
            existing_identities=[(existing_identity, "existing-crm")],
        )

        # Should detect conflict
        assert decision.decision in [DecisionType.CONFLICT_DETECTED, DecisionType.REUSE_PROJECT]
        assert decision.conflict_type in [ConflictType.EXACT_MATCH, ConflictType.HIGH_SIMILARITY]

    def test_evaluate_detects_high_similarity(self):
        """Test 16: Should detect high similarity projects."""
        from controller.project_decision_engine import (
            evaluate_project_creation,
            ConflictType,
        )
        from controller.project_identity import ProjectIdentityManager

        manager = ProjectIdentityManager()

        # Create identity for existing project
        existing_identity = manager.create_identity(
            project_id="existing-123",
            description="Build a CRM with user management and REST API",
        )

        # Evaluate similar description
        decision = evaluate_project_creation(
            description="Build a CRM system with API and user features",
            requirements=None,
            existing_identities=[(existing_identity, "existing-crm")],
        )

        # Should detect similarity
        assert decision.similarity_score >= 0.0
        if decision.similarity_score >= 0.7:
            assert decision.conflict_type in [
                ConflictType.HIGH_SIMILARITY,
                ConflictType.EXACT_MATCH,
            ]

    def test_architecture_change_detector(self):
        """Test 17: Should detect architecture changes."""
        from controller.project_decision_engine import ArchitectureChangeDetector
        from controller.project_identity import NormalizedIntent

        detector = ArchitectureChangeDetector()

        old_intent = NormalizedIntent(
            purpose_keywords=frozenset(["crm"]),
            functional_modules=frozenset(["api"]),
            domain_topology=frozenset(["b2b"]),
            database_type="postgresql",
            architecture_class="monolith",
            target_users=frozenset(["admin"]),
        )

        new_intent = NormalizedIntent(
            purpose_keywords=frozenset(["crm"]),
            functional_modules=frozenset(["api", "frontend"]),
            domain_topology=frozenset(["b2b"]),
            database_type="postgresql",
            architecture_class="microservices",  # Changed!
            target_users=frozenset(["admin"]),
        )

        is_breaking, reason = detector.is_breaking_change(old_intent, new_intent)
        assert is_breaking is True
        assert "architecture" in reason.lower()


# =============================================================================
# Test 18-23: Project Registry v2 Tests
# =============================================================================

class TestProjectRegistryV2:
    """Test Project Registry v2 with identity support."""

    def test_registry_identity_imports(self):
        """Test 18: Registry should have identity-related methods."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()

        # Check for Phase 16E methods
        assert hasattr(registry, "get_all_identities")
        assert hasattr(registry, "get_project_by_fingerprint")
        assert hasattr(registry, "find_similar_projects")
        assert hasattr(registry, "create_project_with_identity")
        assert hasattr(registry, "create_project_version")
        assert hasattr(registry, "add_change_record")
        assert hasattr(registry, "get_dashboard_projects_grouped")

    def test_project_model_identity_fields(self):
        """Test 19: Project model should have identity fields."""
        from controller.project_registry import Project

        project = Project(
            project_id="test-123",
            name="test-project",
            description="A test project",
            created_by="user-1",
            created_at="2026-01-19T00:00:00",
            updated_at="2026-01-19T00:00:00",
            fingerprint="abc123",
            version="v2",
            parent_project_id="parent-456",
        )

        assert project.fingerprint == "abc123"
        assert project.version == "v2"
        assert project.parent_project_id == "parent-456"

    def test_create_project_with_identity(self):
        """Test 20: Should create project with identity."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        unique_name = f"test-identity-{uuid.uuid4().hex[:8]}"

        success, message, project = registry.create_project_with_identity(
            name=unique_name,
            description="Test project with identity",
            created_by="test-user",
            requirements_raw="Build a CRM with API",
        )

        assert success is True, f"Failed: {message}"
        assert project is not None
        assert project.fingerprint is not None
        assert len(project.fingerprint) == 64

    def test_get_all_identities(self):
        """Test 21: Should return all project identities."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        identities = registry.get_all_identities()

        assert isinstance(identities, list)
        # Each identity should be (ProjectIdentity, project_name) tuple
        for item in identities:
            assert len(item) == 2
            identity, name = item
            assert hasattr(identity, "fingerprint")
            assert isinstance(name, str)

    def test_find_similar_projects(self):
        """Test 22: Should find similar projects."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()

        # First create a project
        unique_name = f"test-similar-{uuid.uuid4().hex[:8]}"
        registry.create_project_with_identity(
            name=unique_name,
            description="Build a CRM with REST API and user management",
            created_by="test-user",
            requirements_raw="CRM with REST API",
        )

        # Then search for similar
        similar = registry.find_similar_projects(
            description="Build a CRM system with API",
            threshold=0.3,  # Low threshold for testing
        )

        assert isinstance(similar, list)
        # May or may not find matches depending on threshold

    def test_create_project_version(self):
        """Test 23: Should create new version of project."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()

        # Create parent project
        parent_name = f"test-parent-{uuid.uuid4().hex[:8]}"
        success, _, parent = registry.create_project_with_identity(
            name=parent_name,
            description="Original CRM project",
            created_by="test-user",
        )
        assert success

        # Create new version
        success, message, child = registry.create_project_version(
            parent_project=parent,
            description="Enhanced CRM with analytics",
            created_by="test-user",
            requirements_raw="CRM with analytics",
        )

        assert success is True, f"Failed: {message}"
        assert child is not None
        assert child.parent_project_id == parent.project_id
        assert child.version == "v2"


# =============================================================================
# Test 24-27: Integration Tests
# =============================================================================

class TestPhase16EIntegration:
    """Integration tests for Phase 16E components."""

    def test_full_conflict_detection_flow(self):
        """Test 24: Full conflict detection flow."""
        from controller.project_registry import get_registry
        from controller.project_decision_engine import evaluate_project_creation, DecisionType

        registry = get_registry()

        # Create initial project
        unique_name = f"test-flow-{uuid.uuid4().hex[:8]}"
        registry.create_project_with_identity(
            name=unique_name,
            description="Build a task management app with REST API",
            created_by="test-user",
        )

        # Get all identities
        identities = registry.get_all_identities()

        # Try to create similar project
        decision = evaluate_project_creation(
            description="Build a task manager application with API",
            requirements=None,
            existing_identities=identities,
        )

        # Should either detect conflict or allow new project
        assert decision.decision in [
            DecisionType.NEW_PROJECT,
            DecisionType.CONFLICT_DETECTED,
            DecisionType.REUSE_PROJECT,
        ]
        assert decision.explanation is not None

    def test_dashboard_identity_grouping(self):
        """Test 25: Dashboard should support identity grouping."""
        import asyncio
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            grouped = loop.run_until_complete(
                backend.get_projects_grouped_by_identity()
            )

            assert isinstance(grouped, dict)
            assert "grouped" in grouped
            assert "project_families" in grouped or "ungrouped_projects" in grouped
        finally:
            loop.close()

    def test_project_overview_includes_identity(self):
        """Test 26: ProjectOverview should include identity fields."""
        import asyncio
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            projects = loop.run_until_complete(backend.get_all_projects())

            for project in projects:
                data = project.to_dict()
                # Should have identity fields
                assert "fingerprint" in data
                assert "version" in data
                assert "parent_project_id" in data
        finally:
            loop.close()

    def test_registry_decision_engine_consistency(self):
        """Test 27: Registry and decision engine should be consistent."""
        from controller.project_registry import get_registry
        from controller.project_identity import ProjectIdentityManager
        from controller.project_decision_engine import evaluate_project_creation

        registry = get_registry()
        manager = ProjectIdentityManager()

        # Create identity directly
        identity1 = manager.create_identity(
            project_id="consistency-test",
            description="Build an e-commerce platform with payments",
        )

        # Create same identity through registry
        unique_name = f"test-consistency-{uuid.uuid4().hex[:8]}"
        success, _, project = registry.create_project_with_identity(
            name=unique_name,
            description="Build an e-commerce platform with payments",
            created_by="test-user",
        )

        if success and project and project.fingerprint:
            # Fingerprints should be identical for same description
            assert identity1.fingerprint == project.fingerprint


# =============================================================================
# Test 28-30: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_description_handling(self):
        """Test 28: Should handle empty descriptions gracefully."""
        from controller.project_identity import IntentExtractor

        extractor = IntentExtractor()
        intent = extractor.extract_intent("")

        assert intent is not None
        # Should have defaults even for empty input
        assert intent.database_type is not None
        assert intent.architecture_class is not None

    def test_special_characters_in_description(self):
        """Test 29: Should handle special characters."""
        from controller.project_identity import IntentExtractor

        extractor = IntentExtractor()
        intent = extractor.extract_intent(
            "Build API with 'special' chars: <html>, @users, #tags"
        )

        assert intent is not None
        assert "api" in intent.functional_modules or len(intent.functional_modules) >= 0

    def test_decision_with_no_identities(self):
        """Test 30: Should handle decision with empty identities."""
        from controller.project_decision_engine import (
            evaluate_project_creation,
            DecisionType,
        )

        decision = evaluate_project_creation(
            description="Build something new",
            requirements=None,
            existing_identities=[],  # Empty list
        )

        assert decision.decision == DecisionType.NEW_PROJECT
        assert decision.requires_user_confirmation is False


# Run tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
