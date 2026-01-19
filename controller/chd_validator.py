"""
Phase 16C: CHD (Claude-Human Document) Validation Layer

Validates project requirements before Claude job execution.

HARD CONSTRAINTS:
- At least 1 backend OR frontend must be defined
- Database type must be specified if persistence mentioned
- No auto-deploy to production flag
- Domain mappings must be valid

This layer prevents 500 errors by catching invalid requirements early.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("chd_validator")


# -----------------------------------------------------------------------------
# Domain Mappings
# -----------------------------------------------------------------------------
ASPECT_DOMAIN_MAP = {
    # API mappings
    "api": "backend",
    "rest_api": "backend",
    "graphql": "backend",
    "backend_api": "backend",

    # Frontend mappings
    "frontend_web": "frontend",
    "web_app": "frontend",
    "website": "frontend",
    "spa": "frontend",
    "client": "frontend",

    # Admin mappings
    "admin_panel": "admin",
    "admin": "admin",
    "dashboard": "admin",
    "backoffice": "admin",

    # Core mappings
    "core": "core",
    "shared": "core",
    "common": "core",
    "library": "core",
}

# Database types
VALID_DATABASE_TYPES = [
    "postgresql", "postgres", "mysql", "mariadb", "sqlite",
    "mongodb", "dynamodb", "redis", "firestore", "supabase",
]

# Tech stacks
VALID_TECH_STACKS = {
    "backend": ["python", "node", "go", "rust", "java", "php", "ruby", ".net", "fastapi", "django", "express", "nestjs"],
    "frontend": ["react", "vue", "angular", "svelte", "nextjs", "nuxt", "remix", "astro", "htmx"],
    "admin": ["react-admin", "adminjs", "retool", "appsmith", "custom"],
}


# -----------------------------------------------------------------------------
# Validation Result
# -----------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Result of CHD validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extracted_aspects: List[str] = field(default_factory=list)
    extracted_tech: Dict[str, str] = field(default_factory=dict)
    extracted_database: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "extracted_aspects": self.extracted_aspects,
            "extracted_tech": self.extracted_tech,
            "extracted_database": self.extracted_database,
            "suggestions": self.suggestions,
        }

    def get_user_message(self) -> str:
        """Format validation result for user display."""
        if self.is_valid:
            return "Requirements validated successfully."

        lines = ["Validation failed:"]
        for error in self.errors:
            lines.append(f"  - {error}")

        if self.suggestions:
            lines.append("\nSuggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# CHD Validator
# -----------------------------------------------------------------------------
class CHDValidator:
    """
    Validates project requirements before Claude job execution.

    Ensures:
    1. At least one buildable aspect (backend/frontend)
    2. Valid domain mappings
    3. Database type if persistence needed
    4. No production auto-deploy flags
    """

    def validate(
        self,
        description: str,
        requirements: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate project description and requirements.

        Returns ValidationResult with errors/warnings.
        """
        result = ValidationResult(is_valid=True)
        content = f"{description} {requirements or ''}".lower()

        # 1. Check for at least one aspect
        self._validate_aspects(content, result)

        # 2. Check database requirements
        self._validate_database(content, result)

        # 3. Check for dangerous flags
        self._validate_no_dangerous_flags(content, result)

        # 4. Extract tech stack
        self._extract_tech_stack(content, result)

        # 5. Validate completeness
        self._validate_completeness(content, result)

        # Set overall validity
        result.is_valid = len(result.errors) == 0

        return result

    def _validate_aspects(self, content: str, result: ValidationResult) -> None:
        """Ensure at least one buildable aspect is defined."""
        backend_keywords = ["api", "backend", "server", "rest", "graphql", "database", "service"]
        frontend_keywords = ["frontend", "web app", "website", "ui", "spa", "client", "react", "vue", "angular"]
        admin_keywords = ["admin", "dashboard", "backoffice", "management"]

        has_backend = any(kw in content for kw in backend_keywords)
        has_frontend = any(kw in content for kw in frontend_keywords)
        has_admin = any(kw in content for kw in admin_keywords)

        if has_backend:
            result.extracted_aspects.append("api")
        if has_frontend:
            result.extracted_aspects.append("frontend")
        if has_admin:
            result.extracted_aspects.append("admin")

        # Always include core
        result.extracted_aspects.append("core")

        if not has_backend and not has_frontend:
            result.errors.append(
                "Project must define at least one backend (API) or frontend component"
            )
            result.suggestions.append(
                "Add details about your API endpoints or user interface"
            )

    def _validate_database(self, content: str, result: ValidationResult) -> None:
        """Check database requirements if persistence is mentioned."""
        persistence_keywords = ["database", "store", "persist", "save", "data", "crud", "user accounts", "authentication"]
        needs_database = any(kw in content for kw in persistence_keywords)

        if needs_database:
            # Check if database type is specified
            for db_type in VALID_DATABASE_TYPES:
                if db_type in content:
                    result.extracted_database = db_type
                    break

            if not result.extracted_database:
                result.warnings.append(
                    "Project mentions data persistence but no database type specified"
                )
                result.suggestions.append(
                    "Consider specifying database: PostgreSQL, MySQL, MongoDB, etc."
                )
                # Default to PostgreSQL
                result.extracted_database = "postgresql"

    def _validate_no_dangerous_flags(self, content: str, result: ValidationResult) -> None:
        """Ensure no dangerous flags are present."""
        dangerous_patterns = [
            (r"auto[- ]?deploy\s*(to\s*)?prod", "Auto-deploy to production is not allowed"),
            (r"skip\s*(all\s*)?tests", "Skipping tests is not allowed"),
            (r"no\s*review", "Code review cannot be skipped"),
            (r"bypass\s*approval", "Approval workflow cannot be bypassed"),
            (r"force\s*push", "Force push is not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, content):
                result.errors.append(message)

    def _extract_tech_stack(self, content: str, result: ValidationResult) -> None:
        """Extract tech stack from content."""
        # Backend tech
        for tech in VALID_TECH_STACKS["backend"]:
            if tech in content:
                result.extracted_tech["backend"] = tech
                break

        # Frontend tech
        for tech in VALID_TECH_STACKS["frontend"]:
            if tech in content:
                result.extracted_tech["frontend"] = tech
                break

        # If API mentioned but no backend tech, suggest FastAPI
        if "api" in result.extracted_aspects and "backend" not in result.extracted_tech:
            result.extracted_tech["backend"] = "fastapi"
            result.suggestions.append(
                "No backend framework specified, defaulting to FastAPI (Python)"
            )

        # If frontend mentioned but no frontend tech, suggest React
        if "frontend" in result.extracted_aspects and "frontend" not in result.extracted_tech:
            result.extracted_tech["frontend"] = "react"
            result.suggestions.append(
                "No frontend framework specified, defaulting to React"
            )

    def _validate_completeness(self, content: str, result: ValidationResult) -> None:
        """Check for minimum completeness."""
        word_count = len(content.split())

        if word_count < 10:
            result.warnings.append(
                "Description is very brief. More detail helps Claude generate better code."
            )
            result.suggestions.append(
                "Consider adding: user roles, main features, authentication needs"
            )

        # Check for key missing information
        if "authentication" not in content and "auth" not in content and "login" not in content:
            if "user" in content or "account" in content:
                result.warnings.append(
                    "Users mentioned but no authentication method specified"
                )
                result.suggestions.append(
                    "Consider specifying: email/password, OAuth, SSO, etc."
                )


# -----------------------------------------------------------------------------
# File Content Validator
# -----------------------------------------------------------------------------
class FileContentValidator:
    """Validates uploaded file content."""

    ALLOWED_EXTENSIONS = [".md", ".txt", ".text"]
    MAX_FILE_SIZE = 100 * 1024  # 100KB
    MIN_CONTENT_LENGTH = 20

    def validate_file(
        self,
        filename: str,
        content: bytes,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate uploaded file.

        Returns: (is_valid, error_message, content_text)
        """
        # Check extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}", None

        # Check size
        if len(content) > self.MAX_FILE_SIZE:
            return False, f"File too large. Maximum: {self.MAX_FILE_SIZE // 1024}KB", None

        # Decode content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except:
                return False, "Could not decode file content", None

        # Check minimum content
        if len(text.strip()) < self.MIN_CONTENT_LENGTH:
            return False, f"File content too short. Minimum: {self.MIN_CONTENT_LENGTH} characters", None

        return True, "", text


# -----------------------------------------------------------------------------
# Global Validator Instance
# -----------------------------------------------------------------------------
_validator: Optional[CHDValidator] = None
_file_validator: Optional[FileContentValidator] = None


def get_validator() -> CHDValidator:
    """Get the global CHD validator instance."""
    global _validator
    if _validator is None:
        _validator = CHDValidator()
    return _validator


def get_file_validator() -> FileContentValidator:
    """Get the global file validator instance."""
    global _file_validator
    if _file_validator is None:
        _file_validator = FileContentValidator()
    return _file_validator


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
def validate_requirements(
    description: str,
    requirements: Optional[str] = None,
) -> ValidationResult:
    """Validate project requirements."""
    return get_validator().validate(description, requirements)


def validate_file(
    filename: str,
    content: bytes,
) -> Tuple[bool, str, Optional[str]]:
    """Validate uploaded file."""
    return get_file_validator().validate_file(filename, content)
