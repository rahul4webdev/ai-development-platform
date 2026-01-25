"""
Template Engine for Project Scaffolding
Phase 21: Project Templates & CI Fixes

This module provides template loading, variable substitution, and file generation
for new projects. Templates include pre-configured CI/CD workflows with known
fixes for common issues.

IMPORTANT:
- Templates are read-only during project creation
- Variable substitution is deterministic
- No runtime template modification allowed
"""

import logging
import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("template_engine")

# Template base directory
TEMPLATES_DIR = Path(os.getenv(
    "TEMPLATES_DIR",
    Path(__file__).parent.parent / "shared-templates"
))


class TemplateType(str, Enum):
    """
    Available project template types.

    LOCKED: DO NOT add new types without updating scaffolding logic.
    """
    FASTAPI_REACT = "fastapi-react"
    NEXTJS_APP = "nextjs-app"
    PYTHON_API = "python-api"
    REACT_SPA = "react-spa"


@dataclass(frozen=True)
class TemplateVariables:
    """
    Variables for template substitution.

    FROZEN: Immutable after creation to ensure deterministic output.
    """
    project_name: str
    project_slug: str
    api_domain: str
    web_domain: str
    db_name: str
    github_user: str

    @classmethod
    def from_project_config(cls, config: Dict) -> "TemplateVariables":
        """Create variables from project configuration."""
        project_name = config.get("project_name", "")
        # Create URL-safe slug
        project_slug = re.sub(r'[^a-z0-9_]', '_', project_name.lower())

        # Extract domains from config
        domains = config.get("domains", {})
        api_domain = domains.get("api", f"{project_slug}api.example.com")
        web_domain = domains.get("web", f"{project_slug}.example.com")

        return cls(
            project_name=project_name,
            project_slug=project_slug,
            api_domain=api_domain,
            web_domain=web_domain,
            db_name=config.get("db_name", project_slug),
            github_user=config.get("github_user", "")
        )

    def as_dict(self) -> Dict[str, str]:
        """Convert to dictionary for template substitution."""
        return {
            "PROJECT_NAME": self.project_name,
            "PROJECT_SLUG": self.project_slug,
            "API_DOMAIN": self.api_domain,
            "WEB_DOMAIN": self.web_domain,
            "DB_NAME": self.db_name,
            "GITHUB_USER": self.github_user,
        }


class TemplateEngine:
    """
    Template engine for project scaffolding.

    Features:
    - Load templates from shared-templates directory
    - Variable substitution with {{VARIABLE}} syntax
    - Copy and transform files to target directory
    - Include CI/CD fixes by default
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")

    def get_available_templates(self) -> List[TemplateType]:
        """List available template types."""
        available = []
        for template_type in TemplateType:
            template_path = self.templates_dir / template_type.value
            if template_path.exists():
                available.append(template_type)
        return available

    def detect_template_type(self, tech_stack: Dict) -> TemplateType:
        """
        Detect appropriate template type from tech stack configuration.

        Args:
            tech_stack: Dictionary with 'backend' and 'frontend' keys

        Returns:
            Appropriate TemplateType
        """
        backend = tech_stack.get("backend", "").lower()
        frontend = tech_stack.get("frontend", "").lower()

        # FastAPI + React combination
        if "fastapi" in backend and "react" in frontend:
            return TemplateType.FASTAPI_REACT

        # Next.js application
        if "next" in frontend or "nextjs" in frontend:
            return TemplateType.NEXTJS_APP

        # Python-only API
        if ("fastapi" in backend or "flask" in backend) and not frontend:
            return TemplateType.PYTHON_API

        # React SPA (no backend or external API)
        if "react" in frontend and not backend:
            return TemplateType.REACT_SPA

        # Default to FastAPI + React as most common
        logger.warning(f"Could not detect template type from {tech_stack}, defaulting to fastapi-react")
        return TemplateType.FASTAPI_REACT

    def substitute_variables(self, content: str, variables: TemplateVariables) -> str:
        """
        Substitute {{VARIABLE}} placeholders in content.

        Args:
            content: Template content with placeholders
            variables: Variables for substitution

        Returns:
            Content with variables substituted
        """
        var_dict = variables.as_dict()

        def replace_var(match):
            var_name = match.group(1)
            if var_name in var_dict:
                return var_dict[var_name]
            logger.warning(f"Unknown template variable: {{{{{var_name}}}}}")
            return match.group(0)  # Keep original if not found

        return re.sub(r'\{\{(\w+)\}\}', replace_var, content)

    def scaffold_project(
        self,
        template_type: TemplateType,
        target_dir: Path,
        variables: TemplateVariables,
        overwrite: bool = False
    ) -> Dict[str, List[str]]:
        """
        Scaffold a new project from template.

        Args:
            template_type: Type of template to use
            target_dir: Directory to create project in
            variables: Variables for substitution
            overwrite: Whether to overwrite existing files

        Returns:
            Dictionary with 'created' and 'skipped' file lists
        """
        template_path = self.templates_dir / template_type.value
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_type.value}")

        created = []
        skipped = []

        # Walk through template directory
        for root, dirs, files in os.walk(template_path):
            # Calculate relative path from template root
            rel_root = Path(root).relative_to(template_path)
            target_root = target_dir / rel_root

            # Create directories
            target_root.mkdir(parents=True, exist_ok=True)

            for file in files:
                src_file = Path(root) / file
                dst_file = target_root / file

                # Check if should skip
                if dst_file.exists() and not overwrite:
                    skipped.append(str(dst_file.relative_to(target_dir)))
                    continue

                # Read and substitute variables
                try:
                    content = src_file.read_text()
                    substituted = self.substitute_variables(content, variables)
                    dst_file.write_text(substituted)
                    created.append(str(dst_file.relative_to(target_dir)))
                except UnicodeDecodeError:
                    # Binary file - copy directly
                    shutil.copy2(src_file, dst_file)
                    created.append(str(dst_file.relative_to(target_dir)))

        logger.info(f"Scaffolded {len(created)} files, skipped {len(skipped)} existing files")
        return {"created": created, "skipped": skipped}

    def get_github_workflows(
        self,
        template_type: TemplateType,
        variables: TemplateVariables
    ) -> Dict[str, str]:
        """
        Get GitHub workflow files with variables substituted.

        Args:
            template_type: Type of template
            variables: Variables for substitution

        Returns:
            Dictionary of workflow filename -> content
        """
        workflows_path = self.templates_dir / template_type.value / "github-workflows"
        if not workflows_path.exists():
            return {}

        workflows = {}
        for workflow_file in workflows_path.glob("*.yml"):
            content = workflow_file.read_text()
            substituted = self.substitute_variables(content, variables)
            workflows[workflow_file.name] = substituted

        return workflows

    def get_cyberpanel_config(
        self,
        config_type: str,
        variables: TemplateVariables
    ) -> Optional[str]:
        """
        Get CyberPanel proxy configuration with variables substituted.

        Args:
            config_type: Type of config (python-fastapi, nextjs-ssr, nodejs-express)
            variables: Variables for substitution

        Returns:
            Configuration content or None if not found
        """
        config_path = self.templates_dir / "cyberpanel-configs" / f"{config_type}.conf"
        if not config_path.exists():
            logger.warning(f"CyberPanel config not found: {config_type}")
            return None

        content = config_path.read_text()
        return self.substitute_variables(content, variables)


# Singleton instance
_engine: Optional[TemplateEngine] = None


def get_template_engine() -> TemplateEngine:
    """Get singleton template engine instance."""
    global _engine
    if _engine is None:
        _engine = TemplateEngine()
    return _engine
