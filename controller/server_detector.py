"""
Server Environment Detector for AI Development Platform
Phase 22: Rescue & Recovery System

Detects server environment (CyberPanel, cPanel, Nginx, Apache, etc.)
to provide accurate deployment and rescue instructions.

Detection Priority:
1. CHD-provided server_environment (most reliable)
2. Auto-detect from server paths/files
3. Claude Discovery (fallback - include discovery instructions)

CONSTRAINTS:
- Read-only detection - no modifications
- Deterministic classification
- Graceful fallback to UNKNOWN
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("server_detector")


class ServerEnvironment(str, Enum):
    """
    Server environment/panel types.

    LOCKED: Each type has specific config paths and rescue instructions.
    """
    CYBERPANEL = "cyberpanel"          # OpenLiteSpeed + CyberPanel
    CPANEL = "cpanel"                  # Apache/LiteSpeed + cPanel
    VIRTUALMIN = "virtualmin"          # Apache/Nginx + Virtualmin
    AAPANEL = "aapanel"                # Nginx + aaPanel
    PLESK = "plesk"                    # Nginx/Apache + Plesk
    DIRECTADMIN = "directadmin"        # Apache + DirectAdmin
    CLOUDPANEL = "cloudpanel"          # Nginx + CloudPanel
    NGINX_STANDALONE = "nginx"         # Raw Nginx (no panel)
    APACHE_STANDALONE = "apache"       # Raw Apache (no panel)
    LITESPEED_STANDALONE = "litespeed" # LiteSpeed Enterprise (no panel)
    UNKNOWN = "unknown"                # Cannot determine


class WebServer(str, Enum):
    """Web server types."""
    OPENLITESPEED = "openlitespeed"
    LITESPEED = "litespeed"
    NGINX = "nginx"
    APACHE = "apache"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ServerConfig:
    """
    Server configuration detected from CHD or auto-detection.

    FROZEN: Immutable after creation for audit integrity.
    """
    environment: ServerEnvironment
    web_server: WebServer
    config_path: str          # Main config directory
    vhost_path: str           # Virtual host configs
    service_manager: str      # systemctl, service, etc.
    python_path: str          # Python executable path
    detected_at: datetime
    detection_method: str     # "chd", "auto", "unknown"
    confidence: float         # 0.0 to 1.0
    notes: str = ""           # Any additional info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "environment": self.environment.value,
            "web_server": self.web_server.value,
            "config_path": self.config_path,
            "vhost_path": self.vhost_path,
            "service_manager": self.service_manager,
            "python_path": self.python_path,
            "detected_at": self.detected_at.isoformat(),
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "notes": self.notes
        }


# Detection signatures for each panel/environment
DETECTION_SIGNATURES = {
    ServerEnvironment.CYBERPANEL: {
        "paths": ["/usr/local/CyberCP", "/usr/local/CyberPanel"],
        "web_server": WebServer.OPENLITESPEED,
        "config_path": "/usr/local/lsws/conf",
        "vhost_path": "/usr/local/lsws/conf/vhosts",
        "service_manager": "systemctl",
    },
    ServerEnvironment.CPANEL: {
        "paths": ["/usr/local/cpanel", "/var/cpanel"],
        "web_server": WebServer.APACHE,  # Default, could be LiteSpeed
        "config_path": "/etc/apache2",
        "vhost_path": "/etc/apache2/conf.d",
        "service_manager": "systemctl",
    },
    ServerEnvironment.VIRTUALMIN: {
        "paths": ["/usr/share/webmin/virtual-server", "/etc/webmin/virtual-server"],
        "web_server": WebServer.APACHE,
        "config_path": "/etc/apache2",
        "vhost_path": "/etc/apache2/sites-available",
        "service_manager": "systemctl",
    },
    ServerEnvironment.AAPANEL: {
        "paths": ["/www/server/panel", "/www/server/nginx"],
        "web_server": WebServer.NGINX,
        "config_path": "/www/server/nginx/conf",
        "vhost_path": "/www/server/panel/vhost/nginx",
        "service_manager": "systemctl",
    },
    ServerEnvironment.PLESK: {
        "paths": ["/usr/local/psa", "/opt/psa"],
        "web_server": WebServer.NGINX,
        "config_path": "/etc/nginx",
        "vhost_path": "/var/www/vhosts/system",
        "service_manager": "systemctl",
    },
    ServerEnvironment.DIRECTADMIN: {
        "paths": ["/usr/local/directadmin"],
        "web_server": WebServer.APACHE,
        "config_path": "/etc/httpd",
        "vhost_path": "/etc/httpd/conf/extra",
        "service_manager": "systemctl",
    },
    ServerEnvironment.CLOUDPANEL: {
        "paths": ["/home/clp", "/etc/cloudpanel"],
        "web_server": WebServer.NGINX,
        "config_path": "/etc/nginx",
        "vhost_path": "/etc/nginx/sites-enabled",
        "service_manager": "systemctl",
    },
    ServerEnvironment.NGINX_STANDALONE: {
        "paths": ["/etc/nginx/nginx.conf"],
        "web_server": WebServer.NGINX,
        "config_path": "/etc/nginx",
        "vhost_path": "/etc/nginx/sites-available",
        "service_manager": "systemctl",
    },
    ServerEnvironment.APACHE_STANDALONE: {
        "paths": ["/etc/apache2/apache2.conf", "/etc/httpd/httpd.conf"],
        "web_server": WebServer.APACHE,
        "config_path": "/etc/apache2",
        "vhost_path": "/etc/apache2/sites-available",
        "service_manager": "systemctl",
    },
}


# Rescue instructions per server environment
RESCUE_INSTRUCTIONS = {
    ServerEnvironment.CYBERPANEL: {
        "HTTP_404": [
            "Check vhostconf.conf in /usr/local/lsws/conf/vhosts/{domain}/",
            "Verify proxy context is configured correctly",
            "Check extprocessor definition points to correct backend port",
            "Verify OpenLiteSpeed is restarted after config changes: systemctl restart lsws",
        ],
        "HTTP_502": [
            "Check if backend service is running",
            "Verify extprocessor address (127.0.0.1:PORT) in vhostconf",
            "Check backend logs for startup errors",
            "Verify virtual environment is activated correctly",
        ],
        "CONNECTION_REFUSED": [
            "Verify backend service is running: systemctl status <service>",
            "Check if the correct port is configured in OpenLiteSpeed admin",
            "Verify firewall allows the port",
        ],
    },
    ServerEnvironment.CPANEL: {
        "HTTP_404": [
            "Check .htaccess file in document root",
            "Verify mod_proxy settings in httpd.conf",
            "Check Apache virtual host configuration",
        ],
        "HTTP_502": [
            "Check Apache proxy settings",
            "Verify backend service is running",
            "Check ProxyPass and ProxyPassReverse directives",
        ],
        "SSL_ERROR": [
            "Check AutoSSL status in cPanel",
            "Run: /scripts/autossl_check --host <domain>",
            "Verify certificate files exist",
        ],
    },
    ServerEnvironment.NGINX_STANDALONE: {
        "HTTP_404": [
            "Check location blocks in /etc/nginx/sites-available/<domain>",
            "Verify proxy_pass URL and port",
            "Check if config is symlinked to sites-enabled",
            "Test config: nginx -t && systemctl reload nginx",
        ],
        "HTTP_502": [
            "Check upstream config in nginx",
            "Verify proxy_pass URL: http://127.0.0.1:PORT",
            "Check backend service logs",
        ],
        "HTTP_503": [
            "Check if backend service is running",
            "Verify systemd unit file",
            "Check for port conflicts",
        ],
    },
    ServerEnvironment.UNKNOWN: {
        "HTTP_404": [
            "Check web server routing configuration",
            "Verify API routes are registered correctly",
            "Check if backend service is running",
        ],
        "HTTP_500": [
            "Check backend application logs",
            "Verify database connections",
            "Check environment variables are set",
        ],
        "HTTP_502": [
            "Check backend service status",
            "Verify reverse proxy configuration",
            "Check backend port is correct",
        ],
        "HTTP_503": [
            "Verify service is starting correctly",
            "Check process manager configuration",
            "Look for port conflicts",
        ],
        "HTTP_CORS": [
            "Check CORS configuration in backend",
            "Verify allowed origins include frontend domain",
            "Check for missing CORS middleware",
        ],
        "CONNECTION_REFUSED": [
            "Verify service is running",
            "Check correct port is configured",
            "Verify firewall rules allow the port",
        ],
        "SSL_ERROR": [
            "Verify SSL certificate is valid",
            "Check certificate chain is complete",
            "Verify SSL/TLS configuration",
        ],
    },
}


class ServerDetector:
    """
    Detects server environment from CHD or by auto-detection.

    Detection Priority:
    1. CHD-provided server_environment (most reliable)
    2. Auto-detect from server paths
    3. Return UNKNOWN with discovery instructions
    """

    def __init__(self):
        self._cached_config: Optional[ServerConfig] = None

    def detect_from_chd(self, chd_text: str) -> Optional[ServerConfig]:
        """
        Extract server configuration from CHD document.

        Looks for server_environment section in CHD.
        """
        if not chd_text:
            return None

        try:
            # Look for server_environment block in YAML-like format
            env_match = re.search(
                r'server_environment:\s*\n\s*panel:\s*(\w+)',
                chd_text, re.IGNORECASE
            )
            web_server_match = re.search(
                r'server_environment:\s*\n(?:[^\n]*\n)*?\s*web_server:\s*(\w+)',
                chd_text, re.IGNORECASE
            )
            config_path_match = re.search(
                r'server_environment:\s*\n(?:[^\n]*\n)*?\s*config_path:\s*([^\s\n]+)',
                chd_text, re.IGNORECASE
            )

            if env_match:
                panel_name = env_match.group(1).lower()
                environment = self._map_panel_name(panel_name)

                # Get signature defaults
                sig = DETECTION_SIGNATURES.get(environment, {})

                web_server = WebServer.UNKNOWN
                if web_server_match:
                    ws_name = web_server_match.group(1).lower()
                    web_server = self._map_web_server_name(ws_name)
                elif "web_server" in sig:
                    web_server = sig["web_server"]

                config_path = sig.get("config_path", "/etc")
                if config_path_match:
                    config_path = config_path_match.group(1)

                return ServerConfig(
                    environment=environment,
                    web_server=web_server,
                    config_path=config_path,
                    vhost_path=sig.get("vhost_path", config_path),
                    service_manager=sig.get("service_manager", "systemctl"),
                    python_path="/usr/bin/python3",
                    detected_at=datetime.utcnow(),
                    detection_method="chd",
                    confidence=1.0,
                    notes="Extracted from CHD document"
                )

        except Exception as e:
            logger.warning(f"Failed to parse CHD for server config: {e}")

        return None

    def detect_from_paths(self, check_paths_fn=None) -> Optional[ServerConfig]:
        """
        Auto-detect server environment by checking known paths.

        Args:
            check_paths_fn: Function to check if paths exist (for SSH detection).
                           If None, uses local Path.exists()

        Returns:
            ServerConfig if detected, None otherwise
        """
        if check_paths_fn is None:
            check_paths_fn = lambda p: Path(p).exists()

        # Check each signature in order (most specific panels first)
        panel_priority = [
            ServerEnvironment.CYBERPANEL,
            ServerEnvironment.CPANEL,
            ServerEnvironment.PLESK,
            ServerEnvironment.VIRTUALMIN,
            ServerEnvironment.AAPANEL,
            ServerEnvironment.DIRECTADMIN,
            ServerEnvironment.CLOUDPANEL,
            ServerEnvironment.NGINX_STANDALONE,
            ServerEnvironment.APACHE_STANDALONE,
        ]

        for env in panel_priority:
            sig = DETECTION_SIGNATURES[env]
            for path in sig["paths"]:
                if check_paths_fn(path):
                    logger.info(f"Detected server environment: {env.value} (found {path})")
                    return ServerConfig(
                        environment=env,
                        web_server=sig["web_server"],
                        config_path=sig["config_path"],
                        vhost_path=sig["vhost_path"],
                        service_manager=sig["service_manager"],
                        python_path="/usr/bin/python3",
                        detected_at=datetime.utcnow(),
                        detection_method="auto",
                        confidence=0.9,
                        notes=f"Auto-detected from path: {path}"
                    )

        return None

    def get_unknown_config(self) -> ServerConfig:
        """Return an UNKNOWN config for when detection fails."""
        return ServerConfig(
            environment=ServerEnvironment.UNKNOWN,
            web_server=WebServer.UNKNOWN,
            config_path="/etc",
            vhost_path="/etc",
            service_manager="systemctl",
            python_path="/usr/bin/python3",
            detected_at=datetime.utcnow(),
            detection_method="unknown",
            confidence=0.0,
            notes="Server environment not detected - Claude will discover during execution"
        )

    def get_rescue_instructions(
        self,
        failure_type: str,
        config: Optional[ServerConfig] = None
    ) -> List[str]:
        """
        Get rescue instructions for a failure type.

        Args:
            failure_type: e.g., "HTTP_404", "HTTP_502", "CONNECTION_REFUSED"
            config: Server config (uses UNKNOWN if None)

        Returns:
            List of instruction strings
        """
        if config is None:
            config = self.get_unknown_config()

        env = config.environment
        instructions = RESCUE_INSTRUCTIONS.get(env, RESCUE_INSTRUCTIONS[ServerEnvironment.UNKNOWN])

        # Get specific instructions for failure type
        specific = instructions.get(failure_type, [])
        if specific:
            return specific

        # Fallback to generic instructions
        return RESCUE_INSTRUCTIONS[ServerEnvironment.UNKNOWN].get(failure_type, [
            f"Investigate {failure_type} error",
            "Check server logs for details",
            "Verify service is running correctly"
        ])

    def get_discovery_instructions(self) -> str:
        """
        Get instructions for Claude to discover server environment.

        Used when auto-detection fails.
        """
        return """
DISCOVERY PHASE - Server Environment Unknown

Before proceeding with fixes, discover the server environment:

1. Check for control panels:
   - ls /usr/local/CyberCP 2>/dev/null && echo "CyberPanel detected"
   - ls /usr/local/cpanel 2>/dev/null && echo "cPanel detected"
   - ls /usr/local/psa 2>/dev/null && echo "Plesk detected"
   - ls /www/server/panel 2>/dev/null && echo "aaPanel detected"
   - ls /usr/share/webmin 2>/dev/null && echo "Virtualmin detected"

2. Check web server:
   - which nginx && nginx -v
   - which apache2 || which httpd
   - ls /usr/local/lsws 2>/dev/null && echo "OpenLiteSpeed detected"

3. Find virtual host configs:
   - CyberPanel: /usr/local/lsws/conf/vhosts/{domain}/vhostconf.conf
   - Nginx: /etc/nginx/sites-available/* or /etc/nginx/conf.d/*
   - Apache: /etc/apache2/sites-available/* or /etc/httpd/conf.d/*

4. Document discovered environment in logs/SERVER_DISCOVERY.md:
   ```
   Server Environment: [panel name or "standalone"]
   Web Server: [nginx/apache/openlitespeed]
   Config Path: [path]
   VHost Path: [path]
   ```

5. Proceed with appropriate fix based on discovered server type.
"""

    def _map_panel_name(self, name: str) -> ServerEnvironment:
        """Map panel name string to enum."""
        mapping = {
            "cyberpanel": ServerEnvironment.CYBERPANEL,
            "cyber": ServerEnvironment.CYBERPANEL,
            "cpanel": ServerEnvironment.CPANEL,
            "virtualmin": ServerEnvironment.VIRTUALMIN,
            "webmin": ServerEnvironment.VIRTUALMIN,
            "aapanel": ServerEnvironment.AAPANEL,
            "plesk": ServerEnvironment.PLESK,
            "directadmin": ServerEnvironment.DIRECTADMIN,
            "cloudpanel": ServerEnvironment.CLOUDPANEL,
            "nginx": ServerEnvironment.NGINX_STANDALONE,
            "apache": ServerEnvironment.APACHE_STANDALONE,
            "none": ServerEnvironment.NGINX_STANDALONE,
        }
        return mapping.get(name.lower(), ServerEnvironment.UNKNOWN)

    def _map_web_server_name(self, name: str) -> WebServer:
        """Map web server name string to enum."""
        mapping = {
            "openlitespeed": WebServer.OPENLITESPEED,
            "ols": WebServer.OPENLITESPEED,
            "litespeed": WebServer.LITESPEED,
            "lsws": WebServer.LITESPEED,
            "nginx": WebServer.NGINX,
            "apache": WebServer.APACHE,
            "httpd": WebServer.APACHE,
            "apache2": WebServer.APACHE,
        }
        return mapping.get(name.lower(), WebServer.UNKNOWN)


# Singleton instance
_detector: Optional[ServerDetector] = None


def get_server_detector() -> ServerDetector:
    """Get singleton server detector instance."""
    global _detector
    if _detector is None:
        _detector = ServerDetector()
    return _detector
