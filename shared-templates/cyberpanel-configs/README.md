# CyberPanel/OpenLiteSpeed Proxy Configuration Templates

These configuration templates enable proxying requests from OpenLiteSpeed to
backend applications (Python, Node.js, etc.) running on localhost ports.

## Why Proxy Configuration?

- **No Port in URL**: Users access `https://api.domain.com` instead of `https://api.domain.com:8000`
- **SSL Termination**: OpenLiteSpeed handles HTTPS, backend runs on HTTP internally
- **CORS Headers**: Properly configured for frontend communication
- **Security**: Backend only listens on 127.0.0.1, not exposed directly

## Available Templates

### 1. python-fastapi.conf
For FastAPI/Flask Python applications.
- Proxies to `127.0.0.1:8000`
- Includes CORS headers for frontend
- Optional static files context

### 2. nextjs-ssr.conf
For Next.js Server-Side Rendered applications.
- Proxies to `127.0.0.1:3000`
- Forwards X-Forwarded-Proto for HTTPS detection
- Critical for Next.js proper functioning behind reverse proxy

### 3. nodejs-express.conf
For Node.js/Express applications.
- Proxies to `127.0.0.1:5000`
- Forwards all necessary headers

## How to Apply

1. **SSH into Server**: Connect to your VPS
2. **Copy Configuration**:
   - Go to CyberPanel Dashboard
   - Navigate to Websites > [Your Domain] > vHost Conf
   - Paste the appropriate template
   - Replace `{{VARIABLE}}` placeholders
3. **Restart OpenLiteSpeed**:
   ```bash
   sudo systemctl restart lsws
   ```

## Template Variables

Replace these in the configuration:
- `{{PROJECT_SLUG}}` - URL-safe project name (e.g., `health_tracker`)
- `{{API_DOMAIN}}` - API domain without https:// (e.g., `healthapi.gahfaudio.in`)
- `{{WEB_DOMAIN}}` - Frontend domain without https:// (e.g., `health.gahfaudio.in`)

## Systemd Service Example

To keep your backend running, create a systemd service:

```bash
sudo nano /etc/systemd/system/{{PROJECT_SLUG}}-api.service
```

```ini
[Unit]
Description={{PROJECT_NAME}} API
After=network.target

[Service]
User=www-data
WorkingDirectory=/home/{{API_DOMAIN}}/public_html
ExecStart=/home/{{API_DOMAIN}}/public_html/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable {{PROJECT_SLUG}}-api
sudo systemctl start {{PROJECT_SLUG}}-api
```
