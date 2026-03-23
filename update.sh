#!/bin/bash
# update.sh — pull latest changes and restart services on the VM
# Run as root from /opt/webilastik2

set -euo pipefail
cd /opt/webilastik2

echo "==> Pulling latest changes..."
git pull

echo "==> Installing/updating Python dependencies..."
.venv/bin/pip install -q -r backend/requirements.txt

echo "==> Building frontend..."
cd frontend
npm ci --silent
npm run build
cd ..

echo "==> Restarting services..."
systemctl restart wi2-server wi2-allocator

echo "==> Waiting for services to come up..."
sleep 2
systemctl is-active --quiet wi2-server && echo "  wi2-server:    OK" || echo "  wi2-server:    FAILED"
systemctl is-active --quiet wi2-allocator && echo "  wi2-allocator: OK" || echo "  wi2-allocator: FAILED"

echo "==> Done."
