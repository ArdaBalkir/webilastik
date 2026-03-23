#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# build_deb.sh — Build the webilastik2 Debian package
#
# Run on Ubuntu/Debian.  Requires: dpkg-deb, npm, python3
#
# Output: dist/webilastik2_2.0.0_amd64.deb
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="2.0.0"
PKG="webilastik2_${VERSION}_amd64"
BUILD_DIR="$REPO_ROOT/dist/build/$PKG"
OUT_DIR="$REPO_ROOT/dist"
INSTALL_ROOT="$BUILD_DIR/opt/webilastik2"

echo "==> Cleaning build dir …"
rm -rf "$BUILD_DIR"
mkdir -p "$INSTALL_ROOT" "$OUT_DIR"

# ── 1. Python backend ─────────────────────────────────────────────────────────
echo "==> Copying backend …"
cp -r "$REPO_ROOT/backend" "$INSTALL_ROOT/"

# ── 2. Frontend — build first ─────────────────────────────────────────────────
echo "==> Building frontend …"
cd "$REPO_ROOT/frontend"
npm ci --silent
npm run build
cd "$REPO_ROOT"
cp -r "$REPO_ROOT/frontend/dist" "$INSTALL_ROOT/frontend/dist"

# ── 3. Requirements (ship them; postinst will pip install into venv) ──────────
cp "$REPO_ROOT/backend/requirements.txt" "$INSTALL_ROOT/"

# ── 4. DEBIAN control files ───────────────────────────────────────────────────
echo "==> Copying DEBIAN files …"
cp -r "$REPO_ROOT/package_tree/DEBIAN" "$BUILD_DIR/"
chmod 755 "$BUILD_DIR/DEBIAN/postinst" "$BUILD_DIR/DEBIAN/prerm"

# ── 5. systemd units ──────────────────────────────────────────────────────────
echo "==> Copying systemd units …"
mkdir -p "$BUILD_DIR/lib/systemd/system"
cp "$REPO_ROOT/package_tree/lib/systemd/system/"*.service \
   "$REPO_ROOT/package_tree/lib/systemd/system/"*.target \
   "$BUILD_DIR/lib/systemd/system/"

# ── 6. nginx config ───────────────────────────────────────────────────────────
echo "==> Copying nginx config …"
mkdir -p "$BUILD_DIR/etc/nginx/sites-available"
cp "$REPO_ROOT/package_tree/etc/nginx/sites-available/webilastik2.conf" \
   "$BUILD_DIR/etc/nginx/sites-available/"

# ── 7. /etc/webilastik2 skeleton ─────────────────────────────────────────────
echo "==> Copying config skeleton …"
mkdir -p "$BUILD_DIR/etc/webilastik2"
cp "$REPO_ROOT/package_tree/etc/webilastik2/env.example" \
   "$BUILD_DIR/etc/webilastik2/"

# ── 8. www-data home for SSH known_hosts ─────────────────────────────────────
mkdir -p "$BUILD_DIR/var/lib/webilastik2/.ssh"

# ── 9. Build the .deb ────────────────────────────────────────────────────────
echo "==> Building .deb …"
dpkg-deb --build --root-owner-group "$BUILD_DIR" "$OUT_DIR/${PKG}.deb"

echo ""
echo "==> Done: $OUT_DIR/${PKG}.deb"
echo "    Install with:  sudo dpkg -i $OUT_DIR/${PKG}.deb"
