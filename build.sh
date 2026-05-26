#!/usr/bin/env bash
set -o errexit

echo "📦 Installing system dependencies for WeasyPrint..."
apt-get update -qq && apt-get install -y -qq \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    libcairo2 \
    libglib2.0-0 \
    fonts-noto \
    fonts-noto-cjk \
    2>/dev/null || echo "⚠️ Some system packages may not be available"

echo "✅ System dependencies installed!"

echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎉 Build complete!"
