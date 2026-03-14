#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Uni-OS Setup Script
# Run this ONCE on a fresh machine to install everything needed.
# Usage:  bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit immediately if any command fails

echo "======================================================"
echo "  Uni-OS Setup — University of Waterloo AI Companion"
echo "======================================================"
echo ""

# ── 1. Python dependencies ────────────────────────────────────────────────────
echo "▶ [1/5] Installing Python dependencies..."
pip3 install -r requirements.txt
echo "✅ Python packages installed."
echo ""

# ── 2. Frontend dependencies ──────────────────────────────────────────────────
echo "▶ [2/5] Installing frontend (Node) dependencies..."
cd frontend && npm install && cd ..
echo "✅ Node packages installed."
echo ""

# ── 3. Environment file ───────────────────────────────────────────────────────
echo "▶ [3/5] Setting up environment variables..."
if [ -f ".env" ]; then
  echo "⚠️  .env already exists — skipping. Edit it manually if needed."
else
  cp .env.example .env
  echo "✅ .env created from .env.example"
  echo ""
  echo "  ⚠️  IMPORTANT: Open .env and fill in your API keys before running!"
  echo "     - GOOGLE_API_KEY       → https://console.cloud.google.com/apis/credentials"
  echo "     - GOOGLE_CLOUD_QUOTA_PROJECT → your GCP project ID"
  echo "     - WATERLOO_API_KEY     → https://openapi.data.uwaterloo.ca"
fi
echo ""

# ── 4. Google Cloud Auth ──────────────────────────────────────────────────────
echo "▶ [4/5] Checking Google Cloud authentication..."
if command -v gcloud &> /dev/null; then
  echo "✅ Google Cloud SDK found."
  echo "   Run 'gcloud auth application-default login' if you haven't already."
else
  echo "⚠️  Google Cloud SDK not found."
  echo "   Install it by running:"
  echo "     curl https://sdk.cloud.google.com | bash"
  echo "   Then restart your shell and run:"
  echo "     gcloud auth application-default login"
fi
echo ""

# ── 5. Done ───────────────────────────────────────────────────────────────────
echo "▶ [5/5] Setup complete!"
echo ""
echo "  To run the app, open TWO terminal windows:"
echo ""
echo "  Terminal 1 — Backend (AI + API):"
echo "    cd $(pwd)"
echo "    python3 -m backend.main"
echo ""
echo "  Terminal 2 — Frontend (UI):"
echo "    cd $(pwd)/frontend"
echo "    npm run dev"
echo ""
echo "  Then open: http://localhost:3000 🚀"
echo ""
