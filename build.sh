#!/bin/bash
set -e
pip install --no-cache-dir -r backend/requirements.txt
pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true
echo "✅ nvidia-nccl-cu12 removed — CPU-only mode active"
