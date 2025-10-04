#!/usr/bin/env bash
set -euo pipefail

PY=${1:-python3}
PROJECT_ROOT=${2:-healthai}

echo "[Setup] Creating virtual environment..."
$PY -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[Setup] Upgrading pip..."
pip install --upgrade pip

echo "[Setup] Installing requirements from ${PROJECT_ROOT}/requirements.txt ..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "[Setup] Done." 