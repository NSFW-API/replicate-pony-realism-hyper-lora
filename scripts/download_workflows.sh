#!/bin/bash
# download_workflows.sh
# Downloads the workflow templates needed for HyperLoRA integration

# Use the current directory or script directory for local development
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKFLOW_DIR="${PROJECT_ROOT}/workflows"

# Create workflows directory
echo "Creating workflows directory: ${WORKFLOW_DIR}"
mkdir -p "$WORKFLOW_DIR"

# Download the workflow templates
echo "Downloading HyperLoRA workflow templates..."

# T2I standard workflow
echo "Downloading HyperLoRA-T2I.json..."
curl -L -o "${WORKFLOW_DIR}/HyperLoRA-T2I.json" \
  "https://raw.githubusercontent.com/bytedance/ComfyUI-HyperLoRA/main/assets/HyperLoRA-T2I.json"

# T2I FaceDetailer workflow
echo "Downloading HyperLoRA-T2I-FaceDetailer.json..."
curl -L -o "${WORKFLOW_DIR}/HyperLoRA-T2I-FaceDetailer.json" \
  "https://raw.githubusercontent.com/bytedance/ComfyUI-HyperLoRA/main/assets/HyperLoRA-T2I-FaceDetailer.json"

echo "Workflow templates downloaded successfully to ${WORKFLOW_DIR}"