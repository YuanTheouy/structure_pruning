#!/bin/bash
#
# FINAL CORRECTED VERSION: This script manually clones required Hugging Face evaluate metrics
# and places them directly into the correct cache directory.
#

set -e # Exit immediately if a command exits with a non-zero status.

METRICS_CACHE_DIR="$HOME/.cache/huggingface/evaluate/metrics"
echo "Ensuring cache directory exists: $METRICS_CACHE_DIR"
mkdir -p "$METRICS_CACHE_DIR"

METRICS_TO_CACHE=("exact_match" "rouge" "bleu" "sacrebleu")
CLONE_DIR=$(mktemp -d)
echo "Using temporary directory for cloning: $CLONE_DIR"
cd "$CLONE_DIR"

for metric in "${METRICS_TO_CACHE[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing metric: $metric"
    echo "========================================"
    
    GIT_URL="https://huggingface.co/spaces/evaluate-metric/$metric"
    # The final location for the metric's code
    TARGET_DIR="$METRICS_CACHE_DIR/$metric"
    
    # Clean up any previous attempts
    echo "Cleaning up previous cache at $TARGET_DIR..."
    rm -rf "$TARGET_DIR"
    
    echo "Cloning from $GIT_URL..."
    git clone --depth 1 "$GIT_URL" ./"$metric"-repo
    
    if [ -d ./"$metric"-repo ]; then
        # === THE CRITICAL FIX IS HERE ===
        # The target directory IS the final path.
        echo "Creating final cache path: $TARGET_DIR"
        mkdir -p "$TARGET_DIR"
        
        # Move the contents of the cloned repo DIRECTLY into the target directory.
        # No more '/main' subdirectory.
        mv ./"$metric"-repo/* "$TARGET_DIR/"
        mv ./"$metric"-repo/.* "$TARGET_DIR/" 2>/dev/null || true # Move hidden files like .gitattributes
        
        echo "✅ Successfully cached '$metric' with the correct structure."
    else
        echo "❌ ERROR: Failed to clone '$metric'."
        exit 1
    fi
done

rm -rf "$CLONE_DIR"
echo ""
echo "✅ All metrics have been manually cached with the correct structure!"