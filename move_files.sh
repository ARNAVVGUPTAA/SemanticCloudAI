#!/bin/bash
TARGET_DIR=~/dev/semanticcloudai

echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "Moving files..."
cp -r /home/charismatic/.gemini/antigravity/scratch/pi_doc_cloud/* "$TARGET_DIR"/
cp /home/charismatic/.gemini/antigravity/scratch/pi_doc_cloud/.env "$TARGET_DIR"/ 2>/dev/null || true

echo "Success! Project moved to $TARGET_DIR"
