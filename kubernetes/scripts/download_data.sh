#!/bin/bash
set -e

DOWNLOAD_DIR="/develop/data"
ZIP_NAME="ADM.zip"
URL="https://research.repository.duke.edu/record/176/files/ADM.zip?ln=en"

cd "$DOWNLOAD_DIR"

echo "Downloading ADM dataset..."
wget -v -O "$ZIP_NAME" "$URL"

echo "Extracting..."
unzip -o "$ZIP_NAME" -d "$DOWNLOAD_DIR"

echo "Cleaning up zip..."
rm "$ZIP_NAME"

echo "Download complete. Contents:"
ls "$DOWNLOAD_DIR"
