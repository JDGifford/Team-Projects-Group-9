#!/bin/bash
IMG_DIR="$(dirname "$0")/images"
ENC_DIR="$(dirname "$0")/enc"
mkdir -p "$ENC_DIR"
KEY_FILE="$(dirname "$0")/aes_key.bin"
IV_FILE="$(dirname "$0")/iv.bin"
if [[ ! -f "$KEY_FILE" || ! -f "$IV_FILE" ]]; then
    echo "Error: AES key or IV file is missing. Please provide '$KEY_FILE' and '$IV_FILE'."
    exit 1
fi
counter=1
for img in "$IMG_DIR"/*.png; do
    if [[ ! -e "$img" ]]; then
        echo "No PNG images found in '$IMG_DIR'. Exiting."
        exit 1
    fi
    formatted_counter=$(printf "%02d" "$counter")
    output_file="$ENC_DIR/ds$formatted_counter.png"
    openssl enc -aes-256-cbc -in "$img" -out "$output_file" -K $(xxd -p -c 32 "$KEY_FILE") -iv $(xxd -p -c 16 "$IV_FILE")
    counter=$((counter + 1))
done
echo "Encryption completed. "
