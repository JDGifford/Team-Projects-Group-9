#!/bin/bash

IMG_DIR="$(dirname "$0")/images"
ENC_DIR="$(dirname "$0")/enc"
DECRYPTED_DIR="$(dirname "$0")/decrypted_images"
mkdir -p "$ENC_DIR"
mkdir -p "$DECRYPTED_DIR"

KEY_FILE="$(dirname "$0")/aes_key.bin"
IV_FILE="$(dirname "$0")/iv.bin"
if [[ ! -f "$KEY_FILE" || ! -f "$IV_FILE" ]]; then
    echo "aes_key.bin or iv.bin is missing."
    exit 1
fi
encrypted_zip_file="$ENC_DIR/images-enc.zip"
decrypted_zip_file="$ENC_DIR/images.zip"
if [[ ! -f "$encrypted_zip_file" ]]; then
    echo "Encrypted zip file '$encrypted_zip_file' not present."
    exit 1
fi
echo "Decrypting compressed file"
openssl enc -d -aes-256-cbc -in "$encrypted_zip_file" -out "$decrypted_zip_file" -K $(xxd -p -c 32 "$KEY_FILE") -iv $(xxd -p -c 16 "$IV_FILE")
echo "Decompressing compressed file."
unzip -q "$decrypted_zip_file" -d "$DECRYPTED_DIR"
rm "$decrypted_zip_file"
echo "Process complete. Decrypted images in '$DECRYPTED_DIR'."
