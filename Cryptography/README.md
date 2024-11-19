# Cryptographic Protocol
## Encryption Protocol File Structure
### Encryption Script Parent Directory
- images (Image Directory)
  - image1.png
  - image2.png
  - imageN.png
- enc (Directory to house encrypted zip file)
  - images-enc.zip (Encrypted zip file)
- compress-enc.sh (renaming, compression, and encryption script)
- aes_key.bin (AES key file)
- iv.bin (initialization vector file)
## Decryption Protocol File Structure
### Decryption Script Parent Directory
- decrypted_images
  - images (Image Directory)
    - ds01.png
    - ds02.png
    - dsN.png
- enc (Directory to house encrypted zip file)
  - images-enc.zip (Encrypted zip file)
- decompress-dec.sh (Decompression and decryption script)
- aes_key.bin (AES key file)
- iv.bin (initialization vector file)