#!/bin/bash

# Usage: ./download_gdrive_file.sh "Google Drive File ID" "destination_directory/filename.ext"

FILE_ID="13qzbFHULsKiV77lII6CciiCLU3ErTExC"
DESTINATION="rep_theory/utils/rep_theory_data.tar.gz"

echo "Downloading cache from google drive"

CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "${DESTINATION}"

echo "Downloaded."

echo "Unzipping cache"

tar -xzf rep_theory/utils/rep_theory_data.tar.gz -C rep_theory/utils/

echo "Unzipped cache"
