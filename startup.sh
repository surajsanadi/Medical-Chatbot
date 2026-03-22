#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading NLTK resources..."

python - <<EOF
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
EOF

echo "Starting Gunicorn server..."

gunicorn --bind=0.0.0.0 --timeout 600 app:app