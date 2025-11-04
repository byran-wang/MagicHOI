set -e
cd Cutie
echo "Installing Cutie..."

pip install -e .
python ./scripts/download_models.py