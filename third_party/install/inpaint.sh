set -e
cd Inpaint-Anything
echo "Installing Inpaint-Anything..."

pip install ./segment_anything
python -m pip install -r lama/requirements.txt 