set -e
cd hand_detector.d2

eval "$(conda shell.bash hook)"
conda activate hamer


gdown https://drive.google.com/uc\?id\=1OqgexNM52uxsPG3i8GuodDOJAGFsYkPg
mkdir -p models
mv model_0529999.pth models
cd ..