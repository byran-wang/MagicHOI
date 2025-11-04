set -e
cd Hierarchical-Localization_hold

pip install -e .
pip install trimesh
pip install open3d
pip install plyfile
pip uninstall pycolmap
pip install pycolmap==0.3.0
pip uninstall numpy
pip install numpy==1.24.1

cd ..