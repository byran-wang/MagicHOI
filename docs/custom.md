# Creating custom sequences

This document gives instructions to preprocess custom video sequences. 



## Pipeline overview

Overall, the preprocessing pipeline is as follows:

0. Dependency installation
1. Create dataset scaffold
2. Image segmentation
3. Object pose estimation
4. Hand pose estimation
5. Objcet inpainting

## Dependency installation
If you want to reconstruct a custom video sequence with MagicHOI, you will need to setup the following dependencies. Here we provide tested instructions to install them. For additional installation related issues, refer to the original repo.

The the CUDA version in the following installation instruction is 11.8. If the CUDA is not consistent with your machine, you should modify the CUDA version.




### [Cutie](https://github.com/hkchengrex/Cutie)

```bash
conda create -y --name cutie python=3.8.18
conda activate cutie
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd <project_root_dir>/third_party/
bash ./install/cutie.sh 
```

### [HLoc](https://github.com/cvg/Hierarchical-Localization)

```bash
conda create -y --name hloc python=3.9.17
conda activate hloc
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd <project_root_dir>/third_party/
bash ./install/hloc.sh 
```

### [COLMAP](https://github.com/colmap/colmap)

Following the installation documents in [COLMAP](https://github.com/colmap/colmap) to install colmap.


### [HaMeR](https://github.com/geopavlakos/hamer)

```bash
conda create -y --name hamer python=3.10.14
conda activate hamer
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd <project_root_dir>/third_party/
bash ./install/hamer.sh
```
### [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)

```bash
conda create -y --name inpaint python=3.9.19
conda activate inpaint
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd <project_root_dir>/third_party/
bash ./install/inpaint.sh 
```
Download the [pretrained_models](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg), and put them in the `<project_dir>/third_party/Inpaint-Anything/pretrained_models`

### Export the conda enviroment path

```bash
export PYCUTIE='~/anaconda3/envs/cutie/bin/python'
export PYHLOC='~/anaconda3/envs/hloc/bin/python'
export PYHAMER='~/anaconda3/envs/hamer/bin/python'
export PYINPAINT='~/anaconda3/envs/inpaint/bin/python'
```
Feel free to put them inside your ~/.zshrc or ~/.bashrc depending on your shell.
By default, python refers to '~/anaconda3/envs/MagicHOI/bin/python' in all documentations for simplicity.


## Dataset folder

Let's use the sequence `hold_MC1_ho3d` as an example. The sequence name can be choosed from the results of 'ls ./data'. Start from the project root and duplicate the raw images so the original data stays untouched:

```bash
cdroot
seq_name=hold_MC1_ho3d
mv data/$seq_name/processed data/$seq_name/processed_origin # back up the original processed folder
mkdir -p data/$seq_name/build
cp -r data/$seq_name/processed_origin/images data/$seq_name/build/
```

## Segmentation with [Cutie](https://github.com/hkchengrex/Cutie)

The goal of this step is to extract hand and object masks for the input video. We rely on Cutie: you select the region of interest in the first frame and Cutie propagates the mask through the video.

Follow the installation guide in the Cutie repository, then launch the labeling tool:

```bash
cdroot
seq_name=hold_MC1_ho3d
seq_name_with_suffix=$seq_name.0  # NOTE: include the suffix when selecting from all_sequences list in run.py
python run.py  --execute_list only_3d --process_list images_to_video ho_mask --seq_list $seq_name_with_suffix --rebuild
```
This command opens two Cutie windows: the first for object segmentation and the second for hand segmentation.

Cutie usage tips:
1. Right-click to add a positive prompt.
2. Left-click to add a negative prompt.
3. After providing prompts on the first frame, click **Propagete forward** to propagate the segmentation.
4. When all frames look correct, click **Export binary masks** to write the masks.
5. Use the left/right arrow keys to move between frames and refine masks with additional prompts.

After running `scripts/ho_mask.py`, `data/$seq_name/processed/` should contain:
```text
MagicHOI/
└── ./data/$seq_name/processed
                        ├── images/
                        ├── masks/
                        └── rgbas/
```
## Object pose estimation with [HLoc](https://github.com/cvg/Hierarchical-Localization) and [COLMAP](https://github.com/colmap/colmap)

Run HLoc to estimate the object pose and sparse point cloud, then run MVS to produce dense points and depth maps. These outputs let us align the COLMAP coordinates to the generated object coordinate from the novel view synthesis (NVS) model.

Use the sequence `hold_MC1_ho3d` as an example.
```bash
cdroot
seq_name=hold_MC1_ho3d
seq_name_with_suffix=$seq_name.0  # NOTE: include the suffix when selecting from all_sequences list in run.py
python run.py --mute --execute_list only_3d --process_list colmap validate_colmap gen_HO3D --seq_list $seq_name_with_suffix --rebuild
```

After running the above command, `data/$seq_name/processed/` should contain:
```text
MagicHOI/
└── ./data/$seq_name/processed
                        ├── colmap_2d/
                        └── colmap_$seq_name_with_suffix/
```
Verify the pose by inspecting the reprojection overlays in `data/$seq_name/processed/colmap_2d`.
View the sparse point cloud with `data/$seq_name/processed/colmap_$seq_name_with_suffix/sparse_points.ply`.
Inspect the dense reconstruction at `data/$seq_name/processed/colmap_$seq_name_with_suffix/sfm_superpoint+superglue/mvs/fused.ply`.

## Hand pose estimation with [HaMeR](https://github.com/geopavlakos/hamer)

Since HaMeR has hand detection, we can directly estimate hand poses.
Use the sequence `hold_MC1_ho3d` as an example. You can pick another sequence name from the `all_sequences` list in `run.py`.
Run the commands below to estimate MANO pose:


```bash
cdroot
seq_name=hold_MC1_ho3d
seq_name_with_suffix=$seq_name.0  # NOTE: include the suffix when selecting from all_sequences list in run.py
python run.py --execute_list only_3d --process_list rm_unused_images_after_colmap rebuild_hand crop_hand hand_pose_hamer validate_hamer --seq_list $seq_name_with_suffix --rebuild
```
After running the above command, `data/$seq_name/processed/` should contain:
```text
MagicHOI/
└── ./data/$seq_name/processed
                        ├── 2d_keypoints/
                        ├── crop_image/
                        ├── hpe_vis/
                        ├── boxes.npy
                        ├── hold_fit.init.npy
                        ├── hold_fit.slerp.npy
                        ├── j2d.full.npy
                        └── v3d.npy
```
Verify the pose by inspecting the hand landmark reprojection overlays in `data/$seq_name/processed/2d_keypoints`.



## Object inpainting with [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)
We select a reference frame from the video to serve as a conditioning image for the NVS model. When the hand occludes the object in that frame, we apply the state-of-the-art Inpaint-Anything model to hallucinate the hidden regions.

Use the sequence `hold_MC1_ho3d` as an example.
```bash
cdroot
seq_name=hold_MC1_ho3d
seq_name_with_suffix=$seq_name.0  # NOTE: include the suffix when selecting from all_sequences list in run.py
python run.py --mute --execute_list only_3d --process_list inpaint --seq_list $seq_name_with_suffix --rebuild
```
Tip: adjust the conditioning-frame selection strategy via `cond_select_strategy` in `sequence_config.py`.

Inpaint-Anything usage tips:
1. Left-click hand to add an occluded prompt in the opening image window.
2. Right-click and choose **Panning left** from the poping menu.
3. Enter the desired **inpaint selected number** in the terminal.
4. Follow the Cutie segmentation tips to obtain the object mask after inpainting.

After running the above command, `data/$seq_name/processed/` should contain:
```text
MagicHOI/
└── ./data/$seq_name/processed/inpaint
                                ├── ${selected_id}_rgba_center.png
                                └── ${selected_id}.json
```


