
## Download Hand Models

Follow these steps to populate the hand model assets required by MagicHOI:

1. Register for a MANO account at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) and download `mano_v1_2.zip` from [this link](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip).
2. Extract the archive into `<project_dir>/code/body_models/`.
3. Download the additional assets [contact_zones.pkl](https://hkustgz-my.sharepoint.com/:u:/g/personal/swang457_connect_hkust-gz_edu_cn/EaP-42278YJMsH3IMGwibpcBo34ZTqn4jBcHlF0dx4qa4g?e=UIRBhF) and [sealed_vertices_sem_idx.npy](https://hkustgz-my.sharepoint.com/:u:/g/personal/swang457_connect_hkust-gz_edu_cn/EZrHuHMx5IZGmjgWin0fyLoBK5Nb3C1-9xypSdNA95Wlvw?e=nKyUAd), then place them in the same folder.

After completing the steps above, the directory should look like the tree below:

```text
MagicHOI/
└── code/
    └── body_models/
        ├── LICENSE.txt
        ├── MANO_LEFT.pkl
        ├── MANO_RIGHT.pkl
        ├── SMPLH_female.pkl
        ├── SMPLH_male.pkl
        ├── contact_zones.pkl
        ├── info.txt
        └── sealed_vertices_sem_idx.npy
```
## Download Zero123-XL
MagicHOI uses the Zero123-XL novel-view synthesis prior to guide object reconstruction in unobserved regions. Retrieve the weights as follows:

1. Run `bash load/zero123/download.sh` to fetch the checkpoint, or download it manually from [zero123-xl.ckpt](https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt).
2. Place the resulting `zero123-xl.ckpt` file in `<project_dir>/load/zero123/` (the script stores it there automatically).


After the download, the directory should look like the tree below:

```text
MagicHOI/
└── load/
    └── zero123/
        ├── download.sh
        ├── sd-objaverse-finetune-c_concat-256.yaml
        └── zero123-xl.ckpt
```


## Download Dataset for Training

Download the preprocessed training sequences from [data.zip](https://hkustgz-my.sharepoint.com/:u:/g/personal/swang457_connect_hkust-gz_edu_cn/EUQyDbz62ZBEkKDptmyGI0QB2xVXl1Ad_yGDiQvorv8KnA?e=kM4ubM) and unpack the archive inside `<project_dir>/data/`:

After extraction, the directory should resemble the tree below (only representative files shown; each sequence contains per-frame data):

```text
MagicHOI/
└── data/
    ├── hold_ABF12_ho3d/
    │   └── processed/
    │       ├── colmap_2d/
    │       ├── colmap_hold_ABF12_ho3d/
    │       ├── colmap_hold_ABF12_ho3d.0/
    │       ├── hold_fit.slerp.npy
    │       ├── images/
    │       ├── inpaint/
    │       ├── j2d.full.npy
    │       ├── masks/
    │       └── rgbas/
    ├── hold_ABF14_ho3d/
    │   └── processed/
    │       └── ...
    ├── ...
    └── hold_SM2_ho3d/
        └── processed/
            ├── colmap_2d/
            ├── colmap_hold_SM2_ho3d/
            ├── colmap_hold_SM2_ho3d.0/
            ├── hold_fit.slerp.npy
            ├── images/
            ├── inpaint/
            ├── j2d.full.npy
            ├── masks/
            └── rgbas/
```

# Setup HO3D Data (optional)

To compare reconstructions with the HO3D benchmark in `ait_viewer` and evaluate the hand-object recontruction results, download `HO3D_v3.zip` from the [HO3D repository](https://github.com/shreyashampali/ho3d) and extract it into `<project_dir>/ho3d_v3/`. After extraction the folder should resemble the following (only top-level entries shown here; each sequence folder holds per-frame data):

```text
MagicHOI/
└── ho3d_v3/
    ├── calibration/          # camera intrinsics/extrinsics for train sequences
    │   ├── ABF1/
    │   ├── AP1/
    │   └── … (per-sequence folders)
    ├── evaluation/           # evaluation sequences
    │   ├── AP10/
    │   ├── MPM10/
    │   └── … (per-sequence folders)
    ├── evaluation.txt        # list of frames used for evaluation
    ├── manual_annotations/   # additional manually annotated frames (.npy)
    ├── models/               # object models provided by HO3D
    ├── processed/            # preprocessed annotation data
    ├── train/                # training sequences
    └── train.txt             # list of frames used for training
```

If you add newer releases of HO3D, ensure these baseline directories remain so the provided scripts continue to work.
