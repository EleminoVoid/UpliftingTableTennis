# Uplifting Table Tennis

Personal fork of the official implementation for **"Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation"**.

This fork is configured for local development from `EleminoVoid/UpliftingTableTennis` while keeping the original parent repository available as read-only upstream reference. The codebase combines table keypoint detection, ball detection, camera calibration, and 2D-to-3D trajectory uplifting for table tennis videos.

Original project:

- Paper: <https://arxiv.org/abs/2511.20250>
- Project page: <https://kiedani.github.io/WACV2026/index.html>
- Upstream repository: <https://github.com/KieDani/UpliftingTableTennis>

## Fork Status

This repository is no longer treated as a plain clone of the parent project. It has local fork maintenance changes for Windows/Python setup, table-detection training/evaluation, and repository hygiene.

Current remotes:

```text
origin   https://github.com/EleminoVoid/UpliftingTableTennis.git  fetch/push
upstream https://github.com/KieDani/UpliftingTableTennis.git      fetch only
```

Local git settings used in this checkout:

```text
remote.pushDefault = origin
push.default = simple
pull.ff = only
fetch.prune = true
remote.upstream.pushurl = DISABLED
```

Daily fork workflow:

```bash
git status
git pull --ff-only origin main
git push origin main
```

Only pull from the parent repository when you intentionally want upstream changes:

```bash
git fetch upstream
git merge upstream/main
```

## Repository Layout

```text
balldetection/       Ball detector training, datasets, model helpers
tabledetection/      Table keypoint detection training and inference
uplifting/           2D-to-3D trajectory and spin estimation model
inference/           Evaluation and end-to-end inference scripts
dataprocessing/      Dataset preparation and annotation utilities
syntheticdataset/    MuJoCo-based synthetic trajectory generation
tutorials/           Torch Hub usage examples
scripts/             Local evaluation and maintenance utilities
paths.py             Local data, weight, and log path configuration
```

## Local Changes

Compared with the upstream baseline, this fork currently includes:

- Windows/Python setup notes and pinned Python 3.12 guidance.
- Local `paths.py` values pointing to `C:/Users/312_Lab/Documents/GitHub/Uplifting/...`.
- `.gitignore` coverage for virtual environments, Python caches, logs, weights, datasets, zips, and editor files.
- `inference/inference_tabledetection.py` now writes table-detection evaluation summaries to `logs/evaluation/tabledetection/`, including PNG, PDF, and CSV-style text output.
- `tabledetection/train.py` accepts `--batch_size` for lower-memory training runs.
- `tabledetection/config.py` sanitizes run directory names so Windows does not reject characters such as `:` in experiment IDs.
- `tabledetection/dataset.py` auto-detects comma or semicolon CSV delimiters for `table_detection.csv`.
- `tabledetection/helper_tabledetection.py` reduces temporary tensor allocation in the weighted MSE loss.
- `scripts/prune_missing_images.py` prunes TTHQ CSV rows whose image files are missing, writing timestamped backups.
- `scripts/evaluate_training.py` summarizes the most recent table-detection TensorBoard training logs.
- `TECHNICAL_REPORT_TABLE_KEYPOINT_DETECTION.md` documents the current table keypoint detection method and results.
- `activate.bat` and `activate.ps1` are local convenience activation scripts.

## Installation

### Python Version

Use Python **3.12.4**. This project is not currently set up for Python 3.14 on Windows. With newer interpreters, `pip` may try to build packages such as NumPy or Matplotlib from source and fail without a compiler toolchain.

Check your Python version:

```bash
python --version
```

Create a Python 3.12 virtual environment:

```bash
py -3.12 -m venv .venv
```

Activate it on PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Activate it on Command Prompt:

```bat
.\.venv\Scripts\activate.bat
```

Then update packaging tools:

```bash
python -m pip install --upgrade pip wheel
```

### Inference-Only Installation

Use this path if you only need the pretrained models and inference scripts.

Install PyTorch first:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
```

Then install the inference requirements:

```bash
pip install -r requirements.txt
```

### Full Installation

Use this path if you want training, testing, evaluation, and synthetic data work.

Install PyTorch first:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
```

Install the full requirements:

```bash
pip install -r requirements_full.txt
```

Install MMCV if you need VitPose:

```bash
pip install mmcv==2.2.0
```

VitPose depends on `mmcv`, which can be difficult on Windows. If you are not using VitPose, skip `mmcv` and comment out the VitPose model paths in `inference/inference_balldetection.py` and `inference/inference_tabledetection.py`.

### Quick Import Check

After installing, run:

```bash
python -c "import torch, cv2, numpy, pandas; print('imports ok'); print(torch.__version__)"
```

## Local Paths

This fork currently uses:

```python
data_path    = 'C:/Users/312_Lab/Documents/GitHub/Uplifting/data'
weights_path = 'C:/Users/312_Lab/Documents/GitHub/Uplifting/weights'
logs_path    = 'C:/Users/312_Lab/Documents/GitHub/Uplifting/logs'
```

Those values live in `paths.py`. They are machine-specific and should be changed if the repo is moved to another workstation.

Recommended folder shape:

```text
C:/Users/312_Lab/Documents/GitHub/Uplifting/
  data/
  weights/
  logs/
  UpliftingTableTennis/
```

## Weights

Download the official weights:

<https://mediastore.rz.uni-augsburg.de/get/TL7oQRStHG/>

Unzip them into the folder configured by `weights_path`. The inference scripts expect:

```text
weights/
  inference_balldetection/
    segformerpp_b2/model.pt
    wasb/model.pt
  inference_tabledetection/
    segformerpp_b2/model.pt
    hrnet/model.pt
  inference_uplifting/
    ours/model.pt
```

## Data Setup

Set `data_path` in `paths.py` to the folder containing datasets. Large datasets should stay outside git.

### TTHQ

TTHQ is the dataset created for the WACV 2026 paper.

1. Review `video_list.txt` for suggested source videos and download notes.
2. Create this folder:

```text
<data_path>/tthq_videos/
```

3. Copy downloaded videos into `tthq_videos`.
4. Download `tthq_annotation.zip`:

<https://mediastore.rz.uni-augsburg.de/get/E6idNDRk20/>

5. Unzip it into `tthq_videos`.
6. Extract the dataset:

```bash
python -m dataprocessing.extract_tthq_data
```

If CSV rows point to frames that are not present locally, use the local pruning helper:

```bash
python scripts/prune_missing_images.py
```

The pruning script creates timestamped `.bak` files before rewriting CSVs.

### TTST

TTST is introduced in **"Towards ball spin and trajectory analysis in table tennis broadcast videos via physically grounded synthetic-to-real transfer"**.

Download the updated dataset:

<https://mediastore.rz.uni-augsburg.de/get/jprwueaZYd/>

Unpack it into `data_path`.

Note: this public dataset package does not include all frames required for full-pipeline evaluation. It can be used for uplifting evaluation, but full TTST pipeline evaluation requires the full dataset.

### BlurBall

BlurBall is used for ball-detection pretraining.

1. Create:

```text
<data_path>/blurball/
```

2. Download the dataset:

<https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7>

3. Follow the README included with the downloaded dataset.

### Synthetic Dataset

The synthetic dataset is generated with MuJoCo for 2D-to-3D uplifting.

On Linux/headless systems, start a virtual display:

```bash
xvfb-run -a -s "-screen 0 1400x900x24" bash
```

Generate the dataset:

```bash
python -m syntheticdataset.mujocosimulation --num_trajectories 50000 --num_processes 96 --mode intermediate --direction left_to_right --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 50000 --num_processes 96 --mode intermediate --direction right_to_left --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode first_good --direction left_to_right --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode first_good --direction right_to_left --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_short --direction left_to_right --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_short --direction right_to_left --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_long --direction left_to_right --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_long --direction right_to_left --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_win --direction left_to_right --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_win --direction right_to_left --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_lose --direction left_to_right --folder syntheticdata
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_lose --direction right_to_left --folder syntheticdata
```

This can take days. Adjust `--num_processes` to match available CPU resources.

The original project also provides the synthetic dataset as split downloads:

- Part 1: <https://mediastore.rz.uni-augsburg.de/get/zj3aBN4U9N/>
- Part 2: <https://mediastore.rz.uni-augsburg.de/get/gCq5s6s8EJ/>
- Part 3: <https://mediastore.rz.uni-augsburg.de/get/9HIr7_yQFe/>

Save the parts in `data_path`, then recombine and unzip on a Unix-like shell:

```bash
cat syntheticdata.zip.part* > syntheticdata.zip
unzip syntheticdata.zip
rm syntheticdata.zip.part*
```

## Inference

Run ball detection evaluation:

```bash
python -m inference.inference_balldetection --gpu 0
```

Run table keypoint detection evaluation:

```bash
python -m inference.inference_tabledetection --gpu 0
```

To choose the output folder for table-detection summaries:

```bash
python -m inference.inference_tabledetection --gpu 0 --output_dir logs/evaluation/tabledetection
```

The table-detection evaluation writes:

```text
logs/evaluation/tabledetection/tabledetection_pck_summary.png
logs/evaluation/tabledetection/tabledetection_pck_summary.pdf
logs/evaluation/tabledetection/tabledetection_pck_summary.txt
```

Run 2D-to-3D uplifting evaluation:

```bash
python -m inference.inference_uplifting --gpu 0
```

Run the full pipeline on TTHQ:

```bash
python -m inference.inference_combined --dataset tthq --gpu 0
```

Run the full pipeline on TTST only when full TTST frames are available:

```bash
python -m inference.inference_combined --dataset ttst --gpu 0
```

## Training

Set `logs_path` in `paths.py` before training. This fork writes training logs and saved models under that root.

### Ball Detection

Train with official pretraining weights:

```bash
python -m balldetection.train --gpu 0 --folder results --pretraining --model_name segformerpp_b2
```

Available model names include:

```text
segformerpp_b0
segformerpp_b2
wasb
vitpose
```

Run BlurBall pretraining yourself:

```bash
python -m balldetection.train --gpu 0 --folder pretraining --model_name segformerpp_b2 --data blurball
```

### Table Keypoint Detection

Train the default table detector:

```bash
python -m tabledetection.train --gpu 0 --folder results --model_name segformerpp_b2 --data tthq
```

Use a smaller batch size on limited GPU memory:

```bash
python -m tabledetection.train --gpu 0 --folder results --model_name segformerpp_b2 --data tthq --batch_size 2
```

Available table model names include:

```text
segformerpp_b0
segformerpp_b2
hrnet
vitpose
```

After training, summarize the latest batch-size-2 table-detection run:

```bash
python scripts/evaluate_training.py
```

### 2D-to-3D Uplifting

Train the current default model:

```bash
python -m uplifting.train --gpu 0 --folder results --token_mode dynamic --time_rotation new
```

Paper reproduction options:

```text
Kienzle et al: --token_mode originalmethod --time_rotation old
Mixed:       --token_mode originalmethod --time_rotation new
Ours:        --token_mode dynamic --time_rotation new
```

## Current Table Keypoint Results

This fork includes a local technical report:

```text
TECHNICAL_REPORT_TABLE_KEYPOINT_DETECTION.md
```

Current logged table-detection summary:

```text
segformerpp_b2: PCK@2 = 0.5086, PCK@5 = 0.8793, PCK@10 = 0.9310, PCK@20 = 0.9569
segformerpp_b0: PCK@2 = 0.3793, PCK@5 = 0.8793, PCK@10 = 0.9310, PCK@20 = 0.9569
```

## Torch Hub Tutorials

The original project exposes Torch Hub helpers in `hubconf.py`.

Tutorials:

- `tutorials/full_pipeline.md`
- `tutorials/ball_detection.md`
- `tutorials/table_detection.md`

If using this fork through Torch Hub, replace the upstream repo string with your fork when needed:

```python
repo = "EleminoVoid/UpliftingTableTennis"
```

## Citation

If you use the original research work, cite:

```bibtex
@inproceedings{kienzle2026uplifting,
  title={Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation},
  author={Kienzle, Daniel and Ludwig, Katja and Lorenz, Julian and Satoh, {Shin'ichi} and Lienhart, Rainer},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## License

See `LICENSE.txt`.
