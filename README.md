# Project: Surface Reconstruction from Gaussian Splatting: A Short Survey

This document provides instructions on how to set up the environment and datasets to reproduce the results presented in our project report. The project compares several 3D reconstruction methods (2DGS, 3DGS, Gaussian Opacity Fields, and SuGaR) on the DTU and Tanks and Temples datasets.

This guide focuses on the contents of the `Project/` directory.

## 1. Initial Setup

### 1.1. Environment Variables

To ensure the scripts can locate the project files and datasets correctly, you must define two environment variables. We recommend adding these to your `.bashrc` or `.zshrc` file.

* `3DV_DIR`: The absolute path to the root of this cloned repository.
* `3DV_DATADIR`: The absolute path to the directory where you will store all datasets.

**Example:**
If you clone the repository to `~/3DV-SU25` and create a datasets folder at `~/Datasets`, you would set the variables as follows:

```bash
export 3DV_DIR="~/3DV-SU25"
export 3DV_DATADIR="~/Datasets"
````

**Important**: You will need to ensure these environment variables are loaded in your shell and are used within the `.sbatch` scripts to construct the correct paths.

### 1.2. Conda Environments

Each reconstruction method requires its own specific environment. Furthermore, a separate environment is needed for evaluating results on the Tanks and Temples dataset.

  * **Method-Specific Environments**: The setup instructions for `2d-gaussian-splatting`, `gaussian-opacity-fields`, and `SuGaR` are located in the `README.md` file within their respective directories under `Project/Methods/`. These instructions have been tested on clusters that do not have a system-wide installation of `nvcc`. The environment names used in those guides correspond to the names used in the `.sbatch` files.

  * **Tanks and Temples Evaluation Environment**: To run the evaluation scripts for the TNT dataset, create a separate conda environment using the provided requirements file.

    ```bash
    conda create -n tnt_eval python=3.8
    conda activate tnt_eval
    pip install -r Project/tnt_eval_requirements.txt
    ```

## 2\. Dataset Download and Organization

All datasets should be downloaded and organized within the directory specified by your `$3DV_DATADIR` environment variable.

### 2.1. DTU Dataset

  * **Preprocessed Data (for 2DGS)**

    1.  Download the dataset from the official 2DGS repository: [DTU\_preprocessed](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9).
    2.  Extract the contents to `$3DV_DATADIR/DTU_2DGS`.

  * **Ground Truth Data**

    1.  Go to the DTU Robots website: [DTU MVS Challenge 2014](https://roboimagedata.compute.dtu.dk/?page_id=36).
    2.  At the bottom of the page, download `SampleSet.zip` and `Points.zip`.
    3.  Create the directory `$3DV_DATADIR/DTU`.
    4.  From the extracted `SampleSet`, copy the `Calibration` and `ObsMask` directories into `$3DV_DATADIR/DTU`.
    5.  From the extracted `Points`, copy the `Points` directory into `$3DV_DATADIR/DTU`.
    6.  Your final directory structure should look like this:
        ```
        $3DV_DATADIR/DTU/
        ├── Calibration/
        ├── ObsMask/
        └── Points/
        ```

### 2.2. Tanks and Temples (TNT) Dataset

  * **Preprocessed Data (for GOF)**

    1.  Download the preprocessed data from the Gaussian Opacity Fields Hugging Face repository: [huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main).
    2.  This includes the following scenes: `Barn`, `Caterpillar`, `Courthouse`, `Ignatius`, `Meetingroom`, `Truck`.
    3.  Save the data to `$3DV_DATADIR/TNT_GOF`.

  * **Ground Truth Data**

    1.  Go to the official Tanks and Temples website: [tanksandtemples.org/download/](https://www.tanksandtemples.org/download/).
    2.  Under the "Training Data" section, download the ground truth point cloud (`.ply`) for all scenes except `Church`.
    3.  Under the "Results on Training Data" section, download the `Camera Poses`, `Alignment`, and `Cropfiles`.
    4.  Organize the downloaded and unzipped files into their respective scene directories. For example, the `Barn` directory should look like this:
        ```
        $3DV_DATADIR/TNT/Barn/
        ├── Barn_COLMAP_SfM.log
        ├── Barn.json
        ├── Barn.ply
        └── Barn_trans.txt
        ```

## 3\. Running the Experiments

All training and evaluation jobs are launched using Slurm batch scripts located in the `Project/` directory.

  * **Output**: All Slurm log files and any script outputs will be saved to the `Project/slurm/` directory.
  * **sbatch Files**:
      * `2dgs_dtu.sbatch`: Train and evaluate 2DGS on the DTU dataset.
      * `2dgs_tnt.sbatch`: Train and evaluate 2DGS on the TNT dataset.
      * `3dgs_train_dtu.sbatch`: Train 3DGS on the DTU dataset.
      * `3dgs_eval_dtu.sbatch`: Evaluate the trained 3DGS model on the DTU dataset. (Note: 3DGS failed to run on the TNT dataset).
      * `gof_dtu.sbatch`: Train and evaluate Gaussian Opacity Fields on the DTU dataset.
      * `gof_tnt.sbatch`: Train Gaussian Opacity Fields on the TNT dataset.
      * `mesh_gof_tnt.sbatch`: Extracts the mesh from the trained GOF model for TNT. This is a separate CPU job because the learned representation was too large to process on a GPU.
      * `sugar_train_dtu.sbatch`: Train SuGaR on the DTU dataset.
      * `sugar_eval_dtu.sbatch`: Evaluate the trained SuGaR model on the DTU dataset.
      * `sugar_train_tnt.sbatch`: Train SuGaR on the TNT dataset.
      * `sugar_eval_tnt.sbatch`: Evaluate the trained SuGaR model on the TNT dataset.

To run a job, simply use `sbatch`:

```bash
sbatch Project/2dgs_dtu.sbatch
```
