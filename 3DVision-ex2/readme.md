# Hands-on AI based 3D Vision - Homework 2

Group Member 1:
- Name:
- Matriculation number: 

Group Member 2: 
- Name: 
- Matriculation number:   


# Theoretical Exercises (5 Points)
This time you are lucky. You will only have two exercises for theoretical part (2.5 points each).

The file Assignments-ex2.pdf contains the theoretical exercise, focusing on epipolar geometry.

The assignments are designed to deepen your understanding on the topic. You are encouraged to complete all theoretical questions. 

Submission: Please submit a single PDF document containing your answers to all theoretical questions.

# Programming Exercises (60 Points)

## Getting Started

### Environment Setup

We suggest you to use Visual Studio Code with the python extension (ms-python.python).

#### Uninstall old environment
If you want to uninstall the environment used during exercise 1, you can run:
```
conda remove -n 3DVision-ex1 --all
```

You can also choose to keep that environment and use it to run exercise 2. However, you will need to integrate your install a couple of new dependencies (from requirements.txt)


#### Install the new environment

**Using Python 3.10 or newer** 

you have two options:

#### 1) Conda
First install conda following the official installation guide: [Conda installation instructions](https://www.anaconda.com/docs/getting-started/miniconda/install).<br>
Once installed, create a virtual environment with conda as follows:
```
conda create -n 3DVision-ex2 python==3.12
conda activate 3DVision-ex2
```

#### 2) Python venv

If you prefer venv, you can run:
```
python3 -m venv venv
source venv/bin/activate
```

### Dependencies installation

Next, install PyTorch using the instructions [here](https://pytorch.org/get-started/locally/). Select pip as the installation method. **If you're on Linux and have a CUDA-capable GPU, select the latest CUDA version.** This will give you a command like this:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

then, install cmake (Linux) with:
```bash
sudo apt-get install cmake
pip install cmake
```
cmake (Mac) with:
```bash
brew install cmake
pip install cmake
```

for windows, you can download binaries from [here](https://cmake.org/download/), and only after installation you can run `pip install cmake`

Finally, install this homework's other dependencies:

```
pip install -r requirements.txt
```

You can now open the project directory in VS Code. Within VS Code, open the command palette (<key>⌘ command</key>/<key>ctrl</key> + <key>⇧ shift</key> + <key>P</key>), run `Python: Select Interpreter`, and choose the virtual environment you created in the previous steps.

## Project Components

### Part 1: Eight Point Algorithm (20 Points)
#### **Notebook**: 01_Eight_Point_Algorithm.ipynb

In this part you will implement the **normalized eight-point algorithm** to compute the **fundamental matrix** from correspondences between two images. You will also see how to extract relative camera pose from the essential matrix, which will be used to triangulate points in 3D space.

### Part 2: Epipolar Geometry and View Morphing (14 Points)

#### **Notebook**: 02_Epipolar_Geometry_and_View_Morphing.ipynb

In this part, you will implement key components related to **epipolar geometry**, such as computing the **fundamental matrix**, **epipolar lines**, and visualizing **correspondences**. These elements are essential for understanding the geometric relationship between two views of a scene.

You will also implement parts of a **view morphing** pipeline, which uses these geometric principles to synthesize novel views between two input images.

### Part 3: Stereo Matching with Classical Methods (16 Points)

#### **Notebook**: 03_Stereo_Vision.ipynb

This section focuses on classical stereo matching. You will:
- Implement **cost functions** to compare image patches between left and right views,
- Apply **patch matching** techniques to build a cost volume,
- And finally, compute **disparity maps** and recover **depth** from stereo pairs.

This part will give you hands-on experience with traditional stereo vision pipelines.

### Part 4: Learning-Based Stereo Matching (10 Ponts)

#### **Notebook**: 04_CNNs_for_Stereo_Vision.ipynb

Here, you’ll revisit stereo matching — but instead of hand-crafted cost functions, you’ll use **convolutional neural networks** to learn similarity between image patches.

You’ll implement and train a small CNN (the Fast Architecture proposed in [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://arxiv.org/abs/1510.05970)) to predict a cost volume, and then extract the disparity map similarly to Part 3.

> 💡 You can use Google Colab to run this notebook, especially for training the network if you do not have access to a GPU locally.


--------------------------------------

## Submission Policy

We expect a `group_${group_number}.zip` file with all the codebase and the pdf containing the answers to the theoretical part inside.

You can send the .zip file to:
- ta_3dvision@listserv.uni-tuebingen.de

**Deadline: ** 27.05 12:00 PM

## [Optional] Bonus Problem

Each homework will have a bonus problem that we will use to allocate bonus points. **These problems are completely optional.**.