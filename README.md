# clip-deepfake-detection

This project implements a deepfake detection model that leverages the power of OpenAI's pre-trained CLIP (Contrastive Language-Image Pre-training) model. It is designed to distinguish between real and synthetically generated face images.

The approach uses a frozen CLIP vision encoder as a feature extractor and fine-tunes a small set of new layers (an "adapter" and a classification head) for the specific task of forgery detection. This method, known as parameter-efficient fine-tuning, is both computationally efficient and effective. The model's reasoning is enhanced by comparing image features against a set of text prompts describing real and fake faces.

This repository provides a complete, modular pipeline for training the detector on one type of manipulation and testing its generalization capabilities on another, unseen type.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training and Evaluation](#training-and-evaluation)
  - [Training Only](#training-only)
  - [Evaluation Only](#evaluation-only)
  - [Command-Line Arguments](#command-line-arguments)
- [Output Files](#output-files)
- [Example Workflow](#example-workflow)

## Features

-   **CLIP-Based Architecture**: Utilizes the powerful, pre-trained `ViT-B/32` CLIP model for robust visual feature extraction.
-   **Parameter-Efficient Fine-Tuning**: Freezes the large CLIP backbone and only trains a small adapter and classifier, reducing training time and computational cost.
-   **Text-Prompt Guidance**: Enhances detection by calculating the similarity between an image and descriptive text prompts (e.g., "a real human face" vs. "a synthetic face").
-   **Cross-Manipulation Evaluation**: Designed to train on one type of fake (e.g., `FaceSwap`) and test on another (e.g., `NeuralTextures`) to evaluate the model's generalization ability.
-   **Modular Codebase**: The project is cleanly structured into separate modules for data handling, model definition, training, and evaluation, promoting readability and maintainability.
-   **Comprehensive Evaluation**: Automatically calculates and reports key metrics, including AUC, EER (Equal Error Rate), F1-Score, and Accuracy. It also generates an ROC curve plot and detailed prediction CSVs.

## How It Works

1.  **Image Encoding**: An input image is passed through the frozen CLIP vision transformer to get a 512-dimensional feature vector.
2.  **Text Similarity**: This image vector is compared against feature vectors from a predefined set of text prompts (e.g., "a genuine photograph," "a digitally manipulated face"). This results in a set of similarity scores.
3.  **Feature Engineering**: The similarity scores are aggregated into meaningful features, such as the average score for "real" prompts, the average for "fake" prompts, and the difference between them.
4.  **Feature Adaptation**: The original image vector is passed through a small, trainable MLP (the "projection adapter") to learn task-specific visual features.
5.  **Classification**: The adapted visual features and the engineered text features are concatenated and fed into a final classification head, which outputs the probability of the image being real or fake.


## Project Structure

The codebase is organized into logical modules:

```
├── src/
│   ├── main.py               # Main entry point to run training and evaluation.
│   ├── dataset.py            # Defines the PyTorch Dataset classes for loading images.
│   ├── model.py              # Defines the ForgeryDetector neural network architecture.
│   ├── train.py              # Contains the ModelTrainer class for the training loop.
│   ├── evaluate.py           # Contains the ModelEvaluator and metric calculation logic.
└── README.md             # This file.
```

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   `pip` and `git`
-   (Optional but recommended) An NVIDIA GPU with CUDA for faster training.

### Installation

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

The script expects a specific directory structure for the data. You must organize your images into subfolders corresponding to their class. By default, the script is configured to use `Real_youtube`, `FaceSwap`, and `NeuralTextures`.

Create a root data directory and place your class folders inside it:

```
/path/to/your/dataset/
├── Real_youtube/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── FaceSwap/           # Used for training fakes
│   ├── fake_img_001.png
│   └── ...
└── NeuralTextures/     # Used for testing fakes
    ├── another_fake_001.png
    └── ...
```
### Dataset Source
Download from: https://www.dropbox.com/t/2Amyu4D5TulaIofv

## Usage

All operations are handled via the `main.py` script. You can control the behavior using command-line arguments.

### Training and Evaluation

This is the default mode. The model will be trained on the training classes (`Real_youtube` and `FaceSwap`) and then immediately evaluated on the test classes (`Real_youtube` and `NeuralTextures`).

```bash
python main.py --data_dir /path/to/your/dataset --output_dir ./results
```

### Training Only

To only train the model and save the final checkpoint:

```bash
python main.py --data_dir /path/to/your/dataset --output_dir ./results --mode train
```

### Evaluation Only

To evaluate a previously trained model, ensure the `final_model.pth` file exists in the specified output directory.

```bash
python main.py --data_dir /path/to/your/dataset --output_dir ./results --mode test
```

### Command-Line Arguments

Here are the most important arguments:

| Argument         | Type    | Default       | Description                                                               |
| ---------------- | ------- | ------------- | ------------------------------------------------------------------------- |
| `--data_dir`     | `str`   | **(Required)**| Path to the root dataset directory.                                         |
| `--output_dir`   | `str`   | `./results`   | Directory to save logs, models, and evaluation results.                   |
| `--mode`         | `choice`| `both`        | Execution mode: `train`, `test`, or `both`.                               |
| `--epochs`       | `int`   | `15`          | Number of training epochs.                                                |
| `--batch_size`   | `int`   | `32`          | Batch size for data loaders.                                              |
| `--lr`           | `float` | `1e-4`        | Learning rate for the AdamW optimizer.                                    |
| `--seed`         | `int`   | `42`          | Random seed for reproducibility.                                          |
| `--num_workers`  | `int`   | `4`           | Number of CPU workers for loading data.                                   |

## Output Files

After running, the specified `--output_dir` will contain:

-   `run.log`: A log file with detailed information about the run.
-   `checkpoint_epoch_X.pth`: Model checkpoints saved periodically during training.
-   `final_model.pth`: The final trained model state, saved after the last epoch.
-   `test_predictions.csv`: A CSV file with per-image predictions and scores from the evaluation phase.
-   `test_metrics.json`: A JSON file summarizing the final performance metrics (AUC, EER, etc.).
-   `roc_curve.png`: A plot of the ROC curve from the evaluation phase.

## Example Workflow

1.  Organize your dataset as described in [Dataset Preparation](#dataset-preparation).
2.  Run a full training and evaluation cycle. Let's use 10 epochs and a learning rate of 5e-5.

    ```bash
    python main.py \
        --data_dir /mnt/data/deepfake_images \
        --output_dir ./experiment_01 \
        --epochs 10 \
        --lr 5e-5
    ```

3.  Inspect the results in the `./experiment_01` folder. Check `test_metrics.json` for the final scores and `roc_curve.png` to visualize performance.
4.  If you are satisfied, you can use the `final_model.pth` for inference elsewhere. If not, you can launch a new experiment with different hyperparameters, perhaps changing the learning rate or number of epochs.
