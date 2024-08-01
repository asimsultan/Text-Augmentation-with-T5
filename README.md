
# Text Augmentation with T5

Welcome to the Text Augmentation with T5 project! This project focuses on augmenting text data using the T5 model.

## Introduction

Text augmentation involves generating additional text samples based on existing text. In this project, we leverage the power of T5 to perform text augmentation using a dataset of text pairs.

## Dataset

For this project, we will use a custom dataset of text pairs. You can create your own dataset and place it in the `data/augmentation_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/t5_text_augmentation.git
cd t5_text_augmentation

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes text pairs for augmentation. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: text and augmented_text.

# To fine-tune the T5 model for text augmentation, run the following command:
python scripts/train.py --data_path data/augmentation_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/augmentation_data.csv
