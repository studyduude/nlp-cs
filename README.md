# NLP Text Classification

## Overview
This repository contains a Python-based text classification project using BERT, aimed at classifying text into various predefined categories. It employs an active learning approach for iterative model improvement.

## Project Structure
- `data/`: Directory containing training and test datasets.
- `logs/`: Log files for model training and active learning iterations.
- `results/`: Output from model predictions, including evaluation metrics and saved model states.
- `classification.ipynb`: Jupyter notebook for interactive data exploration and model evaluation.
- `main.py`: The main script for running the classification model and active learning loop.
- `change_csv.py`: Script to manipulate CSV files as needed for data preprocessing.

## Installation
Set up your Python environment and install dependencies with:
```bash
pip install -r requirements.txt
```
## Usage

To get started with this text classification project, follow these instructions:

### Running the Main Script

To initiate the training and active learning loop, execute:

```bash
python main.py
```
### Interactive Analysis with Jupyter Notebooks
If you want to dive deeper into the data and models interactively, you can use the Jupyter notebook: `classification.ipynb`

Ensure that you have Jupyter installed and that all required packages are included in your Python environment.
