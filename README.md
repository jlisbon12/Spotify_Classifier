# Spotify Music Analysis

## Overview

This project is designed to analyze and predict various characteristics of music tracks using a dataset that includes features like popularity, danceability, energy, and more. The project utilizes machine learning models to classify tracks and provide insights into music trends.

## Project Structure

The project is divided into several Python scripts, each handling different aspects of the machine learning pipeline:

- nn.py: Implements neural network models for regression or classification tasks.
- normalize.py: Contains functions to scale and normalize the feature data.
- preprocessing.py: Preprocesses the data, including handling missing values, encoding categorical variables, and more.
- train_test_split.py: Splits the data into training and testing sets.
- dimensionality_reduction.py: Reduces the number of features in the dataset using techniques like PCA.
- EDA.py: Conducts exploratory data analysis to visualize and understand data patterns and distributions.
- classifier.py: Defines and trains classification models and evaluates their performance.

## Setup

To run this project, you will need Python 3.8 or later, and several dependencies including NumPy, Pandas, scikit-learn, TensorFlow, and Matplotlib.

## Installation
Clone the repository:
```bash
git clone https://github.com/jlisbon12/spotify_classifier.git
```
Install the required libraries:
```bash
pip install -r requirements.txt
```

## Usage

To use this project, navigate to the project directory and run the Python scripts in order. Here is a typical workflow:

Data Preprocessing:
```bash
python preprocessing.py
```
Normalization:
```bash
python normalize.py
```
Dimensionality Reduction:
```bash
python dimensionality_reduction.py
```

Model Training:

For neural networks:
```bash
python nn.py
```
For other classifiers:
```bash
python classifier.py
```