# ML Automate

ML Automate is a Python-based command-line tool that automates common machine learning preprocessing tasks and model training. It provides an interactive workflow, allowing users to prepare datasets and train models without repeatedly writing preprocessing code.

## Features

* Load datasets from CSV files
* Drop unnecessary columns
* Handle missing values

  * Drop rows with missing values
  * Fill numerical columns using the mean
  * Fill categorical columns using the mode
* Encode categorical features using:

  * Label Encoding
  * Ordinal Encoding
  * One-Hot Encoding
* Split data into training and testing sets
* Scale numerical features using StandardScaler
* Train machine learning models
* Display model evaluation metrics
* Interactive command-line interface

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ML-Automate.git
```

2. Navigate to the project directory:

```bash
cd ML-Automate
```

3. Install the required packages:

```bash
pip install pandas numpy scikit-learn
```

## Usage

Run the script:

```bash
python Ml_automate.py
```

The program will guide you through:

* Loading a dataset
* Cleaning missing values
* Encoding categorical features
* Selecting the target column
* Splitting the dataset
* Scaling features
* Training a machine learning model
* Viewing evaluation
