# Spam Detection Machine Learning Tool

This tool is designed to detect spam messages using machine learning techniques. It is developed in Python and leverages
various libraries and algorithms to achieve high accuracy in spam detection.

## Features

- Vectorization: Uses TF-IDF (Term Frequency-Inverse Document Frequency) for converting text data into numerical format
  suitable for machine learning.
- Natural Language Processing: The tool integrates tokenization, lemmatization, and normalization to preprocess the text
  data efficiently.
- Classifiers: The primary classifier used is Naive Bayes (variants include GaussianNB, MultinomialNB, and BernoulliNB).
  Users are encouraged to explore other classifiers for better performance.
- Evaluation Metrics: Besides accuracy, the tool computes other metrics like AUC and F1-score, especially useful when
  there's a class imbalance in the dataset.

## Getting Started

### Dependencies

- Python 3.9+
- Required libraries: `pandas`, `nltk`, `matplotlib`, `scikit-learn` and `joblib`.

### Data

- The tool expects data in CSV format.
- The default dataset file is `data/spam.csv` and test examples are in `data/spam_test.csv`. You can modify these
  paths in the script if necessary.

### Usage

Simply run the `main.py` script to execute the spam detection tool.

```commandline
python main.py
```

The script will preprocess the data, train the classifiers, evaluate their performance, and display relevant metrics.

### Files

The script creates a folder `model` where it saves the trained models and a folder `plots` where it saves the plots
from the confusion matrices.

## Customization

### Vectorization

You can adjust the parameters of the TF-IDF vectorizer for better performance or even use other vectorization techniques
like Count Vectorization.

### Classifier Parameters

Adjust the hyperparameters of the classifiers for optimization.

### Data Preprocessing

The preprocessing steps can be modified or extended for better data cleaning and preparation.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## License

This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.

