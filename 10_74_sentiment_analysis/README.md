# Sentiment Analysis

## Overview

This script performs sentiment analysis on a dataset containing tweets. Tweets are labeled as positive, negative, or
neutral. The main objective is to preprocess the data, build a classifier (Logistic Regression), and assess its
performance.

## Features

- Dataset preprocessing, including lemmatization and conversion to lowercase.
- Text vectorization using TF-IDF.
- Training and evaluation of a Logistic Regression classifier.
- Performance metrics include accuracy, with considerations for AUC and F1-score.
- Extra functionalities include building an extra binary classifier for positive and negative tweets only (no neutral)
  and interpreting the weights of the model by outputting the top 10 most positive and negative words.

## Getting Started

### Dependencies

- Python 3.9+
- Required libraries: `pandas`, `nltk`, `matplotlib`, `scikit-learn` and `joblib`.
- Custom modules: `common.utils`

### Data

The tweets are on a file `data/Tweets.csv`. And the original dataset is
from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).

### Usage

Simply run the `main.py` script to execute the sentiment analysis tool and deploy the model.

```bash
python main.py
```

The script will preprocess the data, train the classifiers, evaluate their performance, and display relevant metrics.

### Files

The script creates a folder `cache` where it saves the trained models (`model.joblib` and `model_binary.joblib`)
and a folder `plots` where it saves the plots from the confusion matrices.

- The `binary` joblib files are used when it is a binary classifier. E.g., positive and negative tweets.
- The non `binary` joblib files are used when it is a multiclass classifier. E.g., positive, negative, and neutral
  tweets.

## Notes

- The script uses the `utils` module to load the data and perform the preprocessing steps. The module is located in
  the `common` folder.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## License

This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.

