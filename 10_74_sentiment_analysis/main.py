"""
There's a dataset with tweets, and they are labeled positive, negative, or neutral.
The dataset contains irrelevant information, so you'll need to preprocess the data.
Build a classifier and assess train/test performance.
Use a vectorization technique to convert the text into a numerical representation.
You can use TF-IDF or Count Vectorization.
Options: lemma, stem, remove stopwords, remove punctuation, lowercase, etc.
Classifier: Logistic Regression.
Score function returns accuracy.
Check for class imbalance.
If there is a class imbalance, check metrics like AUC / F1-score. And plot confusion matrix.
AUC and F1 are not available for multiclass classification, so we need to see on sci-kit learn documentation how to use
them for multiclass classification.

Extra:
- Build a binary classifier for positive and negative tweets.
- Interpret the weights of the logistic regression model by printing the top 10 most positive and negative words.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common.utils import CacheManager, PreprocessDocuments, ClassifierResult
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_CSV = "data/Tweets.csv"


def load_documents(file_path: str, binary: bool = False) -> pd.DataFrame:
    """
    Load the documents from the csv file.

    Parameters:
    - file_path (str): The path to the csv file.
    - binary (bool): Whether to use binary classification or not.

    Returns:
    - The documents as a pandas DataFrame.
    """
    df = pd.read_csv(file_path, encoding="latin1")
    df = df[["text", "airline_sentiment"]]
    df = df.rename(columns={"airline_sentiment": "label"})
    # Make everything lowercase
    df["text"] = df["text"].str.lower()
    if binary:
        # Drop all the rows that have the neutral label
        df = df[df["label"] != "neutral"]
    df = df.dropna()
    return df


def load_data(file_path: str, binary: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load the data from cache if it is in cache, otherwise compute, save, and return.

    Parameters:
    - file_path (str): The path to the csv file.
    - binary (bool): Whether to use binary classification or not.

    Returns:
    - The documents and the labels.
    """
    cache_manager = CacheManager()

    print(f'Load the documents')
    # Change the filename depending on whether we want to use binary classification or not
    filename = "docs_binary.joblib" if binary else "docs.joblib"

    # Load the documents from cache if they are in cache, otherwise compute, save, and return
    docs = cache_manager.get_or_compute(filename, load_documents, file_path=file_path, binary=binary)
    documents = docs["text"].tolist()
    labels = docs["label"].tolist()
    return documents, labels


def load_vectorizer(documents: List[str], binary: bool = False) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """
    Load the vectorizer and the vectorized documents from cache if they are in cache, otherwise compute, save, and
    return.

    Parameters:
    - documents (List[str]): The documents to vectorize.
    - binary (bool): Whether to use binary classification or not.

    Returns:
    - The vectorizer and the vectorized documents.
    """
    cache_manager = CacheManager()
    preprocess_documents = PreprocessDocuments()

    print('Preprocess the documents with TfIdfVectorizer')
    # Change the filename depending on whether we want to use binary classification or not
    filename_vec = "vectorizer_binary.joblib" if binary else "vectorizer.joblib"
    filename_vec_fit_tra = "vectorizer_fit_transform_binary.joblib" if binary else "vectorizer_fit_transform.joblib"

    # Load the vectorizer and the vectorized documents from the cache
    vectorizer = cache_manager.load(filename_vec)
    vectorizer_fit_transform = cache_manager.load(filename_vec_fit_tra)

    # If not in cache, compute, save, and return
    if vectorizer is None or vectorizer_fit_transform is None:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, strip_accents='unicode',
                                     tokenizer=preprocess_documents.tokenize_lemmatize_document,
                                     token_pattern=None)
        vectorizer_fit_transform = vectorizer.fit_transform(documents)

        # Save the vectorizer and the vectorized documents to cache
        cache_manager.save(vectorizer, filename_vec)
        cache_manager.save(vectorizer_fit_transform, filename_vec_fit_tra)
    return vectorizer, vectorizer_fit_transform


def load_model(X_train, y_train, max_iter: int = 200, binary: bool = False) -> LogisticRegression:
    """
    Load the model from cache if it is in cache, otherwise compute, save, and return.

    Parameters:
    - X_train: The training data.
    - y_train: The training labels.
    - max_iter (int): The maximum number of iterations.
    - binary (bool): Whether to use binary classification or not.

    Returns:
    - The LogisticRegression model.
    """
    cache_manager = CacheManager()

    print(f'Fit the model with LogisticRegression')
    # Change the filename depending on whether we want to use binary classification or not
    filename = "model_binary.joblib" if binary else "model.joblib"

    # Load the model from cache
    model = cache_manager.load(filename)

    # If not in cache, compute, save, and return
    if model is None:
        print(f'Computing and saving {filename} to cache...')
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)
        cache_manager.save(model, filename)

    return model


def print_top_features(vectorizer: TfidfVectorizer, clf: LogisticRegression, n: int = 10) -> None:
    """
    Print the top n most positive and negative words.

    Parameters:
    - vectorizer (TfidfVectorizer): The vectorizer used to vectorize the documents.
    - clf (LogisticRegression): The classifier used to classify the documents.
    - n (int): The number of top words to print.

    Returns:
    - None
    """
    feature_names = vectorizer.get_feature_names_out()
    # Get the top 10 most positive and negative words
    coefs = clf.coef_[0]
    # Sort the coefficients in ascending order
    top10 = np.argsort(coefs)[-n:]
    # Sort the coefficients in descending order
    bottom10 = np.argsort(coefs)[:n]

    print(f'Top {n} most positive words:')
    for idx in top10:
        print(f'{feature_names[idx]}')
    print(f'Top {n} most negative words:')
    for idx in bottom10:
        print(f'{feature_names[idx]}')


def main(binary: bool = False):
    print(f'{20 * "*"}')
    print(f'Running the sentiment analysis with binary={binary}')

    documents, labels = load_data(DATASET_CSV, binary)
    vectorizer, vectorizer_fit_transform = load_vectorizer(documents, binary)

    print(f'Train and test split')
    # Use the train_test_split function to split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectorizer_fit_transform, labels,
                                                        test_size=0.2, random_state=42)

    # Load the model from cache if it is in cache, otherwise compute, save, and return
    clf = load_model(X_train, y_train, binary=binary)

    multiclasses = False if binary else True
    classifier = ClassifierResult(clf, 'LogisticRegression', multiclasses=multiclasses)
    classifier.train(X_train, y_train)

    metrics = classifier.evaluate(X_train, y_train, X_test, y_test)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")

    # Attach an extra label to the confusion matrix if we are using binary classification
    extra_label = '_binary' if binary else ''
    classifier.plot_confusion_matrix(X_test, y_test, extra_label=extra_label)
    classifier.print_classification_report(X_train, y_train, X_test, y_test)

    # If we are using binary classification, print the top 10 most positive and negative words
    if binary:
        print_top_features(vectorizer, clf, n=10)


if __name__ == '__main__':
    main(binary=False)
    main(binary=True)
