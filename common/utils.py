import os
from typing import List, Dict

import nltk
import numpy as np
from joblib import load, dump
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, \
    classification_report


class PreprocessDocuments:
    """
    Preprocesses documents by tokenizing, lemmatizing, and normalizing
    """

    @staticmethod
    def get_wordnet_pos(tag: str) -> str:
        """
        Map POS tag to first character lemmatize() accepts
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def tokenize_lemmatize_document(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize the text
        """
        wnl = nltk.stem.WordNetLemmatizer()
        tokenize = nltk.tokenize.treebank.TreebankWordTokenizer()
        tokens = tokenize.tokenize(text)
        return [wnl.lemmatize(word, pos=self.get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(tokens)]


class CacheManager:
    """
    A class to manage caching of variables to files.

    Examples of using the cache manager to cache a variable:
    cache_manager = CacheManager()
    cache_manager.save(variable, "filename")
    variable = cache_manager.load("filename")

    Example of using the cache manager to cache the result of a function:
    def expensive_computation(x, y):
        return x * y

    cache = CacheManager()
    result = cache.get_or_compute("product.joblib", expensive_computation, 10, 20)

    Parameters:
        - folder (str): The folder where the cache files will be stored.

    """

    def __init__(self, folder: str = "cache"):
        """
        Initialize the CacheManager with a specific folder to save and load dumps.

        Parameters:
        - folder (str): The folder where the cache files will be stored. Defaults to "cache".
        """
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _get_file_path(self, filename: str) -> str:
        """Get the full path of the cache file."""
        return os.path.join(self.folder, filename)

    def save(self, variable, filename: str):
        """
        Save a variable to a file.

        Parameters:
        - variable: The variable to be saved.
        - filename (str): The name of the file to save the variable to.
        """
        file_path = self._get_file_path(filename)
        dump(variable, file_path)

    def load(self, filename: str):
        """
        Load a variable from a file.

        Parameters:
        - filename (str): The name of the file to load the variable from.

        Returns:
        - Loaded variable or None if the file doesn't exist.
        """
        file_path = self._get_file_path(filename)
        if os.path.exists(file_path):
            return load(file_path)
        return None

    def get_or_compute(self, filename: str, compute_function, *args, **kwargs):
        """
        Get the variable from cache, or compute and save it if it doesn't exist.

        Parameters:
        - filename (str): The name of the cache file.
        - compute_function (callable): The function to compute the variable if not found in cache.
        - *args, **kwargs: Arguments to pass to the compute_function.

        Returns:
        - The variable, either loaded from cache or computed.
        """
        # Try to load the variable from cache
        cached_data = self.load(filename)

        # If not in cache, compute, save, and return
        if cached_data is None:
            print(f'Computing and saving {filename} to cache...')
            computed_data = compute_function(*args, **kwargs)
            self.save(computed_data, filename)
            return computed_data

        return cached_data

    def purge(self, filename: str = None):
        """
        Delete the cache file(s).

        Parameters:
        - filename (str): The name of the cache file to delete. If None, all cache files will be deleted.
        """
        if filename:
            file_path = self._get_file_path(filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        else:
            for file in os.listdir(self.folder):
                file_path = os.path.join(self.folder, file)
                os.remove(file_path)


class ClassifierResult:
    """
    A class to manage the results of a classifier.

    Parameters:
        - model: The model used for classification.
        - model_name (str): The name of the model.
    """

    def __init__(self, model, model_name, multiclasses: bool = False):
        self.model = model
        self.model_name = model_name
        self.plots_folder = 'plots'
        self.multiclasses = multiclasses

    def train(self, X_train, y_train) -> None:
        """
        Train the model.

        Parameters:
            - X_train: The training data.
            - y_train: The training labels.

        Returns:
            - None
        """
        self.model.fit(X_train.toarray(), y_train)

    def predict(self, X) -> np.ndarray:
        """
        Predict the labels of the data.

        Parameters:
            - X: The data to predict the labels for.

        Returns:
            - The predicted labels.
        """
        return self.model.predict(X.toarray())

    def evaluate(self, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the model using the following metrics:
        - Accuracy
        - F1-Score
        - AUC-ROC

        Parameters:
            - X_train: The training data.
            - y_train: The training labels.
            - X_test: The test data.
            - y_test: The test labels.

        Returns:
            - A dictionary of the metrics.
        """
        # Predictions
        predict_train = self.predict(X_train)
        predict_test = self.predict(X_test)

        # Metrics
        metrics = {
            "Accuracy (Train)": accuracy_score(y_train, predict_train),
            "Accuracy (Test)": accuracy_score(y_test, predict_test),
            "F1-Score (Train)": f1_score(y_train, predict_train, average="macro"),
            "F1-Score (Test)": f1_score(y_test, predict_test, average="macro"),
        }
        if not self.multiclasses:
            metrics.update({
                "AUC-ROC (Train)": roc_auc_score(y_train, self.model.predict_proba(X_train.toarray())[:, 1]),
                "AUC-ROC (Test)": roc_auc_score(y_test, self.model.predict_proba(X_test.toarray())[:, 1]),
            })
        else:
            metrics.update({
                "AUC-ROC (Train)": roc_auc_score(y_train, self.model.predict_proba(X_train.toarray()),
                                                 multi_class="ovr", average="macro"),
                "AUC-ROC (Test)": roc_auc_score(y_test, self.model.predict_proba(X_test.toarray()),
                                                multi_class="ovr", average="macro"),
            })

        return metrics

    def plot_confusion_matrix(self, X_test, y_test, extra_label: str = "") -> None:
        """
        Plots the confusion matrix for the model on the test set.

        Parameters:
            - X_test: The test data.
            - y_test: The test labels.
            - extra_label (str): An extra label to add to the plot filename.

        Returns:
            - None
        """
        cm = confusion_matrix(y_test, self.predict(X_test), labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.title(f'Confusion Matrix for {self.model_name} on test set')
        # Save the plot to a file inside the 'plots' folder
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)
        plot_filename = os.path.join(self.plots_folder, f'confusion_matrix_{self.model_name}{extra_label}.png')
        plt.savefig(plot_filename)
        plt.close()

    def print_classification_report(self, X_train, y_train, X_test, y_test) -> None:
        """
        Prints the classification report for the model on the train and test sets.

        Parameters:
            - X_train: The training data.
            - y_train: The training labels.
            - X_test: The test data.
            - y_test: The test labels.

        Returns:
            - None
        """
        print(f'Classification Report on train set:\n{classification_report(y_train, self.predict(X_train))}')
        print(f'Classification Report on test set:\n{classification_report(y_test, self.predict(X_test))}')
