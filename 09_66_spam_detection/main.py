"""
Create a spam detection ML tool

- Use a vectorization strategy like counting or TF-IDF
- Additional options: Tokenization, lemmatization, normalization, etc.
- Classifier: An appropriate form of Naive Bayes. Use scikit-learn or build it yourself
- Feel free to try other classifiers
- Score function returns accuracy
- Check for class imbalance
- If there is class imbalance, check metrics like AUC / F1-score
"""
import os
from typing import List
from joblib import dump, load
import pandas as pd

import nltk
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, \
    ConfusionMatrixDisplay

DATA_CSV_FILENAME = 'data/spam.csv'
EXAMPLES_CSV_FILENAME = 'data/spam_test.csv'


class DocumentPandas:
    """
    Loads documents from a CSV file using pandas
    """

    def __init__(self, filename: str, encoding: str = 'ISO-8859-1'):
        self.encoding = encoding
        self.documents = []
        self.labels = []

        self.prepare_documents(filename)

    def prepare_documents(self, filename: str) -> None:
        try:
            df = pd.read_csv(filename, encoding=self.encoding)

            # Assuming the first column contains labels and the second column contains documents
            self.labels = df.iloc[:, 0].tolist()
            self.documents = df.iloc[:, 1].str.lower().tolist()

        except FileNotFoundError as error:
            print(error)


class Preprocessor:
    """
    Preprocesses documents by tokenizing, lemmatizing, and normalizing
    """

    def __init__(self, file_path):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.file_path = file_path
        self.documents = []
        self.labels = []
        self.vectorizer = None
        self.vectorizer_fit_transform = None

        self.save_folder = 'model'
        self.documents_file = os.path.join(self.save_folder, 'documents.joblib')
        self.labels_file = os.path.join(self.save_folder, 'labels.joblib')
        self.vectorizer_file = os.path.join(self.save_folder, 'vectorizer.joblib')
        self.vectorizer_fit_transform_file = os.path.join(self.save_folder, 'vectorizer_fit_tranform.joblib')

        loaded = self.load_files()
        if not loaded:
            self.raw_data = self.load_data()
            self.documents = self.raw_data['v2'].astype(str)  # Assuming 'v2' is the column name for the text data
            self.labels = self.raw_data['v1']  # Assuming 'v1' is the column name for the labels
            self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, strip_accents='unicode',
                                              tokenizer=self.tokenize_document, token_pattern=None)
            self.vectorizer_fit_transform = self.vectorizer.fit_transform(self.documents)
            self.save_files()

    def load_data(self):
        """
        Load the dataset from a CSV file
        """
        try:
            return pd.read_csv(self.file_path, encoding='ISO-8859-1')
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return pd.DataFrame()

    def load_files(self):
        """
        Load the documents, labels, and vectorizer from a file
        """
        try:
            self.documents = load(self.documents_file)
            self.labels = load(self.labels_file)
            self.vectorizer = load(self.vectorizer_file)
            self.vectorizer_fit_transform = load(self.vectorizer_fit_transform_file)
        except FileNotFoundError:
            print('Dumps not found. Attempting to create the dumps...')
            return False
        else:
            print('Dumps found, loading from file...')
            return True

    def save_files(self):
        """
        Save the documents, labels, and vectorizer to a file
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        dump(self.documents, self.documents_file)
        dump(self.labels, self.labels_file)
        dump(self.vectorizer, self.vectorizer_file)
        dump(self.vectorizer_fit_transform, self.vectorizer_fit_transform_file)

    def get_wordnet_pos(self, tag: str) -> str:
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

    def tokenize_document(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize the text
        """
        wnl = nltk.stem.WordNetLemmatizer()
        tokenize = nltk.tokenize.treebank.TreebankWordTokenizer()
        tokens = tokenize.tokenize(text)
        return [wnl.lemmatize(word, pos=self.get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(tokens)]


class SpamClassifier:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.plots_folder = 'plots'

    def train(self, X_train, y_train):
        self.model.fit(X_train.toarray(), y_train)

    def predict(self, X):
        return self.model.predict(X.toarray())

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the model using the following metrics:
        - Accuracy
        - F1-Score
        - AUC-ROC
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
            "AUC-ROC (Train)": roc_auc_score(y_train, self.model.predict_proba(X_train.toarray())[:, 1]),
            "AUC-ROC (Test)": roc_auc_score(y_test, self.model.predict_proba(X_test.toarray())[:, 1])
        }

        return metrics

    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plots the confusion matrix for the model on the test set.
        """
        cm = confusion_matrix(y_test, self.predict(X_test), labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.title(f'Confusion Matrix for {self.model_name} on test set')
        # Save the plot to a file inside the 'plots' folder
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)
        plot_filename = os.path.join(self.plots_folder, f'confusion_matrix_{self.model_name}.png')
        plt.savefig(plot_filename)
        plt.close()

    def print_classification_report(self, X_train, y_train, X_test, y_test):
        """
        Prints the classification report for the model on the train and test sets.
        """
        print(f'Classification Report on train set:\n{classification_report(y_train, self.predict(X_train))}')
        print(f'Classification Report on test set:\n{classification_report(y_test, self.predict(X_test))}')


def check_class_imbalance(labels):
    """
    Checks for class imbalance in the labels.
    """
    class_counts = labels.value_counts()
    total_samples = len(labels)
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples ({(count / total_samples) * 100:.2f}%)")
    if class_counts.max() / class_counts.min() > 2:  # Arbitrary threshold
        print("Warning: Significant class imbalance detected!")
    else:
        print("Class distribution seems balanced.")

    return class_counts


def main():
    preprocess = Preprocessor(DATA_CSV_FILENAME)

    # Check for class imbalance
    check_class_imbalance(preprocess.labels)

    X_train, X_test, y_train, y_test = train_test_split(preprocess.vectorizer_fit_transform, preprocess.labels,
                                                        test_size=0.2, random_state=42)
    print('Evaluating the models...')
    model_dict = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }
    docs_exp = DocumentPandas(EXAMPLES_CSV_FILENAME)

    for model_name, model_instance in model_dict.items():
        print(f'Evaluating {model_name}...')
        classifier = SpamClassifier(model_instance, model_name)
        classifier.train(X_train, y_train)

        metrics = classifier.evaluate(X_train, y_train, X_test, y_test)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}")

        classifier.plot_confusion_matrix(X_test, y_test)
        classifier.print_classification_report(X_train, y_train, X_test, y_test)

        print('Predicting new examples on the model...')
        X_new = preprocess.vectorizer.transform(docs_exp.documents)
        predict_exp = classifier.predict(X_new)
        accuracy = accuracy_score(docs_exp.labels, predict_exp)
        print(f'Accuracy Score on the examples: {accuracy:.2f}')
        print('\n' + '-' * 50 + '\n')


if __name__ == '__main__':
    main()
