"""
The goal of this example is to make a simple Count Vectorizer.
We fetch the text from different documents and take the words from it.
We then create a vocabulary containing all the words, which will allow us to create the header for the matrix, and the
vectors for each document.
Once we have the headers, we can create each vector per document based on the frequency of the words per doc
"""
import os
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from joblib import dump, load

import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

from typing import List, Tuple, Dict

import scipy

TRAINING_FOLDER = 'training'
TESTING_FOLDER = 'testing'
MODEL_FOLDER = 'model'
CLASSIFIER_FILE = os.path.join(MODEL_FOLDER, 'classifier.joblib')
VECTORIZER_FILE = os.path.join(MODEL_FOLDER, 'vectorizer.joblib')
COMMON_CLASSIFIERS: dict = {
    # 'KNeighborsClassifier': {
    #     'param_grid': {
    #         'n_neighbors': np.arange(5, 11),
    #         # 'weights': ['uniform', 'distance'],
    #         # 'p': [1, 2],
    #     },
    #     'classifier': KNeighborsClassifier,
    # },
    # 'MultinomialNB': {
    #     'param_grid': {
    #         'alpha': np.linspace(0.5, 1.5, 6),  # alpha values between 0.5 and 1.5
    #         'fit_prior': [True, False],  # whether to learn class prior probabilities or not
    #     },
    #     'classifier': MultinomialNB,
    # },
    'SVC': {
        'param_grid': {
            'C': [0.1, 1, 10, 100],  # Regularization parameter
            'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        'classifier': SVC,
    },
    'RandomForestClassifier': {
        'param_grid': {
            'n_estimators': [100, 200],  # The number of trees in the forest
            'max_depth': [10, None],  # The maximum depth of the tree
            'min_samples_split': [2, 10],  # The minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 4],  # The minimum number of samples required to be at a leaf node
            'bootstrap': [True, False]
        },
        # 'param_grid': {
        #     'n_estimators': [500],
        #     'max_depth': [20, 30],
        #     'min_samples_split': [5],
        #     'min_samples_leaf': [2],
        #     'bootstrap': [True, False],
        # },
        'classifier': RandomForestClassifier,
    },
}


class BestValues:
    """
    This class is simply to keep track of the best values when training a model.
    """

    def __init__(self, **kwargs):
        self.test_size = kwargs.get('test_size', 0)
        self.vectorizer = kwargs.get('vectorizer', None)
        self.classifier = kwargs.get('classifier', None)
        self.score = kwargs.get('score', 0)
        self.params = kwargs.get('params', None)

    def compare_score(self, new_test_size, new_vectorizer, new_classifier, new_score, new_params):
        if new_score > self.score:
            self.test_size = new_test_size
            self.vectorizer = new_vectorizer
            self.classifier = new_classifier
            self.score = new_score
            self.params = new_params

    def __str__(self):
        return f"Best values are: test_size {self.test_size}, vectorizer: {self.vectorizer} and " \
               f"classifier: {self.classifier} with {self.score} & {self.params}"


class BestClassifier:
    """
    Another test double class to keep track of the best classifiers for the training
    """

    def __init__(self):
        self.classifier = None
        self.score = 0
        self.params = None

    def compare_score(self, new_score, new_params, new_classifier):
        if new_score > self.score:
            self.classifier = new_classifier
            self.score = new_score
            self.params = new_params

    def __str__(self):
        return f"Best classifier: {self.classifier} with {self.score} & {self.params}"


def get_wordnet_pos(treebank_tag: str):
    """
    Based on treebank_tag, it returns the value from wordnet regarding the type of word
    :param treebank_tag: the string to check
    :return: the equivalent from wordnet
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemma_tokenizer_tokenize(text: str) -> List[str]:
    """
    Tokenizes a text using the lemmatizer
    :param text: The text to tokenize
    :return: A list of tokens lemmatized
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(tokens)]


def lemma_tokenizer_regexp(text: str) -> List[str]:
    """
    Takes care of cleaning up the text by using a tokenizer from NLTK (RegexpTokenizer) and then lemmatizing the tokens
    :param text: The original text from the document
    :return: The transformed, lemmatized text in a list
    """
    tokenizer = RegexpTokenizer(r'\b\w*\S\w*\b')
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(tokens)]


def get_documents() -> Tuple[List[str], List[str]]:
    """
    Gets the documents from the training folder to train and test the model
    :return: A tuple containing two lists, the list of documents and the list of labels
    """
    documents, labels = [], []
    files = os.listdir(TRAINING_FOLDER)
    for filename in files:
        if filename == '.DS_Store':
            continue
        document = os.path.join(TRAINING_FOLDER, filename)
        with open(document) as file:
            lines = file.readlines()
            lines = ' '.join(line.strip() for line in lines).lower()
        documents.append(lines)
        labels.append(filename.split('_')[0])
    return documents, labels


def get_stopwords() -> List[str]:
    """
    Gets the stop words for English from a GitHub project
    :return:
    """
    stopwords_list = requests.get(
        "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/"
        "12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content

    stop_words = stopwords_list.decode().splitlines()
    stop_words.remove('research-articl')
    stop_words.remove("c'mon")
    return stop_words


def perform_grid_search(X_train, X_test, y_train, y_test, classifier_name=''):
    """
    Performs a grid search on X_train, and makes tests on X_test
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param classifier_name: The classifier to use
    :return:
    """
    param_grid, classifier = COMMON_CLASSIFIERS[classifier_name].values()
    grid_search = GridSearchCV(classifier(), param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    # print(f"GridSearchCV score with {classifier_name}: {test_score} - {grid_search.best_params_}")
    return test_score, grid_search.best_params_


def perform_random_search(X_train, X_test, y_train, y_test, classifier_name=''):
    param_grid, classifier = COMMON_CLASSIFIERS[classifier_name].values()
    random_search = RandomizedSearchCV(classifier(), param_distributions=param_grid, n_iter=6, cv=5,
                                       random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    # print(f"RandomizedSearchCV score with {classifier_name}: {test_score} - {random_search.best_params_}")
    return test_score, random_search.best_params_


def train_scipy(best_values: Dict = None):
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

    documents, labels = get_documents()
    english_stop_words = get_stopwords()
    best = BestValues(**best_values)

    # If we already have the best values, no need to look for it
    if best.score == 0:
        analyzers = ('char', 'word')
        stop_words = {'custom': english_stop_words, 'english': 'english', 'NLTK': stopwords.words('english')}
        tokenizers = (lemma_tokenizer_tokenize, lemma_tokenizer_regexp)

        print('Finding the best combinations...')
        for test_size in np.arange(0.15, 0.26, 0.01):
            X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=test_size, random_state=42)
            for analyzer in analyzers:
                if analyzer == 'char':
                    # A: First combination, only using char
                    vectorizer = CountVectorizer(strip_accents='unicode', analyzer=analyzer)
                    best_classifier_a = train_model(vectorizer, X_train, X_test, y_train, y_test)
                    print('BEST A', best_classifier_a)
                    best.compare_score(test_size, vectorizer, best_classifier_a.classifier, best_classifier_a.score,
                                       best_classifier_a.params)
                else:
                    for tokenizer in tokenizers:
                        for stop_word_text, stop_word in stop_words.items():
                            # B: Second combination, using word, stop words and a custom tokenizer
                            vectorizer = CountVectorizer(strip_accents='unicode', analyzer=analyzer,
                                                         stop_words=stop_word,
                                                         tokenizer=tokenizer, token_pattern=None)
                            best_classifier_b = train_model(vectorizer, X_train, X_test, y_train, y_test)
                            print('BEST B', best_classifier_b)
                            best.compare_score(test_size, vectorizer, best_classifier_b.classifier,
                                               best_classifier_b.score,
                                               best_classifier_b.params)

                            # C: Third combination, using word, stop words and no custom tokenizer
                            vectorizer = CountVectorizer(strip_accents='unicode', stop_words=stop_word)
                            best_classifier_c = train_model(vectorizer, X_train, X_test, y_train, y_test)
                            print('BEST C', best_classifier_c)
                            best.compare_score(test_size, vectorizer, best_classifier_c.classifier,
                                               best_classifier_c.score,
                                               best_classifier_c.params)

            # D: Fourth combination, nothing, just the regular CountVectorizer
            vectorizer = CountVectorizer()
            best_classifier_d = train_model(vectorizer, X_train, X_test, y_train, y_test)
            print('BEST D', best_classifier_d)
            best.compare_score(test_size, vectorizer, best_classifier_d.classifier, best_classifier_d.score,
                               best_classifier_d.params)
            print(f'BEST at test_size {test_size}', best)
        print('Found the best combination!')
    else:
        print('Best combination provided by hand')

    print(f"- Test size: {best.test_size}")
    print(f"- Classifier: {best.classifier}")
    print(f"- Params: {best.params}")
    print(f"- Score: {best.score}")
    print(f"- Vectorizer: {best.vectorizer}")

    print('Training the model with the best combination...')
    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=best.test_size, random_state=42)
    vectorizer = best.vectorizer
    X_train_matrix = vectorizer.fit_transform(X_train)
    classifier = COMMON_CLASSIFIERS[best.classifier]['classifier'](**best.params)
    classifier.fit(X_train_matrix, y_train)

    print('Training done, saving model...')
    # save the trained classifier and vectorizer to a file
    dump(classifier, CLASSIFIER_FILE)
    dump(vectorizer, VECTORIZER_FILE)
    print('Model saved!')


def train_model(vectorizer, X_train, X_test, y_train, y_test):
    best = BestClassifier()
    X_train_matrix = vectorizer.fit_transform(X_train)
    X_test_matrix = vectorizer.transform(X_test)

    for classifier_name in COMMON_CLASSIFIERS.keys():
        score, params = perform_grid_search(X_train_matrix, X_test_matrix, y_train, y_test, classifier_name)
        best.compare_score(score, params, classifier_name)
        score, params = perform_random_search(X_train_matrix, X_test_matrix, y_train, y_test, classifier_name)
        best.compare_score(score, params, classifier_name)
    return best


def test_scipy(folder_path):
    print()
    print('Testing the model...')
    loaded_classifier = load(CLASSIFIER_FILE)
    loaded_vectorizer = load(VECTORIZER_FILE)

    correct = 0
    items = os.listdir(folder_path)
    total = len(items)
    items = sorted(items)
    for item in items:
        if item == '.DS_Store':
            continue
        with open(os.path.join(TESTING_FOLDER, item)) as file:
            lines = file.readlines()
            lines = ' '.join(line.strip() for line in lines).lower()
        lines = ' '.join(lemma_tokenizer_regexp(lines))
        new_document = [lines]
        new_X_test = loaded_vectorizer.transform(new_document)
        predicted_label = loaded_classifier.predict(new_X_test)
        real_artist = item.split('_')[0]
        if real_artist == predicted_label[0]:
            print(f'Correct! {item} is {predicted_label[0]}')
            correct += 1
        else:
            print(f'Wrong! {item} is not {predicted_label[0]}')

    print(f'{correct}/{total} - {correct / total:.2%} correct answers')


def main():
    if os.path.exists(CLASSIFIER_FILE) and os.path.exists(VECTORIZER_FILE):
        print('Model already trained, jumping straight to testing...')
    else:
        print('Model not yet trained, beginning of training now...')
        best_models = {
            'KNeighborsClassifier': {
                'test_size': 0.25,
                'vectorizer': CountVectorizer(analyzer='char', strip_accents='unicode'),
                'classifier': 'KNeighborsClassifier',
                'params': {'n_neighbors': 10},
                'score': 0.7719298245614035,
            },
            'MultinomialNB': {
                'test_size': 0.24,
                'vectorizer': CountVectorizer(stop_words='english', strip_accents='unicode', token_pattern=None,
                                              tokenizer=lemma_tokenizer_regexp),
                'classifier': 'MultinomialNB',
                'params': {'alpha': 0.5, 'fit_prior': True},
                'score': 0.8363636363636363,
            },
            'SVC': {
                'test_size': 0.25,
                'vectorizer': CountVectorizer(),
                'classifier': 'SVC',
                'params': {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
                'score': 0.9428571428571428,
            },
            # This one gives 35/43 - 81.40% correct answers
            'RandomForestClassifier': {
                'test_size': 0.16,
                'vectorizer': CountVectorizer(stop_words='english', strip_accents='unicode', token_pattern=None,
                                              tokenizer=lemma_tokenizer_tokenize),
                'classifier': 'RandomForestClassifier',
                'params': {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None,
                           'bootstrap': False},
                'score': 0.972972972972973,
            }}
        best_values = best_models['RandomForestClassifier']
        train_scipy(best_values=best_values)
    test_scipy(TESTING_FOLDER)


if __name__ == '__main__':
    main()
