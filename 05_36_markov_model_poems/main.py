import re
import numpy as np
from typing import List, Dict, Tuple

EDGAR_FILE = 'data/edgar_allan_poe.txt'
ROBERT_FILE = 'data/robert_frost.txt'


class MarkovModel:
    def __init__(self, documents: List[str], ratio_split: float = 0.8):
        """
        Initializes the Markov model with a set of documents and a training/testing split ratio.

        Args:
            documents: List of document strings.
            ratio_split: Float representing the ratio of documents to be used for training.
                         The rest will be used for testing.
        """
        # Define a regex pattern to tokenize English words
        self.__regex_english = re.compile(r"\b[A-Za-z0-9]+[-'’]*[A-Za-z0-9]*\b", re.IGNORECASE)

        self.documents = documents

        # Define an empty vocabulary dictionary for the training set
        self.vocabulary_train = {}

        # Tokenize the documents and split them into training and testing sets
        self.tokenized_docs_train, self.tokenized_docs_test = self.__tokenize_text_split_train_test(ratio_split)

        # Generate the transition probability matrix
        self.transition_probs = self.__create_transition_matrix()

    def __tokenize_text_split_train_test(self, ratio_split):
        """
        Tokenizes the documents and splits them into training and testing sets.

        Args:
            ratio_split: Float representing the ratio of documents to be used for training.
                         The rest will be used for testing.

        Returns:
            Tuple containing the training and testing sets as lists of tokenized documents.
        """
        # Tokenize the documents
        documents = [re.findall(self.__regex_english, doc.lower()) for doc in self.documents]

        # Set a seed for reproducibility
        np.random.seed(42)

        # Shuffle the documents to ensure random distribution
        np.random.shuffle(documents)

        # Calculate the index up to which to split the documents
        up_to = round(len(documents) * ratio_split)

        # Split the documents into training and testing sets
        return documents[:up_to], documents[up_to:]

    def __create_transition_matrix(self):
        """
        Method to create a transition matrix using Laplace smoothing.

        The transition matrix is a two-dimensional dictionary where each key is a word from the training set,
        and its value is another dictionary that represents all the words that can follow the key and their
        probability.

        The method also applies Laplace smoothing to handle transitions that do not occur in the training set
        but might occur in the testing set.
        """

        # Initialize an empty dictionary to store the transition matrix
        matrix = {}

        # Go through each sentence in the training set
        for sentence in self.tokenized_docs_train:

            # Go through each word in the sentence
            for position in range(len(sentence)):
                word = sentence[position]

                # If the word is not already a key in the matrix, add it
                if not matrix.get(word):
                    matrix[word] = {}

                # If there's a next word, add it to the transitions of the current word
                if position + 1 < len(sentence):
                    next_word = sentence[position + 1]
                    matrix[word][next_word] = matrix[word].get(next_word, 0) + 1

        # Create a vocabulary of all unique words in the training set
        for sentence in self.tokenized_docs_train:
            for word in sentence:
                self.vocabulary_train[word] = 0

        # Adjusted Laplace smoothing part
        for word in matrix.keys():
            for next_word in self.vocabulary_train:
                if next_word not in matrix[word]:
                    matrix[word][next_word] = 1
                else:
                    matrix[word][next_word] += 1

        # Normalize the counts to get probabilities
        for word, next_words in matrix.items():
            sum_appearences = sum(next_words.values())
            for next_word, count in next_words.items():
                matrix[word][next_word] = count / sum_appearences

        return matrix

    def check_balance(self):
        """
        Calculates various statistics about the documents in the training set.

        Returns:
            Tuple containing the number of documents, total word count, and average document length in the training set.
        """
        # Calculate the number of documents in the training set
        docs_amount = len(self.tokenized_docs_train)

        # Calculate the length of each document in the training set
        docs_lengths = [len(doc) for doc in self.tokenized_docs_train]

        # Calculate the total number of words in the training set
        total_words = sum(docs_lengths)

        # Calculate the average length of the documents in the training set
        docs_average_length = np.average(docs_lengths)

        return docs_amount, total_words, docs_average_length

    def probability_of_sequence(self, sequence: List[str]) -> float:
        """
        Calculate the probability of a sequence of states.

        Args:
            sequence (list): The sequence of states for which to calculate the probability.

        Returns:
            float: The probability of the sequence of states.
        """

        # Pre-calculate the base probability, which is used for states not seen in the training set.
        base_probability = 1 / len(self.vocabulary_train)

        # Initialize the list for log probabilities.
        log_probs = []

        # Iterate over the states in the sequence.
        for count, state in enumerate(sequence):

            # By default, set the probability to the base probability.
            probability = base_probability

            # If it's not the first state in the sequence, look up the transition probability.
            if count > 0:
                previous_state = sequence[count - 1]

                # If the previous state exists in the transition matrix, look up the transition probability.
                # If the state does not follow the previous state in the training data,
                # use the base probability instead.
                if previous_state in self.transition_probs.keys():
                    probability = self.transition_probs[previous_state].get(state, base_probability)

            # Add the log of the probability to the list.
            log_probs.append(np.log(probability))

        # Return the total probability of the sequence, which is the exponent of the sum of the log probabilities.
        return np.exp(np.sum(log_probs))

    def generate_text(self) -> str:
        """
        Method to generate a text sequence based on the Markov Model.

        The method randomly selects an initial word from a random document in the training set,
        then generates subsequent words based on the transition probabilities.

        The generated sequence is of an average length equivalent to the documents in the training set.

        Returns:
            str: The generated text sequence.
        """
        # Set the seed to current time for randomness
        np.random.seed()

        # Create an empty list to hold the generated sequence
        sequence = []

        # Calculate the average document length in the training set
        avg_length = sum([len(sequence) for sequence in self.tokenized_docs_train]) / len(self.tokenized_docs_train)

        # Select a random document from the training set
        random_sequence_index = np.random.randint(0, high=len(self.tokenized_docs_train))
        random_sequence = self.tokenized_docs_train[random_sequence_index]

        # Select the first word from the random document
        next_word = random_sequence[0]

        # Capitalize the first word and add it to the sequence
        sequence.append(next_word.title())

        # While the generated sequence is shorter than the average document length, keep adding words
        while len(sequence) < avg_length:
            # Select the next word based on the transition probabilities
            next_word = np.random.choice(
                list(self.transition_probs[next_word].keys()),
                p=list(self.transition_probs[next_word].values())
            )

            # Add the selected word to the sequence
            sequence.append(next_word)

        # Join the sequence into a single string and add a period at the end
        return ' '.join(sequence) + '.'


def file_to_text(filename: str) -> List[str]:
    """
    Reads a text file and splits it into sentences.

    Args:
        filename (str): The path of the text file.

    Returns:
        List of sentences extracted from the text file. Each sentence is converted to lowercase.

    Raises:
        FileNotFoundError: If the specified file is not found.
    """
    documents = []
    try:
        # Open and read the file
        with open(filename) as file:
            lines = file.readlines()

            # Join the lines and strip unnecessary spaces
            lines = ' '.join(line.strip() for line in lines)

            # Split the text into sentences
            documents = re.findall(r"\s*[^.!?]*[.!?]", lines)

            # Convert all sentences to lowercase
            documents = [doc.strip().lower() for doc in documents]
    except FileNotFoundError as e:
        print(e)
    finally:
        return documents


def check_balance(models: List[MarkovModel], thresholds: Dict[str, float]):
    """
    Checks if the Markov models are balanced.

    Calculates various metrics (document counts, total word count, average document length) for both models
    and checks if the ratio of these metrics exceeds the provided thresholds.

    Args:
        models: A list of two Markov models to compare.
        thresholds: A dictionary of thresholds for the metrics. Keys can be 'docs_amount', 'total_words',
        'docs_average_length'.

    Returns:
        A dictionary of metrics that exceed the thresholds.
    """
    # Set default values for thresholds if not provided
    thresholds['docs_amount'] = thresholds.get('docs_amount', 1.2)
    thresholds['total_words'] = thresholds.get('total_words', 1.2)
    thresholds['docs_average_length'] = thresholds.get('docs_average_length', 1.1)

    # Calculate metrics for both models
    doc_amount_0, total_words_0, average_doc_length_0 = models[0].check_balance()
    doc_amount_1, total_words_1, average_doc_length_1 = models[1].check_balance()
    problems = {}

    # Check if the ratio of document counts exceeds the threshold
    docs_ratio = max(doc_amount_0, doc_amount_1) / min(doc_amount_0, doc_amount_1)
    if docs_ratio > thresholds['docs_amount']:
        problems['docs_amount'] = docs_ratio

    # Check if the ratio of total word counts exceeds the threshold
    words_ratio = max(total_words_0, total_words_1) / min(total_words_0, total_words_1)
    if words_ratio > thresholds['total_words']:
        problems['total_words'] = words_ratio

    # Check if the ratio of average document lengths exceeds the threshold
    length_ratio = max(average_doc_length_0, average_doc_length_1) / min(average_doc_length_0, average_doc_length_1)
    if length_ratio > thresholds['docs_average_length']:
        problems['docs_average_length'] = length_ratio
    return problems


def calculate_f1_score(models: List[MarkovModel]) -> Tuple[float, float]:
    """
    Calculate the F1 scores for two models using their respective test sets.

    Args:
    models (list): A list of two MarkovModel instances.

    Returns:
    tuple: A tuple of two floats representing the F1 scores for the first and second model, respectively.
    """

    # Extract the models from the input list
    model_a = models[0]
    model_b = models[1]

    # Initialize counters for true positives, true negatives, false positives, and false negatives
    scores = {
        'model_a': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        'model_b': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    }

    # Calculate scores for model A's test set
    for sentence in model_a.tokenized_docs_test:
        prob_a = model_a.probability_of_sequence(sentence)
        prob_b = model_b.probability_of_sequence(sentence)

        if prob_a > prob_b:
            # The Sentence is correctly classified as belonging to A
            scores['model_a']['tp'] += 1
            scores['model_b']['tn'] += 1
        elif prob_b > prob_a:
            # The Sentence is incorrectly classified as belonging to B
            scores['model_a']['fn'] += 1
            scores['model_b']['fp'] += 1

    # Calculate scores for model B's test set
    for sentence in model_b.tokenized_docs_test:
        prob_a = model_a.probability_of_sequence(sentence)
        prob_b = model_b.probability_of_sequence(sentence)

        if prob_b > prob_a:
            # The Sentence is correctly classified as belonging to B
            scores['model_b']['tp'] += 1
            scores['model_a']['tn'] += 1
        elif prob_a > prob_b:
            # The Sentence is incorrectly classified as belonging to A
            scores['model_b']['fn'] += 1
            scores['model_a']['fp'] += 1

    # Calculate F1 scores
    model_a_precision = scores['model_a']['tp'] / (scores['model_a']['tp'] + scores['model_a']['fp'])
    model_a_recall = scores['model_a']['tp'] / (scores['model_a']['tp'] + scores['model_a']['fn'])
    model_a_f1_score = 2 * (model_a_precision * model_a_recall) / (model_a_precision + model_a_recall)

    model_b_precision = scores['model_b']['tp'] / (scores['model_b']['tp'] + scores['model_b']['fp'])
    model_b_recall = scores['model_b']['tp'] / (scores['model_b']['tp'] + scores['model_b']['fn'])
    model_b_f1_score = 2 * (model_b_precision * model_b_recall) / (model_b_precision + model_b_recall)

    return model_a_f1_score, model_b_f1_score


def main():
    """
    Main function to train the Markov models and generate sample sentences.
    It loads documents, trains Markov models for each author, checks for class imbalance,
    calculates the F1 scores for each model, and generates sample sentences.
    """
    # Load documents from each author's files
    documents = file_to_text(ROBERT_FILE)
    model_robert = MarkovModel(documents=documents)
    documents = file_to_text(EDGAR_FILE)
    model_edgar = MarkovModel(documents=documents)

    # List of models
    models = [model_edgar, model_robert]

    # Define thresholds for class balance checks
    thresholds = {'docs_amount': 1.2, 'total_words': 1.2, 'docs_average_length': 1.2}

    # Check if classes are balanced
    classes_balanced = check_balance(models, thresholds=thresholds)

    # If there are imbalances, calculate F1 scores for both models
    if classes_balanced is not None:
        print('Problem. Class imbalance')
        model_edgar_f1_score, model_robert_f1_score = calculate_f1_score(models)
        print(f'F1 score for Edgar: {model_edgar_f1_score:.2%}')
        print(f'F1 score for Robert: {model_robert_f1_score:.2%}')

    # Generate sample sentences for each author
    print('Sample generated sentence for Edgar:')
    print(model_edgar.generate_text())
    print('Sample generated sentence for Robert:')
    print(model_robert.generate_text())


if __name__ == '__main__':
    main()
