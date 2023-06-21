"""
Train a Markov model for poems of Robert Frost. Use each line as a document in the dataset.
Use the trained model to generate poems. We'll do 4 lines at a time.

The idea is to create a second order Markov model, where pi is the initial probability, A1 is the first order
probability, and A2 is the second order probability. We will use only dictionaries and will not include the whole
vocabulary in pi, A1 or A2. This approach of using dictionaries will help us reduce the space these variables take in
memory, so we make them sparse.

Then, we should try to sample from the pi, A1 and A2 dictionaries to create sentences.
"""
import random
import re
from typing import List, Dict, Tuple

ROBERT_FILE = 'data/robert_frost.txt'


def get_docs_tokenized(filename: str) -> List[List[str]]:
    """
    Load documents from a text file and tokenize them.

    Arguments:
        filename (str): Name of the text file to read from.

    Returns:
        List[List[str]]: A list of lists where each sub-list is a tokenized document.
    """

    # Initialize an empty list to store documents
    documents = []
    regex_english = re.compile(r"\b[A-Za-z0-9]+[-'’]*[A-Za-z0-9]*\b", re.IGNORECASE)

    try:
        # Open the file
        with open(filename) as file:
            # Read all the lines from the file
            lines = file.readlines()

            # Iterate over each line
            for line in lines:
                # Remove leading and trailing space and convert to a lower case
                line = line.strip().lower()

                # If the line is not empty
                if line:
                    # Tokenize the line using regex_english and append to the document list
                    documents.append(re.findall(regex_english, line))
    except FileNotFoundError as error:
        # If the file is not found, print the error message
        print(error)

    # Return the tokenized documents
    return documents


def create_pi(documents: List[List[str]]) -> Dict[str, float]:
    """
    Calculate the initial state probabilities for a Markov model.

    The function goes through the first word of each document and counts
    its occurrences. The count is then divided by the total number of documents
    to obtain the initial probability for that word.

    Parameters:
        documents (List[List[str]]): List of documents where each document is a list of tokenized words.

    Returns:
        Dict[str, float]: Dictionary where keys are words and values are the corresponding initial probabilities in
            the Markov model.
    """

    # total number of first words across all documents
    total_words = 0
    # dictionary to store the initial state probabilities
    pi = {}

    # Loop over each document
    for doc in documents:
        # Only consider the first word in each document
        for token in doc[:1]:
            # If the word is already in the dictionary, increment its count,
            # otherwise set its count to 1
            pi[token] = pi.get(token, 0) + 1
            # Increment the total word count
            total_words += 1

    # Calculate the probability for each word by dividing its count by the total word count
    for word, count in pi.items():
        pi[word] = count / total_words

    # return the dictionary of initial state probabilities
    return pi


def create_transition_prob(documents: List[List[str]]) -> \
        Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Create transition probability dictionaries of order one and two for a Markov Model.

    Parameters:
        documents (List[List[str]]): List of tokenized documents.

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
        A tuple of two dictionaries:
        - The first dictionary represents the first-order transition probabilities.
        - The second dictionary represents the second-order transition probabilities.
    """

    # Initialize dictionaries for single and double order transitions
    a1, a2 = {}, {}

    # Append <END> token to each document
    for doc in documents:
        doc.append('<END>')

    # Count the frequency of transitions of order one and two
    for doc in documents:
        # Initialize token trackers
        previous_token, before_previous_token = '', ''

        for num, token in enumerate(doc):
            if num > 0:
                # Count transitions of order one and initialize second order transitions
                a1[previous_token] = a1.get(previous_token, {})
                a2[previous_token] = a2.get(previous_token, {})
                a1[previous_token][token] = a1[previous_token].get(token, 0) + 1
                a2[previous_token][token] = a2[previous_token].get(token, {})

            if num > 1:
                # Count transitions of order two
                current_count = a2[before_previous_token][previous_token].get(token, 0)
                a2[before_previous_token][previous_token][token] = current_count + 1

            # Update token trackers
            before_previous_token, previous_token = previous_token, token

    # Calculate transition probabilities for order one
    for previous_token, rest in a1.items():
        total_count = sum([count for count in rest.values()])
        for token, num in rest.items():
            a1[previous_token][token] = num / total_count

    # Calculate transition probabilities for order two
    for before_previous_token, before_rest in a2.items():
        for previous_token, rest in before_rest.items():
            total_count = sum([count for count in rest.values()])
            for token, num in rest.items():
                a2[before_previous_token][previous_token][token] = num / total_count

    return a1, a2


def make_choice(sequence: List[str], probs: List[float]) -> str:
    """
    Function to randomly select a state based on its probability.

    Parameters:
        sequence (List[str]): A list of states.
        probs (List[float]): A list of probabilities corresponding to the states.

    Returns:
        str: The selected state.
    """

    # generate a random number between 0 and 1
    num = random.uniform(0, 1)

    # initialize the accumulated probability
    accumulated_prob = 0

    # iterate over the states and their corresponding probabilities
    for option, prob in zip(sequence, probs):
        # add the current state's probability to the accumulated probability
        accumulated_prob += prob

        # if the accumulated probability exceeds the random number, return the current state
        if accumulated_prob >= num:
            return option


def create_sentence(pi: Dict[str, float], a1: Dict[str, Dict[str, float]], a2: Dict[str, Dict[str, Dict[str, float]]],
                    sequences: int = 4) -> List[str]:
    """
    Function to generate sentences based on a second order Markov Model.

    Parameters:
        pi (Dict[str, float]): A dictionary representing the initial state probabilities.
        a1 (Dict[str, Dict[str, float]]): A dictionary representing the first order transition probabilities.
        a2 (Dict[str, Dict[str, Dict[str, float]]]): A dictionary representing the second order transition
            probabilities.
        sequences (int): The number of sentences to generate.

    Returns:
        List[str]: A list of generated sentences.
    """

    lines = []
    for _ in range(sequences):
        sentence = []

        # Calculate the initial word based on its probability in pi
        ini_word = make_choice(list(pi.keys()), list(pi.values()))

        # Capitalize the initial word and add it to the sentence
        sentence.append(ini_word.capitalize())

        # Calculate the second word based on its transition probability from the initial word
        second_word_sequence = list(a1[ini_word].keys())
        second_word_probs = list(a1[ini_word].values())
        second_word = make_choice(second_word_sequence, second_word_probs)

        # Ensure the second word is not the end of sentence token
        while second_word == '<END>':
            second_word = make_choice(second_word_sequence, second_word_probs)

        # Add the second word to the sentence
        sentence.append(second_word)

        # Generate remaining words until the end of sentence token is chosen
        not_end = True
        while not_end:
            # Generate the next word based on its second order transition probabilities
            choices = a2[ini_word][second_word]
            next_word = make_choice(list(choices.keys()), list(choices.values()))

            # Stop generating words if the end of sentence token is chosen
            if next_word == '<END>':
                not_end = False
            else:
                # Add the generated word to the sentence and update the current and previous words
                sentence.append(next_word)
                ini_word, second_word = second_word, next_word

        # Join the words of the sentence into a string and add it to the list of sentences
        lines.append(' '.join(sentence))

    return lines


def main() -> None:
    """
    Main function to generate text based on a corpus of documents using a second order Markov Model.

    Steps:
    - Reads documents from a file and tokenizes them.
    - Creates the initial state probabilities (pi).
    - Creates first and second order transition probabilities.
    - Generates new text based on pi and transition probabilities.
    - Prints the generated text.
    """

    # Load and tokenize documents from the specified file
    print('Reading documents from file')
    documents = get_docs_tokenized(ROBERT_FILE)

    # Calculate initial state probabilities from the documents
    print('Creating PI (Initial transition)')
    pi = create_pi(documents)

    # Calculate first and second order transition probabilities from the documents
    print('Creating Transition Probabilities')
    a1, a2 = create_transition_prob(documents)

    # Generate new text based on the initial state probabilities and transition probabilities
    print('Generating text based on PI and Transition Probabilities')
    lines = create_sentence(pi, a1, a2, 10)

    # Print the generated text
    print('Generated text for Robert Frost:')
    print()
    print('\n'.join(lines))


if __name__ == "__main__":
    main()
