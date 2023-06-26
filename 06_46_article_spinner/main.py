"""
The idea is to craete an article spinner, there are several considerations to make it happen:
- Download a dataset like Wikipedia.
- Make a VxVxV matrix and figure out if using a dictionary (sparse) is a better idea.
- What should we consider tokens?
- Use word_tokenize from NLTK, and use punctuation to finish sentences; we should also keep symbols like $.
- How to spin the article? Which words are to replace? How often? How to know if a word can be replaced?
- How to put a list of tokens back together? We can use TreebankWordDetokenizer from NLTK.
- When spinning the article, show also the original article.
"""
import os
import nltk
import pickle
import random
import numpy as np
from typing import List, Dict, Tuple, Any

TAGS = ['N', 'V', 'J', 'R']
DOCUMENTS_FILENAME = "documents.pkl"
MATRIX_FILENAME = "matrix.pkl"


def save_object(obj: Any, filename: str) -> None:
    """
    Saves the given object to a file using pickle.

    :param obj: The object to save.
    :param filename: The name of the file to save the object to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filename: str) -> Any | bool:
    """
    Loads an object from a file using pickle.

    :param filename: The name of the file to load the object from.
    :return: The loaded object.
    """
    obj = False
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    except FileNotFoundError:
        print(f"Filename {filename} not found. Creating a new object.")
    else:
        print('Object successfully loaded from file')
    finally:
        return obj


def prepare_documents(folder: str) -> List[List[str]]:
    """
    Prepares the documents for the article spinner by reading them from a specified folder.

    If the documents' object is already on file, it will load it and skip the creation process. If the documents'
    object is not on file, it will create it and save it for the next time.

    This function assumes that all text files in the specified folder are documents
    to be used by the article spinner. Each document is read line by line, and each line
    is appended to a list representing the document. All documents are then returned as a list.

    :param folder: The path of the folder containing the documents.
    :return: A list of documents, where each document is a list of lines (strings).
    """
    print('Trying to load the documents from file')
    documents = load_object(DOCUMENTS_FILENAME)
    if not documents:
        print('Creating the documents object')
        documents = []
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                document = []
                with open(os.path.join(folder, filename)) as file:
                    text = file.readlines()
                    for line in text:
                        document.append(line)
                documents.append(document)
        print('Saving the documents to file')
        save_object(documents, DOCUMENTS_FILENAME)
    return documents


def tokenize_document(document: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Tokenizes a document using the Treebank tokenizer and tags the tokens using the POS tagger.

    This function uses the nltk library's Treebank tokenizer to tokenize each line in a document.
    After tokenization, each token is tagged using nltk's POS tagger.

    :param document: A document to tokenize, represented as a list of lines (strings).
    :return: A list of lists of tuples, where each list represents a line, and each tuple is a (token, tag) pair.
    """
    tokenize = nltk.tokenize.treebank.TreebankWordTokenizer()
    tokenized_doc = []
    for line in document:
        tokens = tokenize.tokenize(line)
        tagged_tokens = nltk.pos_tag(tokens)
        tokenized_doc.append(tagged_tokens)
    return tokenized_doc


def create_transition_matrix(documents: List[List[str]]) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
    """
    Creates a transition matrix from a list of documents

    If the transition matrix is already on file, it will load it and skip the creation process. If the matrix is not
    on file, it will create it and save it for the next time.

    The transition matrix maps combinations of preceding and following word to dictionaries which
    map POS tags to dictionaries of words and their probabilities.

    :param documents: A list of documents, where each document is a list of strings (sentences).
    :return: A transition matrix in the form of a dictionary mapping tuples of
             (preceding_word, following_word) to dictionaries. These inner dictionaries
             map POS tags to dictionaries of words and their probabilities.
    """
    print('Trying to load the matrix from file')
    matrix = load_object(MATRIX_FILENAME)
    if not matrix:
        print('Creating the matrix')
        # Define the tags of interest
        tmp_probs = {}

        # Iterate over documents and sentences
        for doc in documents:
            new_doc = tokenize_document(doc)
            for tokens in new_doc:
                # Iterate over tokens in the sentence
                for pos, (token, tag) in enumerate(tokens):
                    if any([t for t in TAGS if tag.startswith(t)]):
                        combination = create_combination(pos, tokens)

                        # Increment the count of the current word in the relevant dictionaries
                        tmp_probs[combination] = tmp_probs.get(combination, {})
                        tmp_probs[combination][tag[0]] = tmp_probs[combination].get(tag[0], {})
                        tmp_probs[combination][tag[0]][token] = tmp_probs[combination][tag[0]].get(token, 0) + 1

        # Convert counts to probabilities and remove tokens that only appear once
        matrix = {}
        for combination, tag_dict in tmp_probs.items():
            for tag, tokens in tag_dict.items():
                if len(tokens) > 1:
                    total = sum([count for count in tokens.values()])
                    for token, count in tokens.items():
                        matrix[combination] = matrix.get(combination, {})
                        matrix[combination][tag] = matrix[combination].get(tag, {})
                        matrix[combination][tag][token] = count / total
        print('Saving the matrix to file')
        save_object(matrix, MATRIX_FILENAME)

    return matrix


def spin_article(article: List[str], matrix: Dict[Tuple[str, str], Dict[str, Dict[str, float]]]) -> \
        Tuple[str, Dict[str, Dict[str, List[str]]]]:
    """
    Spins an article using a given transition matrix.

    This function tokenizes and tags the article, and then iterates over the tokens.
    If a token's tag is in the predefined list of tags, it tries to find a replacement for it
    in the transition matrix, based on the preceding and following words.

    The function keeps track of the changes that have been made during the spinning process.

    :param article: The article to spin, represented as a list of lines (strings).
    :param matrix: The transition matrix to use for spinning the article.
    :return: A tuple containing the spun article, and the changes made during the spinning process.
    """
    # Initialize the list for the spun article, and the dictionary for the changes
    spinned_article: list = []
    changes = {}

    # Tokenize and tag the article
    article_tokenized = tokenize_document(article)

    # Define the frequency limit for each tag
    frequency_limit = {'N': 0.6, 'V': 0.5, 'J': 0.5, 'R': 0.4}

    # Iterate over the tokens in the article
    for tokens in article_tokenized:
        # Initialize the list for the spun lines
        spinned_line = []

        # Iterate over the tokens in the line
        for pos, (token, tag) in enumerate(tokens):
            # If the token's tag is in the list of tags, try to find a replacement for it
            if any([t for t in TAGS if tag.startswith(t)]):
                # Create the combination of preceding and following words
                combination = create_combination(pos, tokens)

                # If the combination exists in the transition matrix, try to find a replacement for the token
                if matrix.get(combination, False):
                    replacements = matrix[combination].get(tag[0], False)
                    if replacements:
                        if random.random() < frequency_limit[tag[0]]:
                            # Choose a replacement for the token based on the probabilities in the transition matrix
                            new_token = np.random.choice(list(replacements.keys()), p=list(replacements.values()))

                            # Add the change to the `changes` dictionary
                            changes[tag[0]] = changes.get(tag[0], {})
                            changes[tag[0]][token] = changes[tag[0]].get(token, [])
                            changes[tag[0]][token].append(new_token)

                            # Replace the token with the new token
                            token = new_token
                # Add the token (either original or replaced) to the spun line
                spinned_line.append(token)
            else:
                # If the token's tag is not in the list of tags, add it to the spun line as is
                spinned_line.append(token)

        # Add the spun lines to the article
        spinned_article.append(spinned_line)

    # Detokenize spun article
    detokenize = nltk.tokenize.treebank.TreebankWordDetokenizer()
    spinned_article: str = '\n'.join([detokenize.detokenize(tokens) for tokens in spinned_article])

    return spinned_article, changes


def create_combination(pos: int, tokens: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Creates a combination of preceding and following words for a given position in a list of tokens.

    This function handles three cases:
    - When the current word is the first in a sentence, the preceding word is set to '<START>'.
    - When the current word is not the first nor the last in a sentence, the preceding and following words
      are set to the actual words before and after the current word in the sentence.
    - When the current word is the last in a sentence, the following word is set to '<END>'.

    :param pos: The position of the current word in the sentence.
    :param tokens: The list of tokens in the sentence.
    :return: A tuple of the preceding and following words.
    """
    if pos == 0:
        # Handle the case when the current word is the first in a sentence
        next_token = tokens[pos + 1][0]
        combination = ('<START>', next_token)
    elif pos < len(tokens) - 1:
        # Handle the case when the current word is not the first nor the last in a sentence
        previous_token = tokens[pos - 1][0]
        next_token = tokens[pos + 1][0]
        combination = (previous_token, next_token)
    else:
        # Handle the case when the current word is the last in a sentence
        previous_token = tokens[pos - 1][0]
        combination = (previous_token, '<END>')
    return combination


def main() -> None:
    """
    Main function of the program.

    This function reads a set of documents, tokenizes them, and uses the resulting tokens to create
    a transition matrix. Then, it spins a selected article using this matrix and prints the result.

    The function also prints the original article and the changes made during the spinning process
    for comparison purposes.
    """
    nltk.download('wordnet', quiet=True)

    print("Preparing documents...")
    documents = prepare_documents('data')

    print("Creating transition matrix...")
    matrix = create_transition_matrix(documents)

    random_article = random.choice(documents)
    spinned_article, changes = spin_article(random_article, matrix)
    print('------------------------------')
    print('The original article:')
    print(''.join(random_article))
    print()
    print('------------------------------')
    print('The spinned article:')
    print(spinned_article)
    print()
    print('------------------------------')
    print('The changes made during spinning:')
    for tag, val in changes.items():
        print(f"For {tag}, with {len(val)} change{'s'[:len(val) ^ 1]}:")
        print(val)
        print()


if __name__ == '__main__':
    main()
