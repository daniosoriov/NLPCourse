"""
The assignment is to create your own version of two useful methods for the pretrained GloVe vectors:
 * find_analogies()
 * nearest_neighbors()

****************************************
Exercise: download pretrained GloVe vectors from
https://nlp.stanford.edu/projects/glove/

Implement your own find_analogies() and nearest_neighbors()

Hint: you do NOT have to go hunting around on Stackoverflow
      you do NOT have to copy and paste code from anywhere
      look at the file you downloaded
****************************************
"""

import math
import heapq
import random
from typing import Dict, List, Tuple

# This is the file of vectors produced after pretraining the GloVe model
VECTOR_FILE = 'vectors.txt'


def get_matrix(filename: str) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Function to read the GloVe vectors from the specified file and calculate their norms.

    :param filename: Path to the file containing the word vectors.
    :return: matrix (dict): Dictionary mapping each word to its corresponding vector representation.
             norms (dict): Dictionary mapping each word to the norm of its corresponding vector.
    """
    # Initialize dictionaries to store the vectors and their norms
    matrix = {}
    norms = {}
    try:
        with open(filename) as file:
            for line in file:
                # Split the line into word and vector components
                word_vector_string = line.strip().split(' ')
                # The first element is the word
                word = word_vector_string[0]
                # The remaining elements form the vector, convert them to floats
                vector = [float(val) for val in word_vector_string[1:]]
                # Store the vector in the matrix
                matrix[word] = vector
                # Compute and store the norm of the vector
                norms[word] = math.sqrt(sum([val * val for val in vector]))
    # Catch exception if file is not found and raise it further
    except FileNotFoundError as error:
        raise FileNotFoundError(error)
    return matrix, norms


def get_random_words(k: int = 5) -> List[str]:
    """
    Function to get k random words from a pre-specified GloVe vector file.

    :param k: The number of random words to select from the vector file. Default is 5.
    :return: A list of k random words selected from the vector file.
    """
    with open(VECTOR_FILE) as file:
        # Extract the word (first element) from each line in the vector file
        words = [line.strip().split(' ')[0] for line in file]
    # Use Python's built-in random.choices() to select k words at random
    return random.choices(k=k, population=words)


def nearest_neighbors(matrix: Dict[str, List[float]], norms: Dict[str, float], lookup_word: str,
                      k: int = 5) -> List[Tuple[float, str]]:
    """
    Given a word, find the 'k' most similar words based on their vector representations.

    This function is a wrapper over the find_similarities function that prepares the lookup vector
    and its norm for the given word.

    :param matrix: A dictionary mapping words to their vector representations.
    :param norms: A dictionary mapping words to the norms of their vector representations.
    :param lookup_word: The word to find similar words for.
    :param k: The number of similar words to return.
    :return: A list of tuples, where each tuple contains a word and its similarity to the lookup word.
             The list is sorted in descending order of similarity.
    """
    # Check if the lookup word is in the vocabulary
    if lookup_word not in matrix.keys():
        raise ValueError(f'The word {lookup_word} is not in the vocabulary')

    # Get the vector representation and norm for the lookup word
    lookup_vector = matrix[lookup_word]
    norm_lookup_vector = norms[lookup_word]

    # Call the find_similarities function with the lookup vector, its norm, and the word to ignore (the lookup word
    # itself)
    return find_similarities(matrix, norms, ignore_words=[lookup_word], lookup_vector=lookup_vector,
                             norm_lookup_vector=norm_lookup_vector, k=k)


def find_similarities(matrix: Dict[str, List[float]], norms: Dict[str, float], lookup_vector: List[float],
                      norm_lookup_vector: float, ignore_words: List[str], k: int = 5) -> List[Tuple[float, str]]:
    """
    Given a lookup vector, find the 'k' most similar words based on their vector representations.

    This function is used for nearest neighbors search and also for finding analogies.

    :param matrix: A dictionary mapping words to their vector representations.
    :param norms: A dictionary mapping words to the norms of their vector representations.
    :param lookup_vector: The vector to find similar vectors for.
    :param norm_lookup_vector: The norm of the lookup vector.
    :param ignore_words: A list of words to ignore during the search. This is particularly useful
                         when finding analogies to avoid returning the input words.
    :param k: The number of similar vectors to return.
    :return: A list of tuples, where each tuple contains a word and its similarity to the lookup vector.
             The list is sorted in descending order of similarity.
    """
    # Initialize the list of similarities
    similarities = []
    len_similarities = 1

    # For every word in the vocabulary...
    for word, vector in matrix.items():
        # Skip the words to ignore
        if word in ignore_words:
            continue

        # Calculate the dot product of the lookup vector and the current word's vector
        dot_product = sum([val * lookup_vector[num] for num, val in enumerate(vector)])
        norm_vector = norms[word]

        # Get the norm for the current word's vector
        norm = norm_vector * norm_lookup_vector

        # Calculate the denominator for the cosine similarity formula
        if norm > 0:
            # Calculate the cosine similarity
            similarity = dot_product / norm

            # Use a heap to keep track of the top k similarities
            heapq.heappush(similarities, (similarity, word))
            if len_similarities <= k:
                len_similarities += 1
            else:
                heapq.heappop(similarities)

    # Sort the similarities in descending order before returning
    return sorted(similarities, reverse=True)


def find_analogies(matrix: Dict[str, List[float]], norms: Dict[str, float], word1: str, word2: str, word3: str) \
        -> List[Tuple[float, str]]:
    """
    Given three words, find an analogous fourth word based on their vector representations.

    This function performs the operation "word2 - word1 + word3" in the vector space,
    and then returns the word whose vector representation is closest to the result of that operation.

    The words should be in the form of "A (word1) is to B (word2) what C (word3) is to ___".

    :param matrix: A dictionary mapping words to their vector representations.
    :param norms: A dictionary mapping words to the norms of their vector representations.
    :param word1: The first word in the analogy.
    :param word2: The second word in the analogy.
    :param word3: The third word in the analogy.
    :return: The analogous fourth word, based on their vector representations.
    """
    # Check if the given words are in the vocabulary
    if not any(val in matrix.keys() for val in (word1, word2, word3)):
        raise KeyError('Please provide words that exist in the vocabulary.')

    # Get the vector representations of the given words
    vector1 = matrix[word1]
    vector2 = matrix[word2]
    vector3 = matrix[word3]
    vector4 = []

    # Perform the operation "word2 - word1 + word3" in the vector space
    for col, val2 in enumerate(vector2):
        vector4.append(val2 - vector1[col] + vector3[col])

    # Compute the norm of the result vector
    norm_lookup_vector = math.sqrt(sum([val * val for val in vector4]))
    # Find the word whose vector representation is closest to the result of the operation
    word4 = find_similarities(matrix, norms, lookup_vector=vector4, norm_lookup_vector=norm_lookup_vector, k=1,
                              ignore_words=[word1, word2, word3])

    print(f'{word1} is to {word2} what {word3} is to: {word4[0][1]}')
    return word4


def main():
    """
    This is the main driver function of the script.

    It starts by reading the vector representations of words and their norms using the `get_matrix` function.
    It then gets a list of random words using the `get_random_words` function. For each of these words,
    it finds and prints their nearest neighbors using the `nearest_neighbors` function.

    Finally, it uses the `find_analogies` function to find analogies for a set of words' triplets.
    """
    # Get the vector representations of words and their norms
    matrix, norms = get_matrix(VECTOR_FILE)

    # Get a list of random words
    random_words = get_random_words()

    # For each random word, find and print its nearest neighbors
    for word in random_words:
        neighbors = nearest_neighbors(matrix, norms, word)
        print(f'Closest neighbors to "{word}"')
        for val, neighbor in neighbors:
            print(f'{neighbor} - {val:.2f}')
        print()

    # Find and print analogies for a set of words' triplets
    find_analogies(matrix, norms, 'man', 'king', 'woman')
    find_analogies(matrix, norms, 'france', 'paris', 'london')
    find_analogies(matrix, norms, 'france', 'paris', 'rome')
    find_analogies(matrix, norms, 'france', 'paris', 'italy')
    find_analogies(matrix, norms, 'france', 'french', 'english')
    find_analogies(matrix, norms, 'japan', 'japanese', 'italian')
    find_analogies(matrix, norms, 'japan', 'japanese', 'chinese')
    find_analogies(matrix, norms, 'miami', 'florida', 'texas')


if __name__ == '__main__':
    main()
