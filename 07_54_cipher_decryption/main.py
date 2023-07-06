"""
Create a character unigram and bigram matrix from a dataset (moby_dick.txt)
Use a snippet of another random text to created the crypted message using a random character cipher.

Then, with a random substitution cipher, calculate the likelihood probability of the results to be real English.
Create a loop where we can check the random cipher and have mutiple children using a swapping technique, then we keep
only the best children and make the same for them, until we get to the best possible result.
"""

import string
import numpy as np
from typing import List, Dict, Tuple


def load_document(filename: str) -> List[str]:
    """
    Reads a text file and returns its content as a list of words.

    The function performs the following operations:
    - Reads the file line by line.
    - Strips leading and trailing space from each line.
    - Converts each line to lowercase.
    - Splits each line into words and extends the document list with the words.

    Parameters:
        filename (str): Path to the file to read.

    Returns:
        List[str]: A list of words in the document.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Initialize an empty list to store the words in the document
    document = []

    try:
        # Open the file with 'utf-8-sig' encoding to remove any byte order mark (BOM)
        with open(filename, encoding='utf-8-sig') as file:
            # Read the lines in the file
            lines = file.readlines()

            # Loop over each line in the file
            for line in lines:
                # Remove leading and trailing space and convert to lowercase
                line = line.strip().lower()

                # Split the line into words
                words = line.split()

                # If the line is not empty, extend the document list with the words
                if len(words) > 0:
                    document.extend(words)

    except FileNotFoundError as error:
        # If the file does not exist, print the error message
        print(error)

    # Return the list of words in the document
    return document


def create_matrices(document: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    This function creates the initial character probabilities and transition matrix for a given corpus.

    Parameters:
        document (List[str]): A list of words from the document.

    Returns:
        Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        A tuple of two dictionaries representing the initial character probabilities and the transition matrix.
    """

    # Dictionary to hold the initial character probabilities
    init_char_probs = {}

    # Dictionary to hold the transition probabilities
    transition_matrix = {}

    # Loop over all words in each document
    for word in document:
        # Loop over all characters in each word
        for num in range(len(word)):
            # Current character
            char = word[num]

            # If we are at the last character, we use a placeholder <END> to indicate it is the last
            # Else, the next character is the subsequent character in the word
            next_char = '<END>' if num + 1 == len(word) else word[num + 1]

            # When we are checking the first character of a word
            if num == 0:
                # Increment the count of this character as a first character of a word
                init_char_probs[char] = init_char_probs.get(char, 0) + 1
            # When checking the rest of the word
            else:
                # Increment the count of this character followed by `next_char`
                transition_matrix[char] = transition_matrix.get(char, {})
                transition_matrix[char][next_char] = transition_matrix[char].get(next_char, 0) + 1

    # Smoothing for initial character probabilities and transition matrix, we include also <END>
    vocabulary = [char for char in f"{string.ascii_lowercase}{string.punctuation}{string.digits}"] + ['<END>']
    for char in vocabulary:
        for next_char in vocabulary:
            # If a character combination has not been encountered, initialize it with 1
            init_char_probs[char] = init_char_probs.get(char, 1)
            transition_matrix[char] = transition_matrix.get(char, {})
            if not transition_matrix[char].get(next_char, 0):
                transition_matrix[char][next_char] = 1

    # Normalizing the initial character probabilities
    total = sum(init_char_probs.values())
    for char, num in init_char_probs.items():
        init_char_probs[char] = num / total

    # Normalizing the transition matrix
    for char, next_chars in transition_matrix.items():
        total = sum(next_chars.values())
        for next_char, num in next_chars.items():
            transition_matrix[char][next_char] = num / total

    # Return both matrices
    return init_char_probs, transition_matrix


def create_random_mappings(amount: int, seed: int = None) -> List[Dict[str, str]]:
    """
    Generate a list of crypting mappings.

    Each mapping is a dictionary where the keys are English lowercase letters, and the values are shuffled
    English lowercase letters. For example, {'a': 'x', 'b': 'd', ... }

    Parameters:
        amount (int): Number of mappings to generate.
        seed (int, optional): If provided, sets the random seed to ensure reproducibility.

    Returns:
        List[Dict[str, str]]: A list of mappings.

    Raises:
        ValueError: If the amount parameter is less than 1.
    """
    # Validate the amount parameter
    if amount < 1:
        raise ValueError('The value of "amount" has to be a positive int.')

    # If a seed value is provided, use it to set the random seed for reproducibility
    if seed:
        np.random.seed(seed)

    # Initialize an empty list to store the mappings
    mappings = []

    # List of all lowercase English letters
    voc = [char for char in string.ascii_lowercase]

    # Create the specified number of mappings
    for _ in range(amount):
        # Shuffle the list of letters
        np.random.shuffle(voc)

        # Create a new mapping where the keys are English lowercase letters,
        # and the values are the shuffled English lowercase letters
        new_voc = {char: new_char for char, new_char in zip(string.ascii_lowercase, voc)}

        # Add the new mapping to the list of mappings
        mappings.append(new_voc)

    # Return the list of mappings
    return mappings


def crypt_decrypt_message(document: List[str], mapping: Dict[str, str]) -> List[str]:
    """
    Applies a given character mapping to a list of words, effectively serving as an encryption or decryption mechanism.

    For each word in the document, this function transforms each character based on the provided mapping.
    Characters that are not in the English lowercase alphabet are left unchanged.

    Parameters:
        document (List[str]): The document to apply the mapping to, represented as a list of words.
        mapping (Dict[str, str]): A dictionary mapping each character to another character.

    Returns:
        List[str]: The transformed document, represented as a list of words.
    """

    # Initialize an empty list to store the transformed words
    crypted_message = []

    # Iterate over each word in the document
    for word in document:
        # Initialize an empty list to store the characters of the new word
        new_word = []

        # Iterate over each character in the word
        for char in word:
            # If the character is not an English lowercase letter, leave it unchanged
            if char not in string.ascii_lowercase:
                new_char = char
            # Otherwise, replace it according to the mapping
            else:
                new_char = mapping[char]

            # Add the new character to the new word
            new_word.append(new_char)

        # Join the characters of the new word into a single string and add it to the list of transformed words
        crypted_message.append(''.join(new_word))

    # Return the list of transformed words
    return crypted_message


def swap_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Swaps the values of two randomly chosen keys in a character mapping.

    This function first creates a copy of the provided mapping, then randomly selects two keys from it and swaps
    their values.
    The function then returns the new mapping, leaving the original unchanged.

    Parameters:
        mapping (Dict[str, str]): The character mapping to swap two values in.

    Returns:
        Dict[str, str]: The new character mapping, with the values of two keys swapped.
    """

    # Create a copy of the original mapping to avoid changing it
    mapping = mapping.copy()

    # Randomly select two keys from the mapping
    choices = np.random.choice(list(mapping.keys()), 2)

    # Swap the values associated with the chosen keys
    mapping[choices[0]], mapping[choices[1]] = mapping[choices[1]], mapping[choices[0]]

    # Return the new mapping
    return mapping


def calculate_score(crypted_message: List[str], mappings: List[Dict[str, str]],
                    init_char_probs: Dict[str, float], transition_matrix: Dict[str, Dict[str, float]]) -> \
        List[Tuple[int, float]]:
    """
    Calculates the log probability scores for each mapping applied to the crypted message based on given initial
    character probabilities and transition matrix.

    For each mapping, the function first decrypts the crypted message, then iterates over each word and character in
    the decrypted message to calculate its log probability based on the initial character probabilities (for the first
    character in each word) and transition matrix (for every subsequent character).

    The scores are sorted in descending order before returning.

    Parameters:
        crypted_message (List[str]): The crypted message as a list of words.
        mappings (List[Dict[str, str]]): The list of mappings to score.
        init_char_probs (Dict[str, float]): The initial character probabilities.
        transition_matrix (Dict[str, Dict[str, float]]): The transition matrix.

    Returns:
        List[Tuple[int, float]]: A list of tuples, where the first element of each tuple is the index of a mapping in
        'mappings' and the second is its corresponding score.
    """

    scores = []

    # Iterate over each mapping
    for map_num, mapping in enumerate(mappings):
        prob = 0

        # Decrypt the message using the current mapping
        decrypted_message = crypt_decrypt_message(crypted_message, mapping)

        # Iterate over each word in the decrypted message
        for word in decrypted_message:

            # Iterate over each character in the word
            for num, char in enumerate(word):

                # If the character is the first one in the word, add its initial probability to the total
                if num == 0:
                    prob += np.log(init_char_probs[char])
                else:
                    # For each subsequent character, add the log probability of transitioning from the current
                    # character to the next (or to '<END>' if the current character is the last one in the word)
                    next_char = '<END>' if num + 1 == len(word) else word[num + 1]
                    prob += np.log(transition_matrix[char][next_char])

        # Append the mapping index and its corresponding score to the scores list
        score = (map_num, prob)
        scores.append(score)

    # Sort the scores in descending order
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores


def compare_mappings(map_a: Dict[str, str], inverse_map_b: Dict[str, str]) -> float:
    """
    Compares two mappings and returns the percentage of characters that are mapped identically in both mappings.

    The function considers 'map_a' and 'inverse_map_b' (the inverse of a mapping 'map_b'), comparing the characters
    that 'map_a' maps each character to against the characters that 'inverse_map_b' maps back to the original.

    Parameters:
        map_a (Dict[str, str]): The first mapping to compare, with keys as original characters and values as
            mapped characters.
        inverse_map_b (Dict[str, str]): The inverse of the second mapping to compare, with keys as mapped characters
            and values as original characters.

    Returns:
        float: The percentage of characters that are mapped identically in both mappings.
    """

    # Create a list comprehension with 1s for each character that is mapped to the same character in both mappings.
    same_mapping_chars = [1 for from_char, to_char in map_a.items() if inverse_map_b[to_char] == from_char]

    # Calculate and return the percentage of identical mappings by summing the 1s and dividing by the total
    # number of characters.
    return sum(same_mapping_chars) / len(map_a)


def main(num_epochs: int = 300, num_initial_mappings: int = 50, num_offsprings: int = 15, num_scores: int = 15) -> None:
    """
    Main function to execute the cryptanalysis of a document using a genetic algorithm.

    It starts by loading the training document 'moby_dick.txt' and creating the initial character probabilities and
    transition matrix based on it. Then, it creates a random initial mapping and uses it to crypt the document
    'sample_text.txt'. It prints out the original text and the crypting mapping used.

    Then it performs the specified number of epochs, each one consisting of the following steps:
    1. Create offsprings from the current mappings by swapping characters.
    2. Calculate the score for each mapping.
    3. Keep only the best-scoring mappings.
    4. If the scores didn't change from the previous epoch, the best solution was found and it stops.

    After all epochs or upon finding a solution, it prints the best mapping found and the decrypted text, and
    calculates the accuracy of the mapping.

    Parameters:
        num_epochs (int): The number of epochs to perform. Default is 300.
        num_initial_mappings (int): The number of initial random mappings to create. Default is 50.
        num_offsprings (int): The number of offsprings to create from each mapping. Default is 15.
        num_scores (int): The number of best-scoring mappings to keep after each epoch. Default is 15.
    """

    # Load the document 'moby_dick.txt' and calculate the initial character probabilities and transition matrix
    document = load_document('data/moby_dick.txt')
    init_char_probs, transition_matrix = create_matrices(document)

    # Create a random initial mapping and crypt the document 'sample_text.txt' with it
    initial_mapping = create_random_mappings(1)[0]
    document = load_document('data/sample_text.txt')
    crypted_message = crypt_decrypt_message(document, initial_mapping)

    # Print the original text and the crypting mapping
    print('The original text is:')
    print(' '.join([line for line in document]))
    print('The original crypting mapping is:')
    print(initial_mapping)

    count_epochs = 0
    previous_scores = []

    # Create the initial random mappings
    mappings = create_random_mappings(num_initial_mappings)
    print()
    print(f'Performing {num_epochs} epochs, starting with {num_initial_mappings} initial mappings, '
          f'creating {num_offsprings} offsprings per mapping, and keeping the best {num_scores} mappings per epoch...')

    # Perform the epochs
    for i in range(num_epochs):
        count_epochs += 1
        # We only create offsprings after checking the parents
        if i > 0:
            offsprings = []
            for mapping in mappings:
                for _ in range(num_offsprings):
                    offspring = swap_mapping(mapping)
                    offsprings.append(offspring)
            mappings.extend(offsprings)

        # Calculate the score for each mapping, and keep only the best ones
        scores = calculate_score(crypted_message, mappings, init_char_probs, transition_matrix)[:num_scores]

        # Check if the scores didn't change from the previous epoch
        counts = 1
        if previous_scores != scores:
            previous_scores = scores
        else:
            print(f'The best solution was found with a {scores[0][1]} score after {count_epochs} epochs. Stopping now.')

    # Create a new list of the best mappings based on their scores
    mappings = [mappings[num] for num, _ in scores]

    # Print the results
    print()
    print('Results:')
    best_mapping = mappings[0]
    print(best_mapping)
    print(' '.join(crypt_decrypt_message(crypted_message, best_mapping)))

    # Calculate and print the accuracy of the best mapping
    accuracy = compare_mappings(initial_mapping, best_mapping)
    print(f"The accuracy is: {accuracy:.2%}")


if __name__ == '__main__':
    main(num_epochs=300, num_initial_mappings=50, num_offsprings=15, num_scores=15)
