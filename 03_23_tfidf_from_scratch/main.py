"""
Develop a simple TF-IDF matrix given a set of documents.

This script also creates a Compressed Sparse Row (CSR) matrix from the TF-IDF matrix.
"""
import csv
import random
import re
import math
from typing import Optional, List, Tuple

# Constants
# The data file is assumed to have two columns (document and label)
DATA_FILE = 'data.csv'
# Regex to only include English words in the vocabulary
REGEX_ENGLISH_WORDS = re.compile(r"\b[A-Za-z0-9]+[-'’]*[A-Za-z0-9]*\b", re.IGNORECASE)


def get_documents() -> List[List[str]]:
    """
    Fetch documents from a CSV file and preprocess them.

    :return: A list of documents, each document being a list of words.

    The function opens the 'data.csv' file and reads it line by line. It ignores the first row,
    assuming it's a header. It then extracts all English words in lowercase from the first
    column of each row, and appends them as a document to the "documents" list.
    """
    # Open the CSV file
    with open(DATA_FILE) as file:
        # Create a CSV reader with a specified delimiter
        data = csv.reader(file, delimiter=";")
        # Create a list of documents
        # For each row in the CSV data, excluding the first row (header),
        # extract all English words in lowercase from the first column
        # and append them as a document to the "documents" list
        documents = [re.findall(REGEX_ENGLISH_WORDS, row[0].lower()) for row in list(data)[1:]]
    return documents


def make_tf_idf_matrix(documents: List[List[str]]) -> Tuple[List[List[float]], List[str]]:
    """
    Create a TF-IDF matrix from a list of documents.

    :param documents: List of documents, where each document is a list of tokens.
    :return: A tuple containing a TF-IDF matrix with rows representing documents and columns representing unique words
    across all documents, and a list of indexes to words (vocabulary) representing the column of the matrix.
    """

    # Create a dictionary to map each unique word to a unique index
    word_index = {}

    # Create a dictionary to count the number of documents in which each word appears
    word_frequency = {}

    # Create a list to store information about each document
    doc_info = []
    index = 0

    # Iterate over each document
    for doc in documents:
        # Create a dictionary to count the number of times each word appears in the document
        unique_tokens = {}
        for token in doc:
            # Increment the count of this word in this document
            unique_tokens[token] = unique_tokens.get(token, 0) + 1

            # If this is the first time we have seen this word, assign it a unique index
            if token not in word_index.keys():
                word_index[token] = index
                index += 1

        # Store the count of unique tokens and the length of the document
        doc_info.append({'unique': unique_tokens, 'len': len(doc)})

        # Increment the count of documents in which each unique word appears
        for token in unique_tokens:
            word_frequency[token] = word_frequency.get(token, 0) + 1

    # Create the TF-IDF matrix
    matrix = []
    documents_amount = len(documents)
    for row, doc in enumerate(documents):
        matrix.append([])
        for col, word in enumerate(word_index):
            # Get the count of this word in this document, or zero if the word does not appear in this document
            token_count = doc_info[row]['unique'][word] if word in doc_info[row]['unique'] else 0

            # Calculate the term frequency (TF) for this word in this document and normalize it
            normalized_tf = token_count / doc_info[row]['len']

            # Calculate the inverse document frequency (IDF) for this word
            idf = math.log(documents_amount / (1 + word_frequency[word]))

            # Multiply the TF by the IDF to get the TF-IDF value
            matrix[row].append(normalized_tf * idf)

    vocabulary = [token for token in word_index.keys()]

    return matrix, vocabulary


def print_top_terms(matrix: List[List[float]], vocabulary: List, row: int = None, number: int = 5) -> None:
    """
    Function to find and print the terms with highest TF-IDF scores in a specific or random document.

    Parameters:
    matrix (list[list[float]]): The TF-IDF matrix where each row is a document and each column is a term.
    Vocabulary (list): The list of terms (words) in the same order as they appear in the matrix.
    row (int, optional): The index of the document (row in the matrix) to analyze. If not provided or invalid, a random
    document is chosen.
    number (int, optional): The number of top terms to print. If not provided or less than 1, defaults to 5.

    Returns:
    None. Print the results directly to the console.
    """
    len_matrix = len(matrix)

    # If "row" is not a valid index, select a random document
    if type(row) is not int or row is None or 0 > row > len_matrix:
        row = random.randint(0, len_matrix - 1)

    # If "number" is less than 1, default it to 5
    if number < 1:
        print(f'Parameter "number" should not be less than 1. Setting number parameter to arbitrary default of 5.')
        number = 5

    vector = matrix[row]
    # Create a dictionary mapping indices to term frequencies for non-zero frequencies only
    sparse_vector = {index: tf for index, tf in enumerate(vector) if tf}
    # Sort the items in this dictionary in decreasing order of term frequency
    top_results = sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {number} TF's for document {row}")
    for num, tf_tuple in enumerate(top_results[:number], 1):
        # Print the rank, term, term frequency, and matrix index for each of the top terms
        print(f"{num} - {vocabulary[tf_tuple[0]]} - TF: {tf_tuple[1]} - Matrix index: {tf_tuple[0]}")


def compressed_sparse_row_matrix(matrix: List[List[float]]) -> Tuple[List[float], List[int], List[int]]:
    """
    Convert a 2D array into Compressed Sparse Row (CSR) format.

    :param matrix: A 2D array of floats, representing a matrix.
    :return: A tuple of three lists:
        - data_array: a list of non-zero elements in the matrix.
        - Indices_array: a list of column indices for each non-zero element.
        - Indptr_array: a list of index pointers for the start of each row in the data_array.

    The function loops through each row (document) and column (term frequency - tf) in the input matrix.
    For each non-zero tf, it appends tf to the data_array and the column index to the indices_array.
    At the end of each row, it appends the current length of the data_array to the indptr_array.
    This marks the starting point of the next row in the data_array.
    """
    # List to store non-zero elements in the matrix
    data_array = []
    # List to store column indices of non-zero elements
    indices_array = []
    # List to store the index pointers for the start of each row in data_array
    indptr_array = [0]

    # Loop through each row in the matrix
    for row, doc in enumerate(matrix):
        # Loop through each column in a row
        for col, tf in enumerate(doc):
            # If term frequency is non-zero
            if tf:
                # Add term frequency to data_array
                data_array.append(tf)
                # Add column index to indices_array
                indices_array.append(col)
        # Add the current length of data_array to indptr_array
        indptr_array.append(len(data_array))

    return data_array, indices_array, indptr_array


def get_vector_from_csr(csr_matrix: Tuple[List[float], List[int], List[int]], row: int,
                        vocabulary: Optional[List[str]]) -> List[float]:
    """
    Given a CSR matrix and a row index, this function returns the corresponding vector.
    If a vocabulary is specified, the function returns the full vector (including zero values);
    otherwise, it returns the sparse vector representation.

    :param csr_matrix: A tuple representing a Compressed Sparse Row (CSR) matrix.
                       It includes the data, indices, and indptr arrays.
    :param row: The index of the row (i.e., document) to fetch.
    :param vocabulary: An optional list of the vocabulary words. If specified,
                       the function returns the full vector.
    :return: A list representing the vector corresponding to the specified row.
             This could be either sparse or full based on the vocabulary parameter.
    """
    # Unpack the CSR matrix into its components
    data_array, indices_array, indptr_array = csr_matrix
    # Raise an error if the row parameter is not a valid index
    if row < 0 or row > len(indptr_array) - 1:
        raise ValueError('Invalid row index. Must be in range [0, number of rows in matrix).')

    # Get the start and end index of the row in the data array using the indptr array
    row_begin = indptr_array[row]
    row_end = indptr_array[row + 1]

    # Fetch the non-zero values of the row from the data array
    vector = data_array[row_begin:row_end]

    # If vocabulary is not specified, return the sparse vector
    if vocabulary is None:
        return vector
    else:
        # Initialize a full vector with zeros
        full_vector = [0.0 for _ in vocabulary]
        # Iterate over the indices and non-zero values of the row
        for index, col in enumerate(indices_array[row_begin:row_end]):
            # Assign the non-zero values to their respective positions in the full vector
            full_vector[col] = vector[index]

        # Return the full vector
        return full_vector


def main():
    """
    Main driver function for the TF-IDF processing pipeline.

    This function:
    1. Fetches documents from a CSV file and preprocesses them.
    2. Constructs a TF-IDF matrix from the preprocessed documents.
    3. Print the top TF-IDF terms in a document from the TF-IDF matrix.
    4. Converts the TF-IDF matrix to a compressed sparse row matrix.
    5. Fetches a vector from the CSR matrix and prints it.
    """
    # Step 1: Fetch and preprocess the documents
    documents = get_documents()
    # Step 2: Construct the TF-IDF matrix and the word index-to-vocabulary mapping
    tf_idf_matrix, vocabulary = make_tf_idf_matrix(documents)

    # Get a random document for verification
    random_doc_index = random.randint(0, len(documents) - 1)
    print(f'The selected document is at index {random_doc_index}:')
    print(documents[random_doc_index])

    # Step 3: Print top terms for a random document in the TF-IDF matrix
    print_top_terms(tf_idf_matrix, vocabulary, row=random_doc_index)
    # Step 4: Convert the TF-IDF matrix to a compressed sparse row matrix
    csr_matrix = compressed_sparse_row_matrix(tf_idf_matrix)

    # Step 5: Fetch the vector of the random document and print it.
    vector = get_vector_from_csr(csr_matrix, random_doc_index, vocabulary)
    print(vector)


if __name__ == '__main__':
    main()
