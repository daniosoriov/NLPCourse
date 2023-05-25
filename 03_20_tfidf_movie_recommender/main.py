import numpy as np
import pandas as pd
import json
import os
from joblib import dump, load

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

VECTORIZER_FILE = 'vectorizer.joblib'
TFIDF_MATRIX_FILE = 'tfidf_matrix.joblib'
MOVIES_FILE = 'data/tmdb_5000_movies.csv'
# KEYS = ('genres', 'original_title', 'keywords', 'overview', 'production_companies', 'production_countries',
#         'spoken_languages', 'tagline')
KEYS = ('genres', 'keywords', 'original_title')
# TRANSFORM_KEYS = ('genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages')
TRANSFORM_KEYS = ('genres', 'keywords')


def transform_data(text: str) -> str:
    """
    Converts a JSON-formatted string into a simpler, space-separated string containing only the values
    associated with the "name" keys in the JSON structure.

    This function expects a string which is valid JSON, where each object has a 'name' key. It extracts
    the value of this key for each object, and joins these name values into a single string, with each
    name separated by a space.

    :param text: A string containing a valid JSON structure where each object has a 'name' key.
    :return: A single string where each 'name' value from the JSON is separated by a space.
    """
    return ' '.join([''.join(data['name'].split()) for data in json.loads(text)])


def get_documents() -> tuple[[list], [pd.DataFrame]]:
    """
    This function reads a CSV file containing movie data, performs data cleaning and pre-processing, and
    prepares a list of documents from the dataframe for further analysis or model training. Each document
    is a string composed of all relevant features of a row in the dataframe, converted to lowercase.

    It specifically drops the columns that are not included in the KEYS variable, fills null values with
    an empty string, and applies a data transformation function (transform_data) to the columns specified
    in the TRANSFORM_KEYS variable.

    :return: A tuple of two items. The first item is a list of documents, where each document is a string
    formed from the relevant features of a row in the dataframe. The second item is the list of original titles.
    """
    df = pd.read_csv(MOVIES_FILE)
    drop_keys = [key for key in df.keys() if key not in KEYS]
    df.drop(columns=drop_keys, inplace=True)
    df.fillna('', inplace=True)
    titles = df['original_title']

    for col in TRANSFORM_KEYS:
        df[col] = df[col].apply(transform_data)
    documents = [' '.join(row.values.astype(str)).lower() for _, row in df.iterrows()]

    return documents, titles


def get_wordnet_pos(treebank_tag: str):
    """
    Translates a Penn Treebank tag into an equivalent WordNet part of speech tag.

    This function maps the initial character of a Penn Treebank tag to a corresponding WordNet part of speech tag.
    The mappings are as follows:
        'J' -> wordnet.ADJ
        'V' -> wordnet.VERB
        'N' -> wordnet.NOUN
        'R' -> wordnet.ADV

    If the initial character of the Treebank tag doesn't match any of these, wordnet.NOUN is returned as a default.

    :param treebank_tag: A string representing a Penn Treebank tag.
    :return: The equivalent WordNet part of speech tag.
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


def lemma_tokenizer_tokenize(text: str) -> list[str]:
    """
    Tokenizes and lemmatizes a text string using the WordNetLemmatizer from NLTK.

    The function first tokenizes the input text into individual words. Each word is then lemmatized,
    i.e., reduced to its base or root form (lemma), considering its part of speech.

    The part of speech for each word is determined to use NLTK's pos_tag function and is then translated
    into a WordNet part of speech tag using the get_wordnet_pos function.

    :param text: A string of text to be tokenized and lemmatized.
    :return: A list of lemmatized tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(tokens)]


def ask_movie(titles: pd.Series) -> tuple[str, int]:
    """
    Prompts the user to input a movie title and searches for it in a given pandas Series of movie titles.

    This function converts both the input movie title and Series to lowercase for case-insensitive searching. If the
    input title exactly matches a title in the Series, the function prints the matched title and returns a tuple with
    the input movie title and its index in the Series.

    If there's no exact match, the function performs a partial search to find titles containing the input string. It
    then prints a list of these titles and prompts the user to input another movie title.

    :param titles: A pandas Series of movie titles to search in.
    :return: A tuple containing the input movie title and the index of the matched title in the Series. If there's no
    match, it will recursively call itself until a match is found.
    """
    titles_search = titles.str.lower()
    movie = input('Input a movie to look for similar titles: ').lower()
    mask = titles_search == movie
    title_found = titles[mask]
    if len(title_found) > 0:
        print(f'"{str(titles[title_found.index[0]])}" found on the database.')
        return movie, title_found.index[0]
    else:
        print('No title matches your search. Please try again.')
        mask = titles_search.str.contains(movie)
        titles_found = titles[mask]
        if len(titles_found) > 0:
            print('Did you mean any of the following titles?')
            print(titles_found.to_string(index=False))
            print()
        return ask_movie(titles)


def get_vectorizer(documents: list) -> tuple:
    """
    This function either loads a pre-existing Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer and its
    matrix representation of the documents from file, or creates and saves them if they do not yet exist.

    The function checks if the vectorizer file already exists. If it does, the function loads the vectorizer and its
    matrix from a file. If the files do not exist, it creates a new TF-IDF vectorizer, fits and transforms the provided
    documents, and then saves the vectorizer and matrix to file for future use.

    The vectorizer uses a custom tokenizer that applies lemmatization, and it discards English stop words.

    :param documents: A list of document strings to vectorize.
    :return: A tuple containing the vectorizer (TfidfVectorizer object) and its matrix representation of the documents
             (scipy.sparse.csr.csr_matrix).
    """
    if os.path.exists(VECTORIZER_FILE):
        print('Model existed, loading from file...')
        vectorizer = load(VECTORIZER_FILE)
        X = load(TFIDF_MATRIX_FILE)
    else:
        print('Model did not exist yet, creating it...')
        vectorizer = TfidfVectorizer(tokenizer=lemma_tokenizer_tokenize, token_pattern=None, max_features=2000)
        X = vectorizer.fit_transform(documents)
        dump(vectorizer, VECTORIZER_FILE)
        dump(X, TFIDF_MATRIX_FILE)
    print('Model ready')
    return vectorizer, X


def main():
    """
    This is the main function that executes a series of tasks to find and display the top 5 movies that are most
    similar to a user-specified movie based on the cosine similarity of their TF-IDF vectors.

    The function executes the following sequence of tasks:

    1. Download the 'wordnet' corpus using NLTK, which will be used for lemmatization during text preprocessing.
    2. Obtain the list of document strings and a Series containing movie titles by calling the 'get_documents'
    function.
    3. Ask the user to input a movie title and find it in the list of original titles from the Series.
    4. Either load a pre-existing TF-IDF vectorizer and its matrix representation of the documents from file,
       or create and save them if they do not yet exist.
    5. Transform the user-specified movie's document string into its vector representation using the vectorizer.
    6. Calculate the cosine similarities between the vector of the user-specified movie and those of all other movies.
    7. Identify the indices of the top 5 most similar movies and their respective similarity scores.
    8. Construct a DataFrame containing the rank, similarity index, similarity percentage, and title of the top 5
    most similar movies.
    9. Print the DataFrame to display the result to the user.
    """
    # Ensure necessary resources are available
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

    # Obtain data and user input
    documents, titles = get_documents()
    movie, movie_index = ask_movie(titles)

    # Save these for later use
    movie_original_title = titles[movie_index]
    movie_complete = documents[movie_index]

    # Get the TF-IDF vectorizer and the TF-IDF matrix
    vectorizer, X = get_vectorizer(documents)

    # Get a TF-IDF vector for the chosen movie
    new_vector = vectorizer.transform([movie_complete])

    print('Looking for similar titles...')
    # Compute cosine similarities
    cosine_similarities = cosine_similarity(new_vector, X)

    # Get the top 5 most similar movies (-6 because the movie itself will be in this list)
    top_indices = np.argsort(cosine_similarities.flatten())[-6:-1]
    top_similarities = cosine_similarities.flatten()[top_indices]

    # Prepare the data to display
    movies_found = {'Rank': [], 'Similarity Index': [], 'Similarity %': [], 'Title': []}
    rank = 1
    for index, similarity in zip(reversed(top_indices), reversed(top_similarities)):
        movies_found['Rank'].append(rank)
        movies_found['Similarity Index'].append(f"{similarity:.2}")
        movies_found['Similarity %'].append(f"{(similarity + 1) / 2:.2%}")
        movies_found['Title'].append(titles[index])
        rank += 1

    # Display the results
    movies_df = pd.DataFrame(movies_found)
    print(f'Top 5 similar movies to "{movie_original_title}" using Cosine Similarity:')
    print(movies_df.to_string(index=False))


if __name__ == '__main__':
    main()
