# TF-IDF Movie Recommender

TF-IDF Movie Recommender is a simple machine learning program using the NLTK TfidfVectorizer (Term Frequency - Inverse 
Document Frequency) to recommend 5 different movies based on a movie.

## Installation

Check the `requierements.txt` file for the specific requirements. 

In short, it requires the libraries `numpy`, `pandas`, `nltk`, `joblib`, 
and `scikit-learn`.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## Usage

```bash
python main.py
```

The script will then prompt asking you for a movie. If the movie is in the database, it will train the model using 
TF-IDF, and print the top 5 recommendations based on your input. 

### Saving the model
The first time you run the script, the model needs to be trained, and it will take some time to 
provide an answer. The script will save the vectorizer and the matrix to two different files:
- The TF-IDF vectorizer: `vectorizer.joblib`
- The matrix: `tfidf_matrix.joblib`

The next time you run the script, it will be faster, almost immediate, because it will load the vectorizer and matrix 
from the saved files. If you want to re-train the model with your customizations, remember to delete the `.joblib` 
files.

## Database

The database is a ~5MB library (`data/tmdb_5000_movies.csv`) with 5000 titles. You can only search from titles within 
the database. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)