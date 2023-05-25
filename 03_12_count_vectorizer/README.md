# Incubus vs Taylor Swift

*Incubus vs Taylor Swift* is a machine learning script I created to check whether a song lyric is either from Incubus
or Taylor Swift. This was my first attempt at creating a machine learning script.

## Why Incubus vs Taylor Swift?

Why not? I love Incubus, and it is one of my all-time favorite bands. That doesn't mean I dislike Taylor Swift, but I do
think their style is different, so I made this script to see if there's indeed a difference between the two of them by
using their lyrics as a differentiation point.

## Installation

Check the `requierements.txt` file for the specific requirements.

In short, it requires the libraries `numpy`, `scipy`, `nltk`, `joblib`,
and `scikit-learn`.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## Usage

```bash
python main.py
```

**Attention**: You need to have the training and testing data on the folders `training` and `testing`. See below

## Dataset

The training dataset includes 227 songs:

- 113 from Incubus
- 114 from Taylor Swift

The training dataset includes 42 songs:

- 20 from Incubus
- 1 from OpenAI GPT4 in the style and format of Incubus
- 20 from Taylor Swift
- 1 from OpenAI GPT4 in the style and format of Taylor Swift

### Accuracy

The highest accuracy I could obtain using the test 42 songs was 77%. Meaning, about seven songs out of ten are correctly
classified as being from the right artist, either Incubus or Taylor Swift.

### Training the model

The script trains a model by trying a bunch of hyperparameters, after making many tests using different types of
algorithms; I found the highest score of accuracy (97%) on my training data with RandomForestClassifier from sklearn.
The hyperparameters used to get the highest score are in the code itself, so you can check them out if you want to.

### Replicating the results

You need to have a folder `training` with files for the songs of each artist in the following format:

```text
training/incubus_song1.txt
training/incubus_song2.txt
training/taylorswift_song1.txt
training/taylorswift_song2.txt
```

And another folder `testing` with files for the songs to test the model against:

```text
testing/incubus_song3_test.txt
testing/incubus_song4_test.txt
testing/taylorswift_song3_test.txt
testing/taylorswift_song4_test.txt
```

## The model

After running the script for the first time, it will create a file to save the vectorizer and the model in two separate
`.joblib` files inside a folder `model`. The first time the script runs, it will be slow.

Next time you run the script, the script will load the saved model and vectorizer to run the tests, so it will be way
faster.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)