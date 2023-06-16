# Edgar vs Robert - Text Generation and Classification with Markov Models

The concept of this project is to compare poems from Edgar Allan Poe and Robert Frost using Markov models. The models
check class imbalance, calculate F1 scores and generate sample text.

## Table of Contents

1. [Installation](#installation)
2. [How to Use](#how-to-use)
3. [File Structure](#file-structure)
4. [Inspiration](#inspiration)
5. [License](#license)

## Installation

Clone the repository, navigate to the directory of the project, and run the script using Python 3.8 or higher.

Make sure to have the necessary libraries installed. You can install them via pip:

```
pip install numpy
```

## How to Use

The script consists of a single Python file that you can run from the command line as follows:

```
python main.py
```

The script reads two text files: `data/edgar_allan_poe.txt` and `data/robert_frost.txt` and trains two Markov models
using these documents. It then calculates the F1 scores for these models and generates sample text.

You may need to adjust the paths to these text files depending on the structure of your project.

The F1 score is a measure of a test's accuracy, and it considers both the precision and the recall of the test. The F1
score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision
and recall) and worst at 0.

## File Structure

Here's a list of functions included in the script:

* `MarkovModel`: This is a class for creating a Markov model. It includes methods for tokenizing documents, creating a
  transition matrix, checking for class imbalance, calculating the probability of a sequence of states, and generating
  text based on the model.

* `file_to_text`: This function reads a text file and splits it into sentences.

* `check_balance`: This function checks if the Markov models are balanced.

* `calculate_f1_score`: This function calculates the F1 scores for two models using their respective test sets.

* `main`: The main function loads documents, trains Markov models for each author, checks for class imbalance,
  calculates the F1 scores for each model, and generates sample sentences.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## License

This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.