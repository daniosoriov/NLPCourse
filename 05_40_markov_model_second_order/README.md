# Second Order Markov Model for Robert Frost's Poems

This project uses a Markov model to generate text that mimics the style of Robert Frost's poems. Each line in the corpus
of text is treated as a document to train the model. The model is a second-order Markov model which means it uses the
current and previous states to determine the next state. This script does not rely on any ML libraries on purpose.

## Installation

Clone the repository, navigate to the directory of the project, and run the script using Python 3.8 or higher.

## How to Use

The script can be run from the command line as follows:

```
python main.py
```

This will generate a series of lines that resemble the style of Robert Frost's poems.

The script reads poems from a file named `robert_frost.txt` in the `data` directory.
The format of the file should be plain text where each line represents a separate line from a poem.

## File Structure

The script is divided into several functions, each with a specific task.

1. `get_docs_tokenized`: This function reads a text file, tokenizes each line, and returns a list of tokenized
   documents.

2. `create_pi`: This function calculates the initial state probabilities for the Markov model.

3. `create_transition_prob`: This function creates the first-order and second-order transition probabilities.

4. `make_choice`: This function randomly selects a state based on its probability.

5. `create_sentence`: This function generates sentences based on the Markov model.

6. `main`: This is the main function of the script. It calls the other functions in order to generate the
   text.

## Example Output

Running the script generates text that resembles the style of Robert Frost's poems. Here is an example output:

```
Is troubling granny though
And bring the chalk-pile down i'll tell you who'd remember heman lapish
Someday make his fortune
Till she gave judgment
When he's gone to bed alone and left me
Are riders
But twas not for him
I mean a man at least to pass a winter garden in an old cellar hole
I err
That's what he doesn't make much
```

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## License

This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.