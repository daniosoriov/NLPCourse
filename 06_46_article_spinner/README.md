# Article Spinner

This is an educational project that implements an article spinner using Natural Language Processing (NLP) techniques.
The script reads a collection of text documents, tokenizes them, and generates a transition matrix. This matrix is then
used to "spin" a selected article, generating a new version with alternative word choices.

The transition matrix uses the technique of keeping track of the previous and next word of any given word in the
document. The matrix calculates the probabilities for all combinations, and based on the type of word and its
probability of occurrence, it changes certain words in the original article.

Lastly, when swapping words, the script uses a probabilistic approach to changing only nouns (N), verbs (V),
adjectives (J) and adverbs (R). This means that not all of these word types (N, V, J, R) will change.

Please note that the main purpose of this project is to provide a simplified model of how article spinners work. This
script may not produce perfect results and is not intended for production use.

## Features

1. Load text documents from a folder.
2. Tokenize documents using the Natural Language Toolkit (NLTK).
3. Generate a transition matrix based on preceding and following words for each word of interest in the text.
4. Save the generated matrix (`matrix.pkl`) and documents object (`documents.pkl`) to file for future usage to
   increase efficiency.
5. Spin an article based on the transition matrix, replacing certain words with alternatives.
6. Display the original and spun versions of the article, along with the changes made.

## Requirements

* Python 3.7+
* NLTK
* Numpy

## Setup & Usage

1. Clone the repository and navigate to the project directory. The repository already includes a dataset of business
   articles from the BBC stored in the folder `data`. However, you could add your own data files following the same
   structure and format.

2. Install the requirements:

    ```commandline
    pip install nltk
    pip install numpy
    ```

3. Run the script:

    ```commandline
    python main.py
    ```

## Methods

* `prepare_documents`: Reads all text documents from the specified folder and returns a list of documents.

* `tokenize_document`: Tokenizes a document and returns a list of tokens with their POS tags.

* `create_transition_matrix`: Creates a transition matrix from a list of documents.

* `spin_article`: Spins an article using the transition matrix.

* `create_combination`: Creates a combination of preceding and following words for a given position in a list of tokens.

* `main`: The main function that orchestrates the process of preparing documents, creating the transition
  matrix, and spinning an article.

## Example output

This is an output example for the script. Given the article:

```text
Winn-Dixie files for bankruptcy

US supermarket group Winn-Dixie has filed for bankruptcy protection after succumbing to stiff competition in a market dominated by Wal-Mart.

Winn-Dixie, once among the most profitable of US grocers, said Chapter 11 protection would enable it to successfully restructure. It said its 920 stores would remain open, but analysts said it would most likely off-load a number of sites. The Jacksonville, Florida-based firm has total debts of $1.87bn (£980m). In its bankruptcy petition it listed its biggest creditor as US foods giant Kraft Foods, which it owes $15.1m.

Analysts say Winn-Dixie had not kept up with consumers' demands and had also been burdened by a number of stores in need of upgrading. A 10-month restructuring plan was deemed a failure, and following a larger-than-expected quarterly loss earlier this month, Winn-Dixie's slide into bankruptcy was widely expected. The company's new chief executive Peter Lynch said Winn-Dixie would use the Chapter 11 breathing space to take the necessary action to turn itself around. "This includes achieving significant cost reductions, improving the merchandising and customer service in all locations and generating a sense of excitement in the stores," he said. Yet Evan Mann, a senior bond analyst at Gimme Credit, said Mr Lynch's job would not be easy, as the bankruptcy would inevitably put off some customers. "The real big issue is what's going to happen over the next one or two quarters now that they are in bankruptcy and all their customers see this in their local newspapers," he said.
```

The resulting spinned article could be:

```text
Winn-Dixie files for Japan

US supermarket group Winn-Dixie has filed for bankruptcy protection after succumbing to stiff competition in a market dominated by £200m.

Winn-Dixie, once among the most profitable of US assets, said Chapter 11 protection would make it to successfully restructure. It said its 920 stores would remain solid, but analysts said it would most likely off-load a number of sites. The rise, Florida-based firm has total cost of $1.87bn (£350m). In its bankruptcy petition it listed its biggest creditor as US banking giant Kraft Foods, which it owes $15.1m.

Analysts say Winn-Dixie had not kept up with consumers' demands and had not been tainted by a decree of obesity in need of upgrading. A 10-month restructuring plan was cutting a result, and following a larger-than-expected quarterly profit earlier this time, France's investigation into bankruptcy was widely expected. The paper's new chief executive Peter Chernin said Chalone would use the Chapter 11 breathing space to woo the necessary way to extricate itself around. "This includes achieving significant cost carriers, is the merchandising and customer service in all prices and ordered a fifth of increase in the LSE," he said. Yet Evan Mann, a senior bond strategist at Gimme Credit, said Mr Ghosn's government would only be easy, as the bankruptcy would inevitably put off some customers. "The real big issue is what's going to happen over the next one or two quarters now that they are in order and all their customers see this in their local newspapers," he said.
```

As you can quickly notice, there are several swaps that work, and several that do not work and completely change the
meaning or understanding of the original article. Hence, it is not a bullet-proof script, but rather an educational
project to better comprehend what happens behind the curtains when making an article spinner using NLP techinques.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## License

This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.