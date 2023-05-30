# Word2Vec GloVe Exploration

*Word2Vec GloVe Exploration* is an assignment I did to implement a couple of useful methods on the 
[GloVe](https://nlp.stanford.edu/projects/glove/) ([GloVe on GitHub](https://github.com/stanfordnlp/GloVe)) 
pre-trained vectors. This project is designed to perform word embeddings analysis using the GloVe model, 
particularly focusing on finding nearest neighbors of a word, and word analogies. 

## Description
The script reads in a pre-trained GloVe model and uses it to perform two types of analysis: 

1. **Nearest Neighbors**: Given a word, the script finds the 'k' nearest neighbors in the vector space of the word embeddings.

2. **Word Analogies**: Given three words (a, b, c), the script finds a fourth word (d) such that the relationship 
between 'a' and 'b' is similar to the relationship between 'c' and 'd'.

## Installation

There is no external library used. However, the following Python libraries are used: `heapq`, `math` and `random`.

Python version: Python 3.11.3

For this code to work, you need to download (clone) the latest GloVe code and follow the instructions they present 
to train the model. You should then add `main.py` to the folder you cloned from GloVe and verify that the file 
`vectors.txt` is there.

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## Usage

```bash
python main.py
```

The script will check on the pre-trained vectors in `vectors.txt` file and perform several tests to find analogies 
and the nearest neighbors of some random words. You can tweak the code to your like on the `main()` method to test 
the script.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)