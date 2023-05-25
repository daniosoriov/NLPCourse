# TF-IDF From Scratch

*TF-IDF From Scratch* is an example I did to implement the TF-IDF (Term Frequency - Inverse Document Frequency)
algorithm from scratch with no external libraries, and only using Python's core libraries.
This is a simple educational script so I could better understand the underlying processes of the TF-IDF algorithm.

## Installation

There is no external library used.

Python version: Python 3.11.3

## Inspiration

This project was built when following a lesson from [The Lazy Programmer](https://github.com/lazyprogrammer), from the
course Machine Learning: Natural Language Processing in Python (V2).

## Data

You need to have a dataset in CSV format on a file named `data.csv`, for this script to work.
The CSV file should have two columns, one for the document, and one for the label. The CSV should also include a
header, as the script ignores the first row of the CSV file. The script assumes the CSV is separated by semicolons: `;`

Example of the `data.csv` file's content:

```csv
Text;Label
This is my first document;Label1
Another document;Label2
Yet another document;Label1
```

## Usage

```bash
python main.py
```

The script will run automatically, transforming the dataset to a TF-IDF matrix, and then to a Compressed Sparse Row
(CSR) Matrix. It will select a random document from your dataset and look for the top 5 frequency terms and print them.
It will then reproduce the vector of the same document and print it.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)