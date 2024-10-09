# Advanced Information Retireval Models: TF-IDF and BM25 Vector Search

This project implements two advanced information retrieval models: TF-IDF and BM25. The goal of the project is to retrieve relevant answers to questions about traveling. The project includes tokenizing text, creating term indices, and performing searches based on query vectors using both models. The results are evaluated using standard metrics such as Precision@1, Precision@5, nDCG@5, MRR, and MAP.

## Project Overview
  1. Implement TF-IDF and BM25 information retrieval models.
  2. Use a vector space model and inverted index to store token frequencies and retrieve relevant documents.
  3. Evaluate the results using multiple metrics, including Precision@1, Precision@5, nDCG@5, MRR, and MAP.
  4. Produce a ski-jump plot based on Precision@5 to analyze the performance of both models.

## Setup and Installation
### Prerequisites
This project requires Python 3.x and the following Python libraries:

* beautifulsoup4
* ranx
* tqdm
* matplotlib
* re
* json
* gzip
* csv

  You can install the dependencies using pip:
  
  ```python
  pip install beautifulsoup4 ranx tqdm matplotlib
  ```

### Downloading the Code

Clone the repository or download the code files, and ensure all the necessary Python files are in your project directory.

## Data Files

Ensure you have the following data files available:

* Answers.json (90MB JSON file containing answers)
* topics_1.json and topics_2.json (files containing topics/questions)
* qrel_1.tsv (file containing qrel data for evaluation)

## Usage
### Running the Code

1. Preprocessing: Tokenize answers, create term indices, and vectorize the answers:
   ```python
    python main.py
    ```

2. Search: Perform the search for both TF-IDF and BM25:

    ```python
    python main.py --search
    ```

3. Overwrite Precomputed Files: If you want to overwrite the precomputed files (tokenized answers, term indices, and vectorized answers), run:
     
    ```python
    main.py --overwrite
    ```

## Command-line Arguments

* answers_file: Path to the JSON file containing answers (default: Answers.json).
* topics_1: Path to the first topic set (default: topics_1.json).
* topics_2: Path to the second topic set (default: topics_2.json).
* qrel: Path to the qrel file for evaluation (default: qrel_1.tsv).
* overwrite: Boolean flag to overwrite precomputed files (default: False).
* search: Boolean flag to perform search (default: True).

## Files and Data

* Answers.json: Contains the raw answers data.
* topics_1.json and topics_2.json: Contain the questions/topics to be searched.
* qrel_1.tsv: The qrel file containing relevance judgments for evaluation.
* result_tfidf_1.tsv, result_tfidf_2.tsv, result_bm25_1.tsv, result_bm25_2.tsv: Result files generated after running the searches with TF-IDF and BM25 models.

## File Outputs

* Tokenized Answers: A compressed JSON file mapping answer IDs to their tokenized lists (tokenized_answers.json.gz).
* Term Index: A compressed JSON file containing term frequencies, IDF, and BM25 IDF scores (term_index.json.gz).
* Vectorized Answers: A compressed JSON file with L2-normalized vectors for each answer based on TF-IDF and BM25 weights (vectorized_answers.json.gz).
* Search Results: TSV files in TREC format with top 100 document matches for each topic (result_tfidf_1.tsv, result_tfidf_2.tsv, result_bm25_1.tsv, result_bm25_2.tsv).

## Evaluation

The search results are evaluated using the following metrics:

* Precision@1
* Precision@5
* nDCG@5
* MRR (Mean Reciprocal Rank)
* MAP (Mean Average Precision)

### Example Evaluation:

The full_eval function runs the evaluation:

```python
full_eval('qrel_1.tsv', 'result_tfidf_1.tsv')
full_eval('qrel_1.tsv', 'result_bm25_1.tsv')
```

## Results and Visualization

The project generates a ski-jump plot to visualize the Precision@5 values across different topics:

```python
plot_skijump('qrel_1.tsv', 'result_tfidf_1.tsv', k=5)
plot_skijump('qrel_1.tsv', 'result_bm25_1.tsv', k=5)
```

The plot highlights which queries performed well and which did not, enabling analysis of model performance.

## Assumptions

* The BM25 parameters k=1.2k=1.2 and b=0.75b=0.75 were used.
* Default logarithmic base 2 was used for calculating IDF.
* The document collection contains only the tokenized answers from the provided JSON file.
* Relevance judgments in the qrel file use scores 1 and 2 for relevant documents.
