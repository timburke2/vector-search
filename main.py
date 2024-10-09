from bs4 import BeautifulSoup
from ranx import Qrels, Run, evaluate
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import re 
import json
import gzip
import math
import csv
import time
import argparse

def clean_text(text): 
    """
    Cleans and tokenizes the input text.

    This function removes HTML tags using BeautifulSoup, converts text to lowercase,
    splits it into tokens, removes punctuation from the start and end of tokens, and
    filters out tokens that are not alphabetic or are single characters (except 'a' and 'i').

    Parameters:
        text (str): The raw text string to be cleaned and tokenized.

    Returns:
         List of str: A list of cleaned and tokenized words.
    """

    soup = BeautifulSoup(text, "lxml")
    cleaned_text = soup.get_text(separator=" ")
    tokens = cleaned_text.lower().split()

    clean_tokens = []
    for token in tokens:
        # Remove punctuation at the start and end of the token
        stripped_token = re.sub(r"^[\"']+|[\"']+$", "", token)

        # Check if the token still contains alphabetic characters or an apostrophe
        if re.match(r"^[a-zA-Z']+$", stripped_token):
            # Filter out single characters unless they are "a" or "i"
            if len(stripped_token) > 1 or stripped_token in ('a', 'i'):
                clean_tokens.append(stripped_token)
            #TODO: More token filtering with regex?

    return clean_tokens

def tokenize_answers(answers_file='Answers.json', outfile='tokenized_answers.json.gz'):
    """
    Tokenizes and cleans the text of answers from a JSON file.

    This function reads a JSON file containing answers, cleans and tokenizes the text of each answer,
    and saves the result as a compressed JSON file mapping answer IDs to their token lists.

    Parameters:
        answers_file (str): The file path to the JSON file containing raw answers.
                            Defaults to 'Answers.json'.
        outfile (str): The desired file path for the compressed output file.
                       Defaults to 'tokenized_answers.json.gz'.

    Returns:
        str: The file path to the compressed tokenized answers file.
    """

    progress = 0
    with open(answers_file, 'r', encoding='utf-8') as infile:
        answers = json.load(infile)

        answer_data = {}
        # Iterate over each answer and create a dictonary entry for the cleaned text
        for answer in tqdm(answers, desc='Tokenizing documents'):
            id = answer['Id']
            text = clean_text(answer['Text'])

            answer_data[id] = text
            progress += 1
    
    with gzip.open(outfile, 'wt', encoding='utf-8') as gz_outfile:
        json.dump(answer_data, gz_outfile, ensure_ascii=False, indent=2)

    return outfile

def index_terms(tokenized_answers_file='tokenized_answers.json.gz', outfile='term_index.json.gz'):
    """
    Creates an inverted index with term frequencies and IDF scores.

    This function processes tokenized answers to build an inverted index that includes
    raw term frequencies, logarithmic TF for TF-IDF, and BM25 term frequencies.
    It also calculates IDF and BM25 IDF for each term.

    Parameters:
        tokenized_answers_file (str): The file path to the compressed JSON file containing
                                      tokenized answers. Defaults to 'tokenized_answers.json.gz'.
        outfile (str): The desired file path for the compressed term index output file.
                       Defaults to 'term_index.json.gz'.

    Returns:
        str: The file path to the compressed term index file.
    """

    # Load tokenized_answers from the compressed JSON file
    with gzip.open(tokenized_answers_file, 'rt', encoding='utf-8') as infile:
        tokenized_answers = json.load(infile)

    # Total number of documents in tokenized answers file
    num_docs = len(tokenized_answers)

    # Calculate the average document length
    total_length = sum(len(tokens) for tokens in tokenized_answers.values())
    avg_doc_length = total_length / num_docs

    term_data = {}

    print("Constructing Term Index")
    # Build the term index with term frequencies
    for doc_id, tokens in tqdm(tokenized_answers.items(), desc='Indexing documents'):
        for term in tokens:
            if term not in term_data:
                term_data[term] = {
                    'IDF': 0,
                    'BM25_IDF': 0,
                    'TF': {}
                }
            if doc_id not in term_data[term]['TF']:
                term_data[term]['TF'][doc_id] = {
                    'Raw': 0,
                    'Log': 0,
                    'BM25': 0
                }
            term_data[term]['TF'][doc_id]['Raw'] += 1

    print("Calculating IDF and BM25-IDF scores")
    # Compute IDF and BM25_IDF for each term
    for term in term_data:
        num_occurring_docs = len(term_data[term]['TF'])
        # Compute IDF using log base 2
        term_data[term]['IDF'] = math.log2(num_docs / num_occurring_docs)
        # Compute BM25 IDF
        term_data[term]['BM25_IDF'] = math.log(1 + ((num_docs - num_occurring_docs + 0.5) / (num_occurring_docs + 0.5)))

    # BM25 initial parameters
    k = 1.2
    b = 0.75

    print("Computing TF variants")
    # Compute TF variants for each term in each document
    for term in term_data:
        for doc_id in term_data[term]['TF']:
            tf_raw = term_data[term]['TF'][doc_id]['Raw']
            # Compute logarithmic term frequency
            term_data[term]['TF'][doc_id]['Log'] = 1 + math.log(tf_raw)
            # Document length for the current document
            doc_length = len(tokenized_answers[doc_id])
            # Compute BM25 term frequency component
            tf_denominator = tf_raw + k * (1 - b + b * (doc_length / avg_doc_length))
            term_data[term]['TF'][doc_id]['BM25'] = (tf_raw * (k + 1)) / tf_denominator

    print("Writing to JSON")
    with gzip.open(outfile, 'wt', encoding='utf-8') as gz_outfile:
        json.dump(term_data, gz_outfile, ensure_ascii=False, indent=2)

    return outfile

def get_data_from_gz(infile):
    """
    Loads data from a compressed JSON (.json.gz) file.

    Parameters:
        infile (str): The file path to the compressed JSON file.

    Returns:
        dict or list: The data object loaded from the JSON file.
    """

    print("Loading file to memory")
    with gzip.open(infile, 'rt', encoding='utf-8') as infile:
        return json.load(infile)
    
def get_data_from_json(infile):
    """
    Loads data from a JSON (.json) file.

    Parameters:
        infile (str): The file path to the JSON file.

    Returns:
        dict or list: The data object loaded from the JSON file.
    """

    with open(infile, 'r', encoding='utf-8') as infile:
        return json.load(infile)
    
def vectorize_answers(tokenized_answers, term_index, outfile='vectorized_answers.json.gz'):
    """
    Converts tokenized answers into normalized TF-IDF and BM25 vectors.

    For each tokenized answer, this function calculates the TF-IDF and BM25 weights
    for each term and normalizes the vectors using L2 normalization. The result is
    saved as a compressed JSON file mapping answer IDs to their term vectors.

    Parameters:
        tokenized_answers (dict): A dictionary mapping answer IDs to lists of tokens.
        term_index (dict): The term index containing TF and IDF information.
        outfile (str): The desired file path for the compressed output file.
                       Defaults to 'vectorized_answers.json.gz'.

    Returns:
        str: The file path to the compressed vectorized answers file.
    """

    with gzip.open(outfile, 'wt', encoding='utf-8') as gz_outfile:
        all_vectors = {}

        for answer_id, tokens in tqdm(tokenized_answers.items(), desc='Vectorizing answers'):
            vector = {}

            # To be used during vector normalization
            tfidf_magnitude = 0
            bm25_magnitude = 0

            for term in tokens:

                # Calculates TF-IDF score using the precomputed values in the term index
                tf_idf = term_index[term]['TF'][answer_id]['Log'] * term_index[term]['IDF']
                # Cumulatively increase magnitude by the square of the components score
                tfidf_magnitude += (tf_idf * tf_idf)

                # Calculates BM25 score using the precomputed values in the term index
                bm25 = term_index[term]['BM25_IDF'] * term_index[term]['TF'][answer_id]['BM25']
                # Cumulatively increase magnitude by the square of the components score
                bm25_magnitude += (bm25 * bm25)

                # Appends pair of scores to the component in the vector
                vector[term] = (tf_idf, bm25)

            # Calculate norm by finding the square root of the sum of component squares
            l2_norm_tfidf = math.sqrt(tfidf_magnitude)
            l2_norm_bm25 = math.sqrt(bm25_magnitude)

            # Iterate over and normalize all component scores
            for term in vector:
                tfidf_norm = vector[term][0] / l2_norm_tfidf
                bm25_norm = vector[term][1] / l2_norm_bm25
                # Rounding to save on space, should not affect rankings
                vector[term] = (round(tfidf_norm, 3), round(bm25_norm, 3))

            # Append normalized vector to the full list of vectors
            all_vectors[answer_id] = vector

        # Write full list of vectors to the output file
        json.dump(all_vectors, gz_outfile, ensure_ascii=False, indent=2)

    return outfile

def vectorize_topics_tfidf(query_text, term_index):
    """
    Converts a query text into a normalized TF-IDF vector.

    This function calculates the TF-IDF weights for the terms in the query text
    using logarithmic term frequency and the IDF values from the term index.
    The vector is then L2-normalized.

    Parameters:
        query_text (list of str): A list of cleaned and tokenized query terms.
        term_index (dict): The term index containing IDF information.

    Returns:
        dict: A dictionary mapping terms to their normalized TF-IDF weights.
    """

    vector = {}
    magnitude = 0
    # Iterate over each token in the query
    for term in query_text:
        # Only score terms which exist in the index
        if term in term_index:
            # Logarithmic term frequency for reducing impact of document length
            tf = 1 + math.log(query_text.count(term))
            # IDF from precomputed data
            tf_idf = tf * term_index[term]['IDF']
            #Tracks magnitude as the sum of component squares
            magnitude += (tf_idf * tf_idf)

            vector[term] = tf_idf
        else:
            continue
    
    # Calculates normalizing term as the square root of the sum of component squares
    l2_norm = math.sqrt(magnitude)

    # Normalizes each term in the vector
    for term in vector:
        norm_term = vector[term] / l2_norm
        vector[term] = norm_term
    
    return vector

def vectorize_topics_bm25(query_text, term_index, avg_doc_length):
    """
    Converts a query text into a normalized BM25 vector.

    This function calculates the BM25 weights for the terms in the query text
    using raw term frequencies and BM25 parameters. The vector is then L2-normalized.

    Parameters:
        query_text (list of str): A list of cleaned and tokenized query terms.
        term_index (dict): The term index containing BM25 IDF information.
        avg_doc_length (float): The average document length across all documents.

    Returns:
    dict: A dictionary mapping terms to their normalized BM25 weights.
    """

    # BM25 term frequency default parameters
    k = 1.2
    b = 0.75
    doc_length = len(query_text)
    vector = {}
    magnitude = 0

    # Iterate over each token in the query
    for term in query_text:
        # Only score terms which exist in the index
        if term in term_index:
            tf_raw = query_text.count(term)
            # Expects average doc length to be passed to function to operate
            tf_denominator = tf_raw + k * (1 - b + b * (doc_length / avg_doc_length))
            bm25_tf = (tf_raw * (k + 1)) / tf_denominator
            # BM25 IDF from precomputed term index
            bm25 = bm25_tf * term_index[term]['BM25_IDF']
            # Tracks magnitude for norm
            magnitude += (bm25 * bm25)

            vector[term] = bm25
        else:
            continue
    
    # Square root of sum of component squares
    l2_norm = math.sqrt(magnitude)

    # Normalizes each term in the vector
    for term in vector:
        norm_term = vector[term] / l2_norm
        vector[term] = norm_term
    
    return vector

def find_query_terms_tfidf(query_text, term_index, num_terms=10):
    """
    Selects the top query terms based on TF-IDF weights.

    This function calculates the TF-IDF weights of the terms in the query text
    and returns the top `num_terms` terms with the highest weights.

    Parameters:
        query_text (list of str): A list of cleaned and tokenized query terms.
        term_index (dict): The term index containing IDF information.
        num_terms (int): The number of top terms to select. Defaults to 10.

    Returns:
        list of tuples: A list of tuples where each tuple contains a term and its TF-IDF weight,
                        sorted in descending order by weight.
    """

    # Defaults to using all terms if num_terms is out of range
    if num_terms == 0 or num_terms > len(query_text):
        num_terms = len(query_text)


    query_terms = {}
    # Calculates TF-IDF scores for all terms in query
    for term in query_text:
        if term in term_index:
            tf = 1 + math.log(query_text.count(term))
            idf = term_index[term]['IDF']
            tf_idf = tf * idf

            query_terms[term] = tf_idf

    # Returns descented sorted list of terms, num_terms in length
    return sorted(query_terms.items(), key=lambda x : x[1], reverse=True)[:num_terms]

def find_query_terms_bm25(query_text, term_index, avg_doc_length, num_terms=10):
    """
    Selects the top query terms based on BM25 weights.

    This function calculates the BM25 weights of the terms in the query text
    and returns the top `num_terms` terms with the highest weights.

    Parameters:
        query_text (list of str): A list of cleaned and tokenized query terms.
        term_index (dict): The term index containing BM25 IDF information.
        avg_doc_length (float): The average document length across all documents.
        num_terms (int): The number of top terms to select. Defaults to 10.

    Returns:
        list of tuples: A list of tuples where each tuple contains a term and its BM25 weight,
                        sorted in descending order by weight.
    """

    # Defaults to using full query text if num_terms is out of range
    if num_terms == 0 or num_terms > len(query_text):
        num_terms = len(query_text)

    # BM25 term frequency default parameters
    k = 1.2
    b = 0.75
    doc_length = len(query_text)

    query_terms = {}
    # Calculates BM25 scores for all terms in query
    for term in query_text:
        # Only terms which are in the index
        if term in term_index:
            tf_raw = query_text.count(term)
            # Expects average doc length to be passed to function to operate
            tf_denominator = tf_raw + k * (1 - b + b * (doc_length / avg_doc_length))
            bm25_tf = (tf_raw * (k + 1)) / tf_denominator
            # BM25 IDF from precomputed term index
            bm25 = bm25_tf * term_index[term]['BM25_IDF']

            query_terms[term] = bm25

    # Returns descented sorted list of terms, num_terms in length
    return sorted(query_terms.items(), key=lambda x : x[1], reverse=True)[:num_terms]

def dot_product(query_vector, answer_vector, isBM25):
    """
    Calculates the dot product between a query vector and an answer vector.

    Depending on the `isBM25` flag, the function uses either the TF-IDF or BM25
    component of the answer vector for the calculation.

    Parameters:
        query_vector (dict): A dictionary mapping terms to their normalized weights in the query.
        answer_vector (dict): A dictionary mapping terms to tuples of (TF-IDF weight, BM25 weight).
        isBM25 (bool): A flag indicating whether to use BM25 weights (True) or TF-IDF weights (False).

    Returns:
        float: The similarity score between the query and the answer.
    """

    score = 0
    # A dot B = sum(ai * bi)
    # Answer vectors have tuples for weights, [0] for tf_idf and [1] for BM25
    for term in query_vector:
        if term in answer_vector.keys():
            score += query_vector[term] * answer_vector[term][isBM25]
    return score

def full_search(topic_sets, tokenized_answers, term_index, vectorized_answers, isbm25):
    """
    Performs a full search over multiple topic sets using either TF-IDF or BM25 weighting.

    This function processes each topic in the provided topic sets, vectorizes the queries,
    selects candidate documents, calculates similarity scores, and generates result binaries
    in TREC format.

    Parameters:
        topic_sets (tuple of str): A tuple containing file paths to topic set JSON files.
        tokenized_answers (dict): A dictionary mapping answer IDs to lists of tokens.
        term_index (dict): The term index containing TF and IDF information.
        vectorized_answers (dict): A dictionary mapping answer IDs to their term vectors.
        isbm25 (bool): A flag indicating whether to use BM25 weighting (True) or TF-IDF (False).

    Returns:
        list of str: A list of file paths to the result binary files generated for each topic set.
    """
    
    # Calculates average doc length for BM25 calculations
    total_length = sum(len(tokens) for tokens in tokenized_answers.values())
    num_docs = len(tokenized_answers)
    avg_doc_length = total_length / num_docs

    result_binaries =  [] # Stores result binary file names
    # Two topic sets, so the main search loop repeats
    set_num = 0
    for topic_set in topic_sets:
        all_results  = {}
        topics = get_data_from_json(topic_set)

        for topic in tqdm(topics, desc=f'Searching over topic set {set_num+1}'):
            # Using all text from each topic
            query = f"{topic['Title']} {topic['Body']} {topic['Tags']}"
            query_text = clean_text(query)

            if isbm25:
                query_terms = find_query_terms_bm25(query_text, term_index, avg_doc_length)
                query_vector = vectorize_topics_bm25(query_text, term_index, avg_doc_length)
                result_tag = 'bm25' # Used to specify result binary name
            else:
                query_terms = find_query_terms_tfidf(query_text, term_index)
                query_vector = vectorize_topics_tfidf(query_text, term_index)
                result_tag = 'tfidf' # Used to specify result binary name
            
            # Collects a set of documents which have at least one of the high information terms
            candidate_docs = set()
            for term, _ in query_terms:
                candidate_docs.update(term_index[term]['TF'].keys())

            # Finds vector similarity between the query vector and each candidate document
            results = {}
            for doc in candidate_docs:
                results[doc] = dot_product(query_vector, vectorized_answers[doc], isbm25)

            # Top 100 document matches, decending in rank by score
            all_results[topic['Id']] = sorted(results.items(), key=lambda x: x[1], reverse=True)[:100]
        
        result_binary_name = f'result_{result_tag}_{set_num+1}.tsv'
        result_binary = create_result_binary(all_results, result_binary_name, f'timsearch_{result_tag}')
        result_binaries.append(result_binary)
        set_num += 1 # Iterate counter for topic set
    
    return result_binaries

def create_result_binary(final_results, outfile, run_name):
    """
    Writes search results to a result binary file in TREC format.

    Parameters:
        final_results (dict): A dictionary mapping topic IDs to lists of (document ID, score) tuples,
                              sorted in descending order by score.
        outfile (str): The file path for the output result binary file.
        run_name (str): The name of the run/system performing the search.

    Returns:
        str: The file path to the generated result binary file.
    """

    with open(outfile, mode='w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        for topic, result in final_results.items():
            rank = 0
            for doc in result:
                rank += 1
                tsv_writer.writerow([topic, "Q0", doc[0], rank, round(doc[1], 4), run_name])
        
        file.flush()
        file.close()

    return outfile

def full_eval(qrel_file, result_binary, benchmark_tag):
    """
    Evaluates a search run against the qrel using standard metrics.

    This function calculates metrics such as precision@1, precision@5, NDCG@5,
    MRR, and MAP, and displays them in a table, saved in .png format

    Parameters:
        qrel_file (str): The file path to the qrel file in TREC format.
        result_binary (str): The file path to the result binary file generated by the search.
        benchmark_tag (str): Benchmark type for output file name

    Returns:
        None
    """

    qrels = Qrels.from_file(qrel_file, kind="trec")
    run = Run.from_file(result_binary, kind="trec")

    results = evaluate(qrels, run, ["precision@1", "precision@5", "ndcg@5", "mrr", "map"])
    
    eval_data = [
        ["precision@1", round(float(results["precision@1"]), 5)],
        ["precision@5", round(float(results["precision@5"]), 5)],
        ["ndcg@5", round(float(results["ndcg@5"]), 5)],
        ["mrr", round(float(results["mrr"]),5)],
        ["map", round(float(results["map"]), 5)]
    ]

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=eval_data, loc='center')

    plt.savefig(f'{benchmark_tag}_eval.png', bbox_inches='tight')
    plt.close()

def plot_skijump(qrel_file, result_binary, benchmark_tag, k=5):
    """
    Plots a ski jump graph of precision@k for each query in the search run.

    This function evaluates precision@k for each query, sorts the queries by their scores,
    and generates a bar plot to visualize the distribution. Saves to .png format

    Parameters:
        qrel_file (str): The file path to the qrel file in TREC format.
        result_binary (str): The file path to the result binary file generated by the search.
        benchmark_tag (str): Benchmark type for output file name
        k (int): The cutoff rank for precision calculation. Defaults to 5.

    Returns:
        None
    """

    qrels = Qrels.from_file(qrel_file, kind="trec")
    run = Run.from_file(result_binary, kind="trec")

    p5 = []
    # Finds p@k for each query in the run
    # There is probably a better way to do this
    for query_id in qrels.qrels.keys():
        single_qrels = Qrels({query_id: qrels.qrels[query_id]})
        single_run = Run({query_id: run.run[query_id]})
        
        result = evaluate(single_qrels, single_run, f"precision@{k}")
        p5.append((query_id, result))

    # Sorts and organizes data
    sorted_precision = sorted(p5, key=lambda x : x[1], reverse=True)
    ranks = [id for id, _, in sorted_precision]
    scores = [score for _, score in sorted_precision]

    # Plot the ski jump slope
    plt.bar(ranks, scores)

    plt.xticks([])

    # Add labels and title
    plt.xlabel('Rank')
    plt.ylabel(f'P@{k}')
    plt.title(f'Ski Jump Plot of P@{k}')

    plt.grid(False)
    plt.savefig(f'{benchmark_tag}_ski_jump_plot.png', bbox_inches='tight')
    plt.close()

def precompute(answers_file='Answers.json'):
    """
    Precomputes necessary data files for the search system.

    This function tokenizes the answers, creates the term index,
    and vectorizes the answers. It is intended to be run once
    before performing searches.

    Parameters:
        answers_file (str): The file path to the raw answers JSON file.
                            Defaults to 'Answers.json'.

    Returns:
        tuple of str:
            - tokenized_answers (str): File path to the tokenized answers file.
            - term_index (str): File path to the term index file.
            - vectorized_answers (str): File path to the vectorized answers file.
    """

    tokenized_answers = tokenize_answers(answers_file)
    term_index = get_data_from_gz(index_terms(tokenized_answers))
    tokenized_answers = get_data_from_gz(tokenized_answers)
    vectorized_answers = get_data_from_gz(vectorize_answers(tokenized_answers, term_index))

    return (tokenized_answers, term_index, vectorized_answers)

def main(answers_file='Answers.json', topics_1 ='topics_1.json', topics_2 ='topics_2.json'):
    """
    Main function to execute the entire search and evaluation pipeline.

    This function checks for necessary precomputed files and generates them if needed.
    It then performs searches using both TF-IDF and BM25 weighting schemes, evaluates
    the results, and generates evaluation plots.

    Parameters:
        answers_file (str): The file path to the raw answers JSON file. Defaults to 'Answers.json'.
        topics_1 (str): The file path to the first set of topics. Defaults to 'topics_1.json'.
        topics_2 (str): The file path to the second set of topics. Defaults to 'topics_2.json'.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Run the IR system with TF-IDF or BM25")
    parser.add_argument('--overwrite', action='store_true', default=False, 
                        help="Set to overwrite precomputed files (default: False)")
    parser.add_argument('--search', action='store_true', default=True, 
                        help="Enable search process (default: True)")
    parser.add_argument('--no-search', dest='search', action='store_false', 
                        help="Disable search process")
    parser.add_argument('--evaluate', action='store_true', default=True, 
                        help="Enable evaluation process (default: True)")
    parser.add_argument('--no-evaluate', dest='evaluate', action='store_false', 
                        help="Disable evaluation process")
    args = parser.parse_args()
    

    # Verify whether or not the files exist
    check_files = [Path('tokenized_answers.json.gz').exists(), 
                   Path('term_index.json.gz').exists(),
                   Path('vectorized_answers.json.gz').exists()]
    
    # If any of them don't, or if overwrite is specified, recreate them and load data
    if args.overwrite or not all(check_files):
        print("Creating files")
        tokenized_answers, term_index, vectorized_answers = precompute(answers_file)
    else: # Specify file paths for preexisting files and load data
        print('Files already found')
        tokenized_answers = get_data_from_gz('tokenized_answers.json.gz')
        term_index = get_data_from_gz('term_index.json.gz')
        vectorized_answers = get_data_from_gz('vectorized_answers.json.gz')


    if args.search:
        topics = (topics_1, topics_2)

        # Full search is both topic sets
        tfidf_binaries = full_search(topics, tokenized_answers, term_index, vectorized_answers, isbm25=False)
        bm25_binaries = full_search(topics, tokenized_answers, term_index, vectorized_answers, isbm25=True)
    else:
        # Assumes result binaries exist, sets them to preexisting files
        tfidf_binaries = ('result_tfidf_1.tsv', 'result_tfidf_2.tsv')
        bm25_binaries = ('result_bm25_1.tsv', 'result_bm25_2.tsv')

    if args.evaluate:
        # Checks if qrel is the default, otherwise assume it is qrel 2
        if Path('qrel_2.tsv').exists():
            qrel = 'qrel_2.tsv'
            topic_flag = 1
        else:
            qrel = 'qrel_1.tsv'
            topic_flag = 0
        # Creates table of benchmarks for each binary
        full_eval(qrel, tfidf_binaries[topic_flag], 'TF-IDF')
        full_eval(qrel, bm25_binaries[topic_flag], 'BM25')

        # Creates ski jump for p@5 for all queries
        plot_skijump(qrel, tfidf_binaries[topic_flag], 'TF-IDF')
        plot_skijump(qrel, bm25_binaries[topic_flag], 'BM25')
    else:
        print("Evaluation skipped")


if __name__ == '__main__':
    main()