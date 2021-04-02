import os
import collections
from queue import PriorityQueue
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import sys
sys.stdout.reconfigure(encoding='utf-8')

dict_for_file = collections.defaultdict(int)  # frequency of individual alphanum terms in the target file
dict_file_occur = collections.defaultdict(int)


def stem_string(string):
    # divide the text into tokens
    text_tokens = word_tokenize(string)
    # convert every token into its stem
    stemmer = SnowballStemmer('english')
    for i in range(len(text_tokens)):
        text_tokens[i] = stemmer.stem(text_tokens[i])

    return text_tokens


def stem_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        whole_text = file.read()

    return stem_string(whole_text)


def analyze_target_text(path):
    text_stem = stem_file(path)
    for word in text_stem:
        if word.isalnum():
            dict_for_file[word] += 1


def list_corpus_texts(path, list_of_corpus):
    """
    Function for listing all the text files from the root folder
    """
    for file in os.listdir(path):
        if file.endswith('.txt'):
            list_of_corpus.append(os.path.join(path, file))
        else:
            list_corpus_texts(os.path.join(path, file), list_of_corpus)


def count_doc_frequency(path):
    """
    Finds the document frequency of the words from dict_from_file global variable
    """
    num_of_files = 0
    list_of_corpus = []  # contains paths to all txt files in the corpus
    list_corpus_texts(path, list_of_corpus)
    for txt_path in list_of_corpus:
        num_of_files += 1
        curr_file_occur = collections.defaultdict(lambda: False)
        text_stem = stem_file(txt_path)
        for term in text_stem:
            if dict_for_file[term] and not curr_file_occur[term]:
                curr_file_occur[term] = True
                dict_file_occur[term] += 1

    return num_of_files


dict_for_score = {}  # dictionary for keeping count of the score of each term


def calculate_word_score(corpus_path, file_path):
    """
    Calculates the TF_IDF score of every term inside the targeted text(file)
    """
    global dict_for_score
    analyze_target_text(file_path)
    n = count_doc_frequency(corpus_path)
    for key in dict_file_occur:
        dict_for_score[key] = dict_for_file[key] * np.log(n / dict_file_occur[key])

    dict_for_score = dict(sorted(dict_for_score.items(), key=lambda item: (-item[1], item[0])))


def print_top_terms():
    term_limit = 0
    length = len(dict_for_score)
    for k, v in dict_for_score.items():
        if k.isalnum():
            print(k, end="")
        term_limit += 1
        length -= 1
        if term_limit < 10 and length:
            print(", ", end="")
        else:
            print()
            break


def analyze_text_sentences(path):
    with open(path, 'r', encoding='utf-8') as file:
        whole_text = file.read()

    pq_s = PriorityQueue()  # queue for finding top 5 sentences
    list_of_sentences = sent_tokenize(whole_text)
    for idx, s in enumerate(list_of_sentences):
        pq_t = PriorityQueue()  # queue for finding top terms for one sentence
        curr_sentence_stems = stem_string(s)
        for stem in curr_sentence_stems:
            if stem.isalnum():
                pq_t.put((-dict_for_score[stem], stem))

        it, sum = 0, 0
        while it != 10 and not pq_t.empty():
            sum -= pq_t.get()[0]
            it += 1

        pq_s.put((-sum, (idx, s)))

    top_sentences, it = [], 0
    # take the top 5 sentences and put them in the result string
    while it != 5 and not pq_s.empty():
        top_sentences.append(pq_s.get()[1])
        it += 1

    top_sentences.sort()
    for curr in top_sentences:
        print(curr[1] + " ", end="")


def main():
    corpus_path = r"D:\PSIML_PRVI\public\corpus"
    file_path = r"D:\PSIML_PRVI\public\corpus\quantum\Quantum Leap.txt"
    calculate_word_score(corpus_path, file_path)
    print_top_terms()
    analyze_text_sentences(file_path)


if __name__ == "__main__":
    main()
