import re
import numpy as np
import spacy

nlp = spacy.load('en')
stopwords = [s.strip() for s in open("./resources/stopwords_custom.txt").readlines()]


def clean_str(string):
    """
    From https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string:
    :return: string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    """
     Fro https://github.com/nadbordrozd/blog_stuff/tree/master/classification_w2v
     load embedding_vectors from word2vec
    
    :param vocabulary: 
    :param filename: 
    :param binary: 
    :return: 
    """
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors

def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    """
    load embedding_vectors from GLOVE-trained embeddings
    :param vocabulary:
    :param filename:
    :param vector_size:
    :return:
    """
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def make_tokenized_instances(ls, text_col=""):
    """
    This is mainly used to create the data used by the embeddings-dic code
    :param ls:
    :param text_col:
    :return:
    """
    instances =[]
    text_ls = ls
    for text in text_ls:
        spacy_doc = nlp(re.sub("[^a-zA-Z]+", " ", unicode(text)))
        # don't use NLTK or Spacy's stopwords
        tokens = [sent.string.strip().lower() for sent in spacy_doc if sent.string.strip().lower() not in stopwords]
        instances.append(tokens)
    return instances

def is_number(s):
    """

    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def create_embeddings_dic(all_words, embeddings_path, num_dimensions):
    """

    :param all_words:
    :param embeddings_path:
    :param num_dimensions:
    :return:
    """
    embeddings_dic = {}
    print("Building embeddings dic from " + embeddings_path)
    with open(embeddings_path, "rb") as infile:
        next(infile)  # skip the header line with dimensions/vocab info
        for line in infile:
            # check if multi-word
            parts = line.split()
            lexical_entry = parts[:-num_dimensions]
            if len(lexical_entry) > 1:
                pass
            else:
                word = parts[0]
                nums = map(float, parts[-num_dimensions:])
                if word in all_words:
                    embeddings_dic[word] = np.array(nums)
    return embeddings_dic


def check_embeddings(X, embeddings_dic):
    print("looking at embeddings for X")
    dim = len(embeddings_dic.itervalues().next())
    words = X
    print("row in X ", words)
    for w in words:
        print("looking at w ", w)
        if w in embeddings_dic:
            print("embedding ", embeddings_dic[w])
    print("mean --> ", np.mean([embeddings_dic[w] for w in words if w in embeddings_dic]
                             or [np.zeros(dim)], axis=0))

