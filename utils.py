import re
from collections import defaultdict, Counter
import numpy as np
import spacy
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
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
     From https://github.com/nadbordrozd/blog_stuff/tree/master/classification_w2v
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
    :return: list of lists
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
    :return: boolean
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


def add_extras(y, cls_cols):
    for l in list(set(y)):
        if l not in cls_cols:
            cls_cols.append(l)
    return cls_cols


def one_hot_dataframe(data, cols, replace=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)


def dic_check(dic, ky):
    if dic.get(ky) is not None:
        return True
    return False


def make_num_feature_from_bow(df, col_label, categories, raw_freq=False):
    """
    Counts the occurence of a label in a seq and returns
    n new cols where n is the number of unique tags. Returns a modified
    version of the df with the n categories added
    :param df:
    :param col_label:
    :param categories:
    :param raw_freq:
    :return: pandas dataframe augmented with new columns
    """
    all_dicts = []
    the_col = list(df[col_label])

    if raw_freq:
        new_col = []
        for s in the_col:
            if s and (type(s) != float and type(s) != np.float64):
                    new_col.append(len(s.split()))
            else:
                new_col.append(0)
        df[col_label+"_count"] = new_col
        return df
    for idx, r in enumerate(the_col):
        if type(r) == float:
            r = "neutral"
       # print("doing ",r, " at ", idx)
        very_spt =r.split("very")
       # print("very splt", very_spt)
        cat_tags = []
        for item in very_spt:
            if item.startswith(" "):
                tag = "very " + item.split()[0]
                cat_tags.append(tag)
                for s in item.split()[1:]:
                    cat_tags.append(s)
            else:
                for s in item.split():
                    cat_tags.append(s)

        sent_tags = [st.strip() for st in cat_tags]
        #print("sent tags ", sent_tags)
        sent_cnt = dict(Counter(sent_tags))
        # for k, v in dict(cnt).items():
        # get sentiment polarity counts
        processed_feat_dict = {}

        for c in categories:
            processed_feat_dict[c] = 0
        #print("sent tags ", sent_tags)
        for tag in sent_tags:
            #print("doing tag ", tag)
            #if dic_check(sent_cnt, tag):
    #        if tag in categories:
             processed_feat_dict[tag+"_count"] = sent_cnt[tag]
        all_dicts.append(processed_feat_dict)
    #for each cat in categories make a new df column
    dd = defaultdict(list)
    for d in all_dicts:  # you can list as many input dicts as you want here
        for key, value in d.iteritems():
            dd[key].append(value)
    for cat in categories:
        new_col = []
        for d in all_dicts:
            new_col.append(d[cat])
        df[cat] = new_col
    return df

