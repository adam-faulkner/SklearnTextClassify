
import numpy as np
from collections import defaultdict
from utils import make_tokenized_instances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    From http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html

    For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class NumExtractor(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, nums):
        return [{self.key: num}
                for num in nums]

class FeatExtractor(BaseEstimator, TransformerMixin):
    """Creates a list of features for a Pipeline."""
    def __init__(self, labels):
        self.labels = labels

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        dtype_ls =[(label, object) for label in self.labels]
        features = np.recarray(shape=(len(posts),),
                               dtype=dtype_ls)
        for l in self.labels:
            for i, col in posts.iterrows():
                if l.endswith("bow"): #don't break if there's no text
                    if not col[l]:
                        col[l] = "x"
                features[l][i] = col[l]
        return features

class MeanEmbeddingVectorizer(object):
    """ From https://github.com/nadbordrozd/blog_stuff/tree/master/classification_w2v
        Creates a feature based on the mean of the embeddings of all words
        in a string"""
    def __init__(self, embeddings_dic):
        self.embeddings_dic = embeddings_dic
        self.dim = len(embeddings_dic.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        for each embedding for each word in an instance, load into a vector (w_em_0...w_emN);
        then, take the mean of that vector of embeddings.If w is not in the embeddings_dic,
        then use a vector of zeros
        """

        X = make_tokenized_instances(list(X))
        vec =  np.array([
            np.mean([self.embeddings_dic[w] for w in words if w in self.embeddings_dic]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return vec

class TfidfEmbeddingVectorizer(object):
    """
    From https://github.com/nadbordrozd/blog_stuff/tree/master/classification_w2v
    """
    def __init__(self, embeddings_dic):
        self.embeddings_dic = embeddings_dic
        self.word2weight = None
        self.dim = len(embeddings_dic.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # default idf for unknown words is the max of known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        X = [x.split() for x in X]
        return np.array([
            np.mean([self.embeddings_dic[w] * self.word2weight[w]
                     for w in words if w in self.embeddings_dic] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def make_transformer_list(labels, embeddings_dic):
    """
    :param labels:
    :param embeddings_dic:
    :return: a list of tuples of the form (label, Pipeline object)
    """
    transformer_list =[]
    for lab in labels:
        if lab.endswith("embeddings"):
            print("embeddings for  ", lab)
            embed_transformer = (lab, Pipeline([
            (lab + "_+selector", ItemSelector(key=lab)),
            ("mean_embeddings", MeanEmbeddingVectorizer(embeddings_dic))]))
            transformer_list.append(embed_transformer)
        elif lab.endswith("bow"):
            print("tfidf for ", lab)
            bow_transformer= (lab, Pipeline([
                (lab+"_+selector", ItemSelector(key=lab)),
                ("bow_vectorizer", TfidfVectorizer(min_df=50))]))
            transformer_list.append(bow_transformer)

        elif lab.endswith("num"):
             num_transformer = (lab, Pipeline([
                 (lab + "_+selector", ItemSelector(key=lab)),
                 ('num_vectorizer', NumExtractor(lab)),  # returns a list of dicts
                 ('dict_vectorizer', DictVectorizer()),  # list of dicts -> feature matrix
             ]))
             transformer_list.append(num_transformer)
        else:
            raise ValueError("Feature type not recognized. All feature headers must end with 'bow', 'embeddings', or 'num'")
    return transformer_list
