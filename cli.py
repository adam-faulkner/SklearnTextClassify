import os.path
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from utils import make_num_feature_from_bow
from train import classify


all_features =["text_bow", "text_embeddings", "positive_num", "negative_num"]

classifiers = "naive_bayes_bernouli, naive_bayes_multinomial , random_forest, log_reg"

parser = argparse.ArgumentParser(description='CLI for SklearnTextClassify. -h for all available command line args')

parser.add_argument('-train_path', action='store', dest='train_path',
                    help='Path to the training data')

parser.add_argument('-test_path', action='store', dest='test_path',
                    help='Path to the test data')

parser.add_argument('-dev_path', action='store', dest='dev_path',
                    help='Path to the development set')

parser.add_argument('-feat_list', nargs='+', help='Provide a space-separated list of features.', dest="mult_feat_collection")

parser.add_argument('-class_col', action='store', dest='class_col', help="The column header of the class column; default is 'target'")

parser.add_argument('-labels', nargs='+', help='Provide a space-separated list of class labels. Default is "pos" and "neg" ',dest="label_collection")

parser.add_argument('-cl', '--classifier' ,action='store', dest='classifier',
                    help='The classifier to use. Supported classifiers: '+classifiers)
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

parser.add_argument('-available_features')

parser.add_argument('-save_model_path', action='store', dest='save_model_path',
                    help="Path to where you'd like the classification model saved")

parser.add_argument('-cross_valid', action="store_true", dest="cross_valid_bool", default=False, help="Boolean. Use 5 fold cross validation; default is False")

results = parser.parse_args()
print 'user provided train path =', results.train_path
print 'user provided test path =', results.test_path
print 'user provided dev set path =', results.dev_path
print 'user provided feature list = ', results.mult_feat_collection
print 'user provided classifier = ', results.classifier
print 'user provided labels = ', results.label_collection
print 'user provided class column = ', results.class_col
print 'user provided path to save the model to = ', results.save_model_path

classifier_dic= {"mlp":MLPClassifier(),"naive_bayes_bernouli":BernoulliNB() , "naive_bayes_multinomial": MultinomialNB() ,  "random_forest":RandomForestClassifier(class_weight="balanced"),
                 "log_reg_balanced": LogisticRegression(class_weight="balanced"), "log_reg": LogisticRegression()}

#default values for all args
classifier = classifier_dic["log_reg"]
panda_file_dev = pd.read_csv("./resources/review_polarity_dev.tsv", delimiter="\t", encoding = "latin1")
panda_file_train = pd.read_csv("./resources/review_polarity_train.tsv", delimiter="\t",encoding = "latin1")
panda_file_test = pd.read_csv("./resources/review_polarity_test.tsv", delimiter="\t",encoding = "latin1")

user_provided_dev = False

#get args from commandline (if any)
if (results.class_col is not None):
    class_col = results.class_col.strip()
else:
    class_col = "target"

if (results.train_path is not None):
    assert os.path.isfile(results.train_path)# Check parameter's validity
    panda_file_train =  pd.read_csv(results.train_path, delimiter="\t",encoding = "latin1")

if (results.dev_path is not None):
    assert os.path.isfile(results.train_path)# Check parameter's validity
    panda_file_dev = pd.read_csv(results.dev_path, delimiter="\t",encoding = "latin1")
    user_provided_dev = True


if (results.test_path is not None):
    assert os.path.isfile(results.train_path)# Check parameter's validity
    panda_file_test = pd.read_csv(results.test_path, delimiter="\t",encoding = "latin1")

if (results.mult_feat_collection is not None):
    features = results.mult_feat_collection
else:
    features = all_features

if (results.label_collection is not None):
    class_labels = results.label_collection
else:
    class_labels = ["pos", "neg"]

if (results.classifier is not None):
    classifier = classifier_dic[results.classifier]

if (results.save_model_path is not None):
    assert os.path.isfile(results.save_model_path)
    save_model_path = results.save_model_path
else:
    save_model_path = "../models/"


print "Using classifier ",classifier
print "Using features ", features
print "Using class labels ", class_labels

#process categorical and numeric features in the dataframe

def create_cat_features(df, categories):
    new_df = df
    for category in categories:
        new_df = make_num_feature_from_bow(df, category,[""], raw_freq=True)
    return new_df

num_features = [cat for cat in features if cat.endswith("num")]

if user_provided_dev: #the dev set serves as the test set
    new_test_df = create_cat_features(panda_file_dev, num_features)
    new_test_df = new_test_df.fillna("x")
else:
    new_train_df = create_cat_features(panda_file_train,num_features )
    new_test_df = create_cat_features(panda_file_test, num_features)
    new_train_df = new_train_df.fillna("x")
    new_test_df = new_test_df.fillna("x")


classify(new_train_df, new_test_df, class_labels, features, class_col, classifier, embed=True,   embeddings_dic="w2v_google", model_path ="../models/2018-01-04LogisticRegression", save_model=True)
