import os.path
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


all_features ='semantic_types_bow , concept_cuis_bow , preferred_names_bow , negated_comply_words_bow_count ,' \
          'positive_count ,' \
          ' negative_count , pos_bow , comply_words_bow_count , neutral_count , deps_gov_gen_bow ,' \
          ' note_bow , deps_dep_gen_bow , very_positive_count , variant_list_bow , very_negative_count ,' \
          ''
classifiers = "naive_bayes_bernouli, naive_bayes_multinomial , random_forest, log_reg"

parser = argparse.ArgumentParser(description='CLI for engagement classification. -h for all available command line args')

parser.add_argument('-train_path', action='store', dest='train_path',
                    help='Path to the training data')

parser.add_argument('-test_path', action='store', dest='test_path',
                    help='Path to the test data')

parser.add_argument('-dev_path', action='store', dest='dev_path',
                    help='Path to the development set')

parser.add_argument('-feat_list', nargs='+', help='Provide a space-separated list of features. Default is all. Supported features: '+all_features, dest="mult_feat_collection")

parser.add_argument('-class_col', action='store', dest='class_col', help="The column header of the class column; default is 'goal_status_bow'")

parser.add_argument('-labels', nargs='+', help='Provide a space-separated list of class labels. Default is "Met" and "Other" ',dest="label_collection")

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
panda_file_train = "./resources/train.tsv"
panda_file_dev = "./resourecs/dev.tsv"
panda_file_test = "./resources/test.tsv"


#get args from commandline (if any)
if (results.class_col is not None):
    class_col = results.class_col.strip()
else:
    class_col = "target"

if (results.train_path is not None):
    assert os.path.isfile(results.train_path)# Check parameters validity
    panda_file_train = results.train_path

if (results.mult_feat_collection is not None):
    features = results.mult_feat_collection
else:
    features = [f.strip() for f in all_features.split(",") if f]

if (results.label_collection is not None):
    class_labels = results.label_collection
else:
    class_labels = ["Met", "Other"]

if (results.classifier is not None):
    classifier = classifier_dic[results.classifier]

if (results.save_model_path is not None):
    assert os.path.isfile(results.save_model_path)
    save_model_path = results.save_model_path
else:
    save_model_path = "../models/"


class_labels = ["engagement", "lack_of_engagement", "cm_advice", "other"]
print "Using classifier ", classifier
print "Using features ", features
print "Using class labels ", class_labels

def add_extras(y, cls_cols):
    for l in list(set(y)):
        if l not in cls_cols:
            cls_cols.append(l)
    return cls_cols



print "getting dev"
dev_df = pd.read_csv("../data/engage_dev.tsv", delimiter="\t", encoding = "latin1")
print list(dev_df.columns.values)
print "getting train"
train_df = pd.read_csv("../data/engage_train.tsv", delimiter="\t",encoding = "latin1")
#test_df = pd.read_csv("../data/engage_test.tsv", delimiter="\t",encoding = "latin1")
print "train has columns ", train_df.columns.values

#dev_df = pd.read_csv("../data/goal_attain_dev_refiltered.csv", encoding = "latin1")
#print list(dev_df.columns.values)
#train_df = pd.read_csv("../data/goal_attain_train_filtered.tsv",delimiter="\t",encoding = "latin1")
#test_df = pd.read_csv("../data/engage_test.tsv", delimiter="\t",encoding = "latin1")


#engagement_untagged ="/Users/afaulkner/Documents/engagement/engagement_git/engagement/data/goal_intrv_notes_range_nov01oh_v4_sentences_just_features_11_09.csv"
#engagement_untagged_test = "/Users/afaulkner/Documents/engagement/engagement_git/engagement/data/test_sentences.tsv"
#emrs =  "/Users/afaulkner/Documents/engagement/engagement_git/engagement/data/emr_notes_hpi_ap_xsgSentences_all.tsv"


print "getting test"
engagement_untagged_test = "/Users/afaulkner/Documents/engagement/engagement_git/engagement/data/goal_intrv_notes_range_dec02_oh_v4_all_features.tsv"
test_df = pd.read_csv(engagement_untagged_test,delimiter="\t",encoding = "latin1")
print "got test df with col values ", test_df.columns.values


#process categorical and numeric features in the dataframe

def create_cat_features(df):
    new_df = make_num_feature_from_bow(df, "sentiment", ["positive_count", "negative_count", "very_positive_count", "very_negative_count", "neutral_count"])
    new_df = make_num_feature_from_bow(new_df, "comply_words_bow", [""], raw_freq=True)
    new_df = make_num_feature_from_bow(new_df, "negated_comply_words_bow", [""], raw_freq=True)
    #cols_to_transform = ["g_type"]
    #df_with_dummies = pd.get_dummies(newer_df , prefix=['dummy_'], columns=cols_to_transform )
    return new_df


new_dev_df = create_cat_features(dev_df)
new_train_df = create_cat_features(train_df)
new_test_df = create_cat_features(test_df)
#engagement_untagged_df = pd.read_csv(engagement_untagged)#, delimiter="\t",encoding = "latin1",  error_bad_lines=False).fillna("x")
new_dev_df = new_dev_df.fillna("x")
new_train_df = new_train_df.fillna("x")
new_test_df = new_test_df.fillna("x")
#new_test_df = engagement_untagged_df.fillna("x")#new_test_df.fillna("x")
print("Available labels: ", new_train_df.columns.values)
#new_test_df = new_test_df.fillna("x")
#print(list(new_dev_df.columns.values))
#print(list(new_train_df.columns.values))
#print(list(new_test_df.columns.values))
#print(list(df_with_dummies.columns.values))

#class_labels = ['Met','Other']
#class_col = "g_status"
#features = ["note_bow"]
#classifier = LogisticRegression()
#classifier = RandomForestClassifier(class_weight="balanced")
#feature_gen.classify(new_train_df, new_test_df, class_labels, features, class_col, classifier, embed=False,
 #                    embeddings_dic="w2v_google", model_path ="../models/2017-11-05-16:57-engagement-lack_of_engagement-cm_advice-other-LogisticRegression", save_model=False)




print "passing these features to classify ", features
print "here are the cols for train ", new_train_df.columns.values
print "here are the cols for test ", new_test_df.columns.values
features = ['deps_dep_gen_bow','pos_bow', 'comply_words_bow','semantic_types_bow', 'preferred_names_bow' ,
            'variants_bow', 'deps_gov_gen_bow' ,'note_bow','concept_cuis_bow' ,'positive_count' ,'negative_count',
            'very_positive_count', 'very_negative_count' ,'neutral_count','comply_words_bow_count', 'negated_comply_words_bow_count']
