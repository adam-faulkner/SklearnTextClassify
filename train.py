from utils import make_tokenized_instances
from utils import create_embeddings_dic
from features import make_transformer_list
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import cohen_kappa_score
from collections import Counter
import pandas as pd
from sklearn import metrics
from features import FeatExtractor
from sklearn.linear_model import LogisticRegression
from utils import make_num_feature_from_bow

GOOGLE_W2V = "./resources/GoogleNews-vectors-negative300.txt"
GLOVE = "./resources/glove.6B.300d.txt"


def classify(train_df, test_df, class_labels, feature_labels, target_col, classifier, embed=False,embeddings_dic="glove",
             model_path="../models/", save_model=False):
    embed_dic ={}
    train_instances = make_tokenized_instances(train_df, text_col="review_bow")
    test_instances =  make_tokenized_instances(test_df, text_col="review_bow")
    all_instances = train_instances+test_instances
    if embed:
        if embeddings_dic=="w2v_google":
            print("Making google w2v dic")
            embed_dic = create_embeddings_dic(set(w for words in all_instances for w in words),GOOGLE_W2V , 300)
            print("done")
        elif embeddings_dic=="glove":
            print("Making glove dic")
            embed_dic = create_embeddings_dic(set(w for words in all_instances for w in words),GLOVE , 300)
            print("done")
    y_tr = list(train_df[target_col])
    y_ts = list(test_df[target_col])
    print("Making pipeline")
    transformer_list = make_transformer_list(feature_labels, embed_dic)
    pipeline = Pipeline([
        # Extract the feature types
        ('features', FeatExtractor(feature_labels)),
        # Use FeatureUnion to combine the features
        ('union', FeatureUnion(
            # transformer list is provided by make_transformer_list()
            transformer_list
        )),
        ('classifier', classifier),
    ])
    pipeline.fit(train_df, y_tr)
    y_pred = pipeline.predict(test_df)
    print("done")
    report = metrics.classification_report(y_pred, y_ts)
    print("========================================");
    print("\n")
    print(confusion_matrix(y_ts, y_pred))
    print("\n")
    kappa = cohen_kappa_score(y_ts, y_pred)
    print(report)
    print("\nKappa: " + str(kappa) + "\n")
    print("Accuracy: {:0.3f}".format(metrics.accuracy_score(y_ts, y_pred))) + "\n"
    print("========================================")
