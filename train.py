from utils import make_tokenized_instances
from utils import create_embeddings_dic
from features import make_transformer_list
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score
from collections import Counter
import pandas as pd
from sklearn import metrics
from features import FeatExtractor


def classify(train_df, test_df, class_labels, feature_labels, target_col, classifier, embed=False,embeddings_dic="medical_w2v_dic", model_path="../models/", save_model=False):
    #train_df = pd.read_csv(train_csv)
    #test_df = pd.read_csv(test_csv)
    #f5 = feature_selection.RFE(estimator=LinearSVC1, n_features_to_select=500, step=1)
    embed_dic ={}

    train_instances = make_tokenized_instances(train_df, text_col="note_")
    test_instances =  make_tokenized_instances(test_df, text_col="note_bow")
    all_instances = train_instances+test_instances
    if embed:
        if embeddings_dic=="w2v_google":
            print("making google w2v dic")
            embed_dic = create_embeddings_dic(set(w for words in all_instances for w in words),GOOGLE_W2V_PATH , 300)
            print("done")
        elif embeddings_dic=="glove":
            print("making glove  dic")
            embed_dic = create_embeddings_dic(set(w for words in all_instances for w in words),GLOVE_6B_50D_PATH , 300)
            print("done")

        else:
            print("making med w2v dic")
            embed_dic = create_embeddings_dic(set(w for words in all_instances for w in words), MEDICAL_W2V, 300)
            print("done")

    #if not classify_unlabaled:
    y_tr = list(train_df[target_col])#check_class_labels(list(train_df[target_col]), class_labels)
    #else:
    #trained_model =  joblib.load(model_path)
    #print("y_tr ", y_tr)
    y_ts = list(test_df[target_col])#check_class_labels(list(test_df[target_col]), class_labels)
    #print("y_ts ", y_ts)
    print("making pipeline")
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

    #if not classify_unlabaled:
    #saved_model_name = model_path+now.strftime("%Y-%m-%d-%H:%M")+"-"+"-".join(class_labels)+"-"+classifier.__class__.__name__

    pipeline.fit(train_df, y_tr)
    #clf = joblib.load('filename.pkl')
    #if save_model:
    #    print("saving mode to "+saved_model_name)
    #    joblib.dump(pipeline, saved_model_name)

    y_pred = pipeline.predict(test_df)
    #print("y_pred", list(y_pred))
    print("done")
    #print(list(y_ts))
    report = metrics.classification_report(y_pred, y_ts)
    print("========================================");
    print("\n")
    print(confusion_matrix(y_ts, y_pred))
    print("\n")
    kappa = cohen_kappa_score(y_ts, y_pred)
    print(report)
    print("\nK: " + str(kappa) + "\n")
    #print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_ts, y_pred))) + "\n"
    print("========================================")

