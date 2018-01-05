# SklearnTextClassify


This project implements basic text classification functionality using Sklearn. It will be of interest to anyone who'd like to  quickly run text classification experiments with a minimum of fuss and wrangling with the Sklearn API. Sklearn tends to make common feature generation functionality such as concatenation of different feature types (e.g., categorical and numeric types) an oddly difficult process. Several Stackoverflow exchanges testify to this.  Here, I've collected various solutions to creating heterogenous feature types in Sklearn along with more contemporary types such as embeddings-based features. 


## Example usage

The code determines feature types based on the data headers of your .tsv-formatted data. Simply add any of the 
following to end of the header 
  * $feature$bow : Standard tf-idf weighted bag-of-words vectorization
  * $feature$embeddings : For each item (usually a word) in a space separated string, look up its embedding in an
  accompanying embeddings file and average the combined embeddings of the string.
  * $feature$num : For each item in a space separated string, count the item, and use the resulting count as a feature
  
  
Here is a (truncated) example from the review polarity dataset provided in the resources folder:
doc_id	text_bow	text_embeddings	positive_num	negative_num	target
| doc_id        | text_bow      | text_embeddings  | positive_num     | negative_num     |
| ------------- |---------------| -----------------| -----------------| -----------------|
| cv700_21947.txt | latest bond film [..] | positive_word positive_word | negative_word negative_word negative_word [...]|





```
$python ./cli.py -h

    Warning: no model found for 'en'

    Only loading the 'en' tokenizer.

usage: cli.py [-h] [-train_path TRAIN_PATH] [-test_path TEST_PATH]
              [-dev_path DEV_PATH]
              [-feat_list MULT_FEAT_COLLECTION [MULT_FEAT_COLLECTION ...]]
              [-class_col CLASS_COL]
              [-labels LABEL_COLLECTION [LABEL_COLLECTION ...]]
              [-cl CLASSIFIER] [--version]
              [-available_features AVAILABLE_FEATURES]
              [-save_model_path SAVE_MODEL_PATH] [-cross_valid]

CLI for SklearnTextClassify. -h for all available command line args

optional arguments:
  -h, --help            show this help message and exit
  -train_path TRAIN_PATH
                        Path to the training data
  -test_path TEST_PATH  Path to the test data
  -dev_path DEV_PATH    Path to the development set
  -feat_list MULT_FEAT_COLLECTION [MULT_FEAT_COLLECTION ...]
                        Provide a space-separated list of features.
  -class_col CLASS_COL  The column header of the class column; default is
                        'target'
  -labels LABEL_COLLECTION [LABEL_COLLECTION ...]
                        Provide a space-separated list of class labels.
                        Default is "pos" and "neg"
  -cl CLASSIFIER, --classifier CLASSIFIER
                        The classifier to use. Supported classifiers:
                        naive_bayes_bernouli, naive_bayes_multinomial ,
                        random_forest, log_reg
  --version             show program's version number and exit
  -available_features AVAILABLE_FEATURES
  -save_model_path SAVE_MODEL_PATH
                        Path to where you'd like the classification model
                        saved
  -cross_valid          Boolean. Use 5 fold cross validation; default is False
  ```
  
  ```
$python ./cli.py -train_path ./resources/review_polarity_train.tsv -test_path ./resources/review_polarity_test.tsv -feat_list text_bow text_embeddings positive_num negative_num -class_col target -labels pos neg -cl log_reg

    Warning: no model found for 'en'

    Only loading the 'en' tokenizer.

user provided train path = ./resources/review_polarity_train.tsv
user provided test path = ./resources/review_polarity_test.tsv
user provided dev set path = None
user provided feature list =  ['text_bow', 'text_embeddings', 'positive_num', 'negative_num']
user provided classifier =  log_reg
user provided labels =  ['pos', 'neg']
user provided class column =  target
user provided path to save the model to =  None
Using classifier  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
Using features  ['text_bow', 'text_embeddings', 'positive_num', 'negative_num']
Using class labels  ['pos', 'neg']
Making google w2v dic
Building embeddings dic from ./resources/GoogleNews-vectors-negative300.txt
done
Making pipeline
('tfidf for ', 'text_bow')
('embeddings for  ', 'text_embeddings')
done
========================================


[[162  38]
 [ 41 159]]


             precision    recall  f1-score   support

        neg       0.81      0.80      0.80       203
        pos       0.80      0.81      0.80       197

avg / total       0.80      0.80      0.80       400


Kappa: 0.605

Accuracy: 0.802

========================================
```
  
