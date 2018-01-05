# SklearnTextClassify


This project implements basic text classification functionality using Sklearn. It will be of interest to anyone who'd like to  quickly run text classification experiments with a minimum of fuss and wrangling with the Sklearn API. Sklearn tends to make common feature generation functionality such as concatenation of different feature types (e.g., categorical and numeric types) an oddly difficult process. Several Stackoverflow exchanges testify to this.  Here, I've collected various solutions to creating heterogenous feature types in Sklearn along with more contemporary types such as embeddings-based features. 

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
