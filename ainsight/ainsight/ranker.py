import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split

from ainsight.explainer_model import train, inquire

class Ranker(object):
  def __init__(self, model):
    self.model = model
    self.one_hot_encoder             = None
    self.encoded_data                = None
    self.categorical_names           = None
    self.target                      = None
    self.predict_proba_fn            = None
    self.predictions                 = None
    self.X_train                     = None
    self.X_test                      = None
    self.y_train                     = None
    self.y_test                      = None
    self.features                    = None
    self.categorical_feature_indices = None
    self.explainer                   = None

  def train(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.encoded_data,
                                                                            self.target)
    self.model.fit(self.one_hot_encoder.transform(self.X_train),
                   self.y_train)
    self.predict_proba_fn = lambda samples: self.model.predict_proba(self.one_hot_encoder.transform(samples))

  def predict(self, X_test=None):
    X_test = self.X_test if X_test is None else X_test # use precomputed test set (created when self.train is called)
    self.predictions = np.argmax(self.predict_proba_fn(X_test), axis=1)
    return self.predictions

  def print_prediction_results(self, predictions=None):
    predictions = self.predictions if predictions is None else predictions
    cm = metrics.classification.confusion_matrix(self.y_test,
                                                 predictions)
    print(cm)
    print(metrics.classification_report(self.y_test, predictions))
    print(metrics.accuracy_score(self.y_test, predictions))

  def explain_prediction(self, row_to_explain, **kwargs):
    if self.explainer is None:
      self.explainer = train({'X_train': self.X_train.values, 'y_train': self.y_train.values},
                             {'feature_names': self.features,
                              'categorical_feature_indices': self.categorical_feature_indices,
                              'categorical_names': self.categorical_names})
    args = {'instance': row_to_explain,
            'predict_fn': self.predict_proba_fn}
    args.update(kwargs)
    return inquire(self.explainer, args)
