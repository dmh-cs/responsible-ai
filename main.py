import pprint

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from explainer_model import train, inquire
from util import get_label, encode_data

def filter_data(data):
  column_names = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
  data = data[data['days_b_screening_arrest'] <= 30]
  data = data[data['days_b_screening_arrest'] >= -30]
  data = data[data['is_recid'] != -1]
  data = data[data['c_charge_degree'] != 'O']
  data = data[data['score_text'] != 'N/A']
  return data[column_names]

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

  def _build_dataset_for_user(self, restaurants_info, user_info):
    pass

  def rank_restaurants_for_user(self, restaurants_info, user_info):
    user_dataset = self._build_dataset_for_user(restaurants_info, user_info)
    return self.model.predict_proba(user_dataset)

  def prepare_data(self, path='./compas-scores-two-years.csv'):
    raw_data = pd.read_csv(path)
    filtered_data = filter_data(raw_data)
    keep_features = ['priors_count', 'sex', 'age_cat', 'race', 'c_charge_degree']
    data = filtered_data[keep_features].rename(columns={'priors_count': 'num_priors',
                                                        'age_cat': 'age_range',
                                                        'c_charge_degree': 'charge_degree'})
    data['num_priors'] = data['num_priors'].astype(np.float32)
    self.features = ['sex', 'age_range', 'race', 'charge_degree', 'num_priors']
    categorical_features = ['sex', 'age_range', 'race', 'charge_degree']
    categorical_feature_indices = [self.features.index(feature_name) for feature_name in categorical_features]
    self.encoded_data, self.categorical_names = encode_data(data,
                                                            self.features,
                                                            categorical_features)
    self.target = filtered_data['is_recid']
    self.one_hot_encoder = OneHotEncoder(categorical_features=categorical_feature_indices)
    self.one_hot_encoder.fit(self.encoded_data)

  def train(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.encoded_data,
                                                                            self.target)
    self.model.fit(self.one_hot_encoder.transform(self.X_train),
                   self.y_train)
    self.predict_proba_fn = lambda samples: self.model.predict_proba(self.one_hot_encoder.transform(samples))

  def predict(self, X_test=None):
    X_test = X_test or self.X_test # use precomputed test set (created when self.train is called)
    self.predictions = np.argmax(self.predict_proba_fn(X_test), axis=1)
    return self.predictions

  def print_prediction_results(self, predictions=None):
    predictions = None or self.predictions
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

def main():
  printer = pprint.PrettyPrinter(indent=2)
  model = ensemble.RandomForestClassifier()
  ranker = Ranker(model)
  ranker.prepare_data(path='./compas-scores-two-years.csv')
  ranker.train()
  ranker.predict()
  ranker.print_prediction_results()
  explanation = ranker.explain_prediction(ranker.X_test.iloc[0], num_features=2)
  printer.pprint(explanation.as_list())


if __name__ == "__main__": main()
