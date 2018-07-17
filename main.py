import pprint

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from explainer_model import train, inquire

def get_label(explainer, feature, value):
  return list(explainer.categorical_names[explainer.feature_names.index(feature)]).index(value)

def filter_data(data):
  column_names = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
  data = data[data['days_b_screening_arrest'] <= 30]
  data = data[data['days_b_screening_arrest'] >= -30]
  data = data[data['is_recid'] != -1]
  data = data[data['c_charge_degree'] != 'O']
  data = data[data['score_text'] != 'N/A']
  return data[column_names]

def encode_data(data, features, categorical_features):
  categorical_names = {}
  encoded_data = {}
  for i, feature_name in enumerate(features):
    if feature_name in categorical_features:
      label_encoder = LabelEncoder()
      encoded_data[feature_name] = label_encoder.fit_transform(data[feature_name])
      categorical_names[i] = label_encoder.classes_
    else:
      encoded_data[feature_name] = data[feature_name]
  return pd.DataFrame(encoded_data, columns=features), categorical_names

def main():
  printer = pprint.PrettyPrinter(indent=2)
  raw_data = pd.read_csv('./compas-scores-two-years.csv')
  filtered_data = filter_data(raw_data)
  keep_features = ['priors_count', 'sex', 'age_cat', 'race', 'c_charge_degree']
  data = filtered_data[keep_features].rename(columns={'priors_count': 'num_priors',
                                                      'age_cat': 'age_range',
                                                      'c_charge_degree': 'charge_degree'})
  data['num_priors'] = data['num_priors'].astype(np.float32)
  features = ['sex', 'age_range', 'race', 'charge_degree', 'num_priors']
  categorical_features = ['sex', 'age_range', 'race', 'charge_degree']
  categorical_feature_indices = [features.index(feature_name) for feature_name in categorical_features]
  encoded_data, categorical_names = encode_data(data, features, categorical_features)
  X = encoded_data
  y = filtered_data['is_recid']
  one_hot_encoder = OneHotEncoder(categorical_features=categorical_feature_indices)
  one_hot_encoder.fit(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  model = ensemble.RandomForestClassifier()
  model.fit(one_hot_encoder.transform(X_train), y_train)
  predict_fn = lambda samples: model.predict_proba(one_hot_encoder.transform(samples))
  predictions = model.predict(one_hot_encoder.transform(X_test))
  cm = metrics.classification.confusion_matrix(y_test, predictions)
  print(cm)
  print(metrics.classification_report(y_test, predictions))
  print(metrics.accuracy_score(y_test, predictions))
  explainer = train({'X_train': X_train.values, 'y_train': y_train.values},
                    {'feature_names': features,
                     'categorical_feature_indices': categorical_feature_indices,
                     'categorical_names': categorical_names})
  explanation = inquire(explainer,
                        {'instance': X_test.iloc[0],
                         'predict_fn': predict_fn,
                         'num_features': 2})
  printer.pprint(explanation.as_list())


if __name__ == "__main__": main()
