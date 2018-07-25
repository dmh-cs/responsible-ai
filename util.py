import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def get_label(explainer, feature, value):
  return list(explainer.categorical_names[explainer.feature_names.index(feature)]).index(value)

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

def load_glove():
  print("Loading Glove Model")
  with open('glove.6B.300d.txt','r') as f:
    model = {}
    for line in f:
      splitLine = line.split()
      word = splitLine[0]
      embedding = np.array([float(val) for val in splitLine[1:]])
      model[word] = embedding
  print("Done.",len(model)," words loaded!")
  return model

def report(predict_fn, X, y):
  predictions = predict_fn(X)
  cm = metrics.classification.confusion_matrix(y, predictions)
  print(cm)
  print(metrics.classification_report(y, predictions))
  print(metrics.accuracy_score(y, predictions))
