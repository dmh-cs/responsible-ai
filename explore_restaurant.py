import pydash as _
import pprint

import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from util import get_label, encode_data, report
from explainer_model import train, inquire

def main():
  printer = pprint.PrettyPrinter(indent=2)
  restaurant_columns = ['alcohol', 'smoking_area', 'dress_code', 'accessibility', 'price', 'Rambience', 'franchise']
  restaurant_cat_columns = ['alcohol', 'smoking_area', 'dress_code', 'accessibility', 'price', 'Rambience', 'franchise']
  user_columns = ['smoker', 'drink_level', 'dress_preference', 'ambience', 'transport', 'marital_status', 'hijos', 'birth_year', 'interest', 'personality', 'religion', 'activity', 'weight', 'budget', 'height']
  user_cat_columns = ['smoker', 'drink_level', 'dress_preference', 'ambience', 'transport', 'marital_status', 'hijos', 'interest', 'personality', 'religion', 'activity', 'budget']
  features = user_columns + restaurant_columns
  categorical_features = user_cat_columns + restaurant_cat_columns
  categorical_feature_indices = [features.index(feature_name) for feature_name in categorical_features]
  restaurant_info = pd.read_csv('./data/geoplaces2.csv',
                                dtype=dict(zip(restaurant_cat_columns, [str] * len(restaurant_cat_columns))))
  user_info = pd.read_csv('./data/userprofile.csv',
                          dtype=dict(zip(user_cat_columns, [str] * len(user_cat_columns))))
  cuisine_type = pd.read_csv('./data/chefmozcuisine.csv')
  user_rating = pd.read_csv('./data/rating_final.csv')
  user_preferences = pd.read_csv('./data/usercuisine.csv')
  place_ids = _.intersection(user_rating['placeID'].tolist(),
                             restaurant_info['placeID'].tolist())
  user_ids = _.intersection(user_rating['userID'].tolist(),
                            user_info['userID'].tolist())
  joined_data_dict = {column: [] for column in features + ['rating']}
  for row in user_rating[['userID', 'placeID', 'rating']].iterrows():
    user_id = row[1]['userID']
    place_id = row[1]['placeID']
    if user_id in user_ids and place_id in place_ids:
      joined_data_dict['rating'].append(row[1]['rating'])
      for column in user_columns:
        joined_data_dict[column].append(user_info[user_info['userID'] == user_id][column].iloc[0])
      for column in restaurant_columns:
        joined_data_dict[column].append(restaurant_info[restaurant_info['placeID'] == place_id][column].iloc[0])
  joined_data = pd.DataFrame(joined_data_dict)
  data = joined_data.loc[:, joined_data.columns != 'rating']
  rating = joined_data['rating']
  encoded_data, categorical_names = encode_data(data, features, categorical_features)
  X = encoded_data
  y = rating.values == 2
  one_hot_encoder = OneHotEncoder(categorical_features=categorical_feature_indices)
  one_hot_encoder.fit(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  model = ensemble.RandomForestClassifier(class_weight='balanced')
  model.fit(one_hot_encoder.transform(X_train), y_train)
  predict_fn = lambda X: model.predict(one_hot_encoder.transform(X))
  predict_proba_fn = lambda X: model.predict_proba(one_hot_encoder.transform(X))
  report(predict_fn, X_test, y_test)
  explainer = train({'X_train': X_train.values, 'y_train': y_train},
                    {'feature_names': features,
                     'categorical_feature_indices': categorical_feature_indices,
                     'categorical_names': categorical_names})
  explanation = inquire(explainer,
                        {'instance': X_test.iloc[0],
                         'predict_fn': predict_proba_fn,
                         'num_features': 5})
  printer.pprint(explanation.as_list())


if __name__ == "__main__": main()
