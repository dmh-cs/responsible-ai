import numpy as np
import pandas as pd

from ranker import Ranker
from util import get_label, encode_data
from sklearn.preprocessing import OneHotEncoder

class CompasExplainer(Ranker):
  def __init__(self, model):
    super().__init__(model)

  def filter_data(self, data):
    column_names = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    data = data[data['days_b_screening_arrest'] <= 30]
    data = data[data['days_b_screening_arrest'] >= -30]
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] != 'O']
    data = data[data['score_text'] != 'N/A']
    return data[column_names]

  def prepare_data(self, path='./compas-scores-two-years.csv'):
    raw_data = pd.read_csv(path)
    filtered_data = self.filter_data(raw_data)
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
