import re
from functools import reduce

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

from ranker import Ranker
from util import load_glove, encode_data

class ResaurantRecommender(Ranker):
  def __init__(self, model):
    super().__init__(model)
    self.data = None

  def build_dataset_for_user(self, user_id):
    user_columns = ['userID', 'cuisine_id_y','smoker','drink_level','dress_preference','ambiance','transport','marital_status', 'hijos','interest','personality','religion','activity','color','budget', 'Upayment', 'height', 'weight', 'birth_year']
    user_restaurant_data = self.data_with_ids.drop_duplicates('placeID').to_dict('list')
    user_info = self.data_with_ids[self.data_with_ids['userID'] == user_id].iloc[0].to_dict()
    for column in user_columns:
      for i in range(len(user_restaurant_data[column])):
        user_restaurant_data[column][i] = user_info[column]
    return pd.DataFrame(user_restaurant_data, columns=self.features + ['userID', 'placeID'])

  def prepare_data(self):
    # Restaurant Features
    df_res_accept = pd.read_csv('data/chefmozaccepts.csv',index_col='placeID')
    df_res_cuisine = pd.read_csv('data/chefmozcuisine.csv',index_col='placeID')
    df_res_parking = pd.read_csv('data/chefmozparking.csv',index_col='placeID')
    df_res_hours = pd.read_csv('data/chefmozhours4.csv',index_col='placeID')
    df_res_location = pd.read_csv('data/geoplaces2.csv',index_col='placeID')
    df_res_accept['placeID'] = df_res_accept.index
    df_res_cuisine['placeID'] = df_res_cuisine.index
    df_res_parking['placeID'] = df_res_parking.index
    df_res_hours['placeID'] = df_res_hours.index
    df_res_location['placeID'] = df_res_location.index
    # User Features
    df_user_cuisine = pd.read_csv('data/usercuisine.csv',index_col='userID')
    df_user_payment = pd.read_csv('data/userpayment.csv',index_col='userID')
    df_user_profile = pd.read_csv('data/userprofile.csv',index_col='userID')
    df_user_cuisine['userID'] = df_user_cuisine.index
    df_user_payment['userID'] = df_user_payment.index
    df_user_profile['userID'] = df_user_profile.index
    late_hours = [int(item.split('-')[1].strip(';').split(':')[0]) > 21 for item in df_res_hours.hours]
    # Pre-process Hours into binary categorical feature
    df_res_latehours = pd.DataFrame(index=df_res_hours.index,data=late_hours,columns=['late_hours'])
    df_res_latehours['placeID'] = df_res_latehours.index
    # Pre-process cuisine of both restaurants and users
    glv_model = load_glove()
    user_cuisine_list = [re.sub("[^a-zA-Z0-9 ]", " ", item.lower()).strip().split(' ') \
                         for item in df_user_cuisine['Rcuisine'].values]
    res_cuisine_list = [re.sub("[^a-zA-Z0-9 ]", " ", item.lower()).strip().split(' ') \
                        for item in df_res_cuisine['Rcuisine'].values]
    total_list = user_cuisine_list + res_cuisine_list
    vector_list = []
    for item_list in total_list:
      cur_vector = np.zeros([300,])
      for item in item_list:
        cur_vector += glv_model[item]
      cur_vector = cur_vector / len(item_list)
      vector_list.append(cur_vector)
    vector_matrix = np.asarray(vector_list)
    model = MiniBatchKMeans(n_clusters=4)
    clusters = model.fit_transform(vector_matrix)
    cluster_idx = np.argmin(clusters,axis=1)
    df_user_cuisine['cuisine_id'] = cluster_idx[:df_user_cuisine.shape[0]]
    df_res_cuisine['cuisine_id'] = cluster_idx[df_user_cuisine.shape[0]:]
    df_res_location = df_res_location.rename(columns={'Rambience': 'Rambiance'})
    res_location_cols = ['placeID', 'latitude','longitude','alcohol','smoking_area','dress_code','accessibility','franchise','Rambiance']
    df_res_info = df_res_location[res_location_cols]
    # Merge all rest. features in one frame
    dfs = [df_res_accept,df_res_cuisine,df_res_latehours,df_res_parking,df_res_info]
    df_res = reduce(lambda left,right: pd.merge(left,right,on='placeID', how='outer'), dfs)
    df_res.set_index('placeID', inplace=True)
    dfs = [df_user_cuisine,df_user_payment,df_user_profile]
    df_user = reduce(lambda left,right: pd.merge(left,right,on='userID', how='outer'), dfs)
    df_user.set_index('userID', inplace=True)
    # interaction data
    df_interaction = pd.read_csv('data/rating_final.csv')
    place_ids = list(set(df_interaction['placeID'].tolist()).intersection(set(df_res.index)))
    interaction_res = df_res.loc[place_ids]
    # fill na values
    interaction_res.loc[:,'cuisine_id'] = interaction_res.loc[:,'cuisine_id'].fillna(method='bfill')
    interaction_res.loc[:,'late_hours'] = interaction_res.loc[:,'late_hours'].fillna(interaction_res['late_hours'].value_counts().idxmax())
    interaction_res.loc[:,'Rpayment'] = interaction_res.loc[:,'Rpayment'].fillna('cash')
    interaction_res['placeID'] = interaction_res.index
    user_ids = list(set(df_interaction['userID'].tolist()).intersection(set(df_user.index)))
    interaction_user = df_user.loc[user_ids,:]
    # fill na values
    interaction_user = interaction_user.fillna(method='bfill')
    interaction_user['userID'] = interaction_user.index
    restaurant_cat_colums = ['Rpayment','cuisine_id_x','late_hours','parking_lot','alcohol','smoking_area','dress_code', 'accessibility','franchise','Rambiance']
    user_cat_columns = ['cuisine_id_y','smoker','drink_level','dress_preference','ambiance','transport','marital_status', 'hijos','interest','personality','religion','activity','color','budget', 'Upayment']
    merged = pd.merge(df_interaction,interaction_res,on='placeID',how='left')
    merged = merged.drop_duplicates(keep='first',subset=['placeID','userID'])
    merged = pd.merge(merged,interaction_user,on='userID',how='left')
    merged = merged.drop_duplicates(keep='first',subset=['placeID','userID'])
    merged = merged.rename(columns={'ambience': 'ambiance'})
    self.data = merged
    self.features = ['Rambiance', 'Upayment', 'accessibility', 'activity', 'alcohol', 'ambiance', 'birth_year', 'budget', 'color', 'cuisine_id_x', 'cuisine_id_y', 'dress_code', 'dress_preference', 'drink_level', 'franchise', 'height', 'hijos', 'interest', 'late_hours', 'marital_status', 'parking_lot', 'personality', 'religion', 'smoker', 'smoking_area', 'transport', 'weight', 'Rpayment']
    categorical_features = restaurant_cat_colums + user_cat_columns
    self.categorical_feature_indices = [self.features.index(feature_name) for feature_name in categorical_features]
    self.encoded_data, self.categorical_names = encode_data(merged,
                                                            self.features,
                                                            categorical_features)
    self.target = merged['rating'] > 1
    self.one_hot_encoder = OneHotEncoder(categorical_features=self.categorical_feature_indices)
    self.one_hot_encoder.fit(self.encoded_data)
    self.data_with_ids = self.encoded_data.copy()
    self.data_with_ids['userID'] = merged['userID']
    self.data_with_ids['placeID'] = merged['placeID']


  def rank_restaurants_for_user(self, user_dataset):
    without_ids = user_dataset.loc[:, (user_dataset.columns != 'userID') & (user_dataset.columns != 'placeID')]
    probas = np.squeeze(self.predict_proba_fn(without_ids).T)[0]
    indexes = np.argsort(probas)
    return user_dataset['placeID'].values[indexes], indexes
