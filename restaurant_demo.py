import pprint

from sklearn import ensemble
import pandas as pd

import restaurant_recommender as rr

def main():
  restaurants = pd.read_csv('data/geoplaces2.csv',index_col='placeID')
  printer = pprint.PrettyPrinter(indent=2)
  model = ensemble.RandomForestClassifier()
  recommender = rr.ResaurantRecommender(model)
  recommender.prepare_data()
  recommender.train()
  recommender.predict()
  recommender.print_prediction_results()
  user_id = 'U1077'
  user_dataset = recommender.build_dataset_for_user(user_id)
  without_ids = user_dataset.loc[:, (user_dataset.columns != 'userID') & (user_dataset.columns != 'placeID')]
  recommendations, recommended_indexes = recommender.rank_restaurants_for_user(user_dataset)
  for place_id, row in zip(recommendations[:10], without_ids.values[recommended_indexes][:10]):
    print(place_id)
    print(restaurants.loc[place_id]['name'])
    explanation = recommender.explain_prediction(row, num_features=5)
    printer.pprint(explanation.as_list())


if __name__ == "__main__": main()
