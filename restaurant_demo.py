from sklearn import ensemble

import restaurant_recommender as rr

def main():
  model = ensemble.RandomForestClassifier()
  recommender = rr.RestaurantRecommender(model)
  recommender.prepare_data()
  recommender.train()
  recommender.predict()
  recommender.print_prediction_results()
  user_id = 'U1077'
  # recommender.print_recs(user_id, 10)
  # recommender.print_rec_not_using_features(user_id, ['religion', 'cuisine_id_x'])
  recommender.record_recs(user_id)

if __name__ == "__main__": main()
