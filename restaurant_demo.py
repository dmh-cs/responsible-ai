import pprint

from sklearn import ensemble

import restaurant_recommender as rr

def main():
  printer = pprint.PrettyPrinter(indent=2)
  model = ensemble.RandomForestClassifier()
  recommender = rr.ResaurantRecommender(model)
  recommender.prepare_data()
  recommender.train()
  recommender.predict()
  recommender.print_prediction_results()
  explanation = recommender.explain_prediction(recommender.X_test.iloc[0],
                                               num_features=5)
  printer.pprint(explanation.as_list())


if __name__ == "__main__": main()
