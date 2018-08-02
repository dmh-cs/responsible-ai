import pprint

from sklearn import ensemble
from ainsight.compas_explainer import CompasExplainer

def run():
  # printer = pprint.PrettyPrinter(indent=2)
  model = ensemble.RandomForestClassifier()
  compas_explainer = CompasExplainer(model)

  compas_explainer.prepare_data(path='ainsight/compas-scores-two-years.csv')
  compas_explainer.train()
  compas_explainer.predict()
  # compas_explainer.print_prediction_results()
  explanation = compas_explainer.explain_prediction(compas_explainer.X_test.iloc[0], num_features=2)
  return explanation.as_list()

  # printer.pprint(explanation.as_list())

# if __name__ == "__main__": run()