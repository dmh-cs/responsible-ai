import lime.lime_tabular


def train(train_data, train_args):
  X = train_data['X_train']
  y = train_data['y_train']
  feature_names = train_args['feature_names']
  categorical_feature_indices = train_args['categorical_feature_indices']
  categorical_names = train_args['categorical_names']
  explainer = lime.lime_tabular.LimeTabularExplainer(X,
                                                     training_labels=y,
                                                     feature_names=feature_names,
                                                     categorical_features=categorical_feature_indices,
                                                     categorical_names=categorical_names)
  return explainer

def inquire(explainer, explainer_args):
  instance = explainer_args['instance']
  predict_fn = explainer_args['predict_fn']
  num_features = explainer_args.get('num_features') or 10
  num_samples = explainer_args.get('num_samples') or 5000
  distance_metric = explainer_args.get('distance_metric') or 'euclidean'
  model_regressor = explainer_args.get('model_regressor')
  return explainer.explain_instance(instance,
                                    predict_fn,
                                    num_features=num_features,
                                    num_samples=num_samples,
                                    distance_metric=distance_metric,
                                    model_regressor=model_regressor)
