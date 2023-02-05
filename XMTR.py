from sklearn.metrics import mean_absolute_error
import random
import statistics
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor
from mlxtend.preprocessing import TransactionEncoder
from sklearn.multioutput import MultiOutputRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class MTR:
  def __init__(self, model=None, X_train=None, X_test=None, y_train=None, y_test=None, feature_names=None, target_names=None):  
    """Init function
        Args:
            model: The trained RF model
            trainData: the data that the RF was trained on
            feature_names: The names of the features from our dataset
            target_names: The names of the targets from our dataset
            mae: the mean absolute error of the trained RF model
            targets: the number of target values
        Attributes:
            model: The classifier/regression model
            trees: The trees of an trained ensemble system
            feature_names: The names of the features
            min_max_feature_values: A helping dictionary for the path/feature reduction process
            ranked_features: The features ranked based on SHAP Values (Small-Medium Datasets) or Feature Importance (Huge Datasets)
    """
    if X_train is None or X_test is None or y_train is None or y_test is None:
      print("non specified data")
    if feature_names is None:
      print("non specified features names")
    if target_names is None:
      print("non specified target names")

    self.trainData = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

    if model is not None:
      self.model = model
    else:
      # add gridsearch
      parameters = [{
         'criterion': ['squared_error'],#, 'absolute_error'],
         #'max_depth': [2],#, 2, 5],   ----> if it does not extend fully, it may have an issue
         'max_features': ['sqrt'],#, 'log2', 0.75, None],
         'min_samples_leaf' : [1, 2, 5, 10]
      }]
      RF = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
      clf = GridSearchCV(estimator=RF, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')
      clf.fit(self.trainData, self.y_train)
      RF = clf.best_estimator_
      self.model = RF
    self.allowed_error = 0
    self.amountOfReduction = None
    self.trees = self.model.estimators_  # model is never None
    self.predicted = self.model.predict(self.X_test)
    self.feature_names = feature_names
    self.target_names = target_names
    if target_names is not None:
      self.targets = len(target_names)
    self.silly_local_importance = {} # It will fill in only if AR reduction will be applied!
    self.min_max_feature_values = {}
    self.ranked_features = {}
    self.feature_rule_limits = {} # for testing
    self.decisions_and_erros = [] # for testing
    

  def getModel(self):
    if self.model is not None:
      return self.model
    print("you should define the model first")

  def getAllowedError(self):
    return self.allowed_error

  # for testing
  def getFeatureLimits(self):
    return self.feature_rule_limits

  # for testing
  def getDecisionsAndErros(self):
    return self.decisions_and_erros

  def getAmountOfReduction(self):
    return self.amountOfReduction

  def fitError(self, allowed_error=None):
    # if error is int, create array of 1xself.targets with the same error
    # if error is list of len=1, the final errors on the rule will average less than the error
    # if error is list with len=self.targets, the final errors on the rule will be less than the respective error
    if allowed_error is not None:
      if type(allowed_error) == int:
        self.allowed_error = np.array([allowed_error] * self.targets)
      else:
         self.allowed_error = np.array(allowed_error)
    else:
      self.allowed_error = mean_absolute_error(self.predicted, self.y_test, multioutput="raw_values")

  def explain(self, instance, allowed_error=None):
    # fit the model
    self.fitError(allowed_error)

    rules, predictions = self.label_paths(instance) # ranges=rules

    # find min/max of all leaves per tree
    minmax = self.find_regression_trees_min_maxes(self.feature_names)

    # reduce the rules
    reduced_rules, reduced_probabilities, local_error = self._reduce_through_association_rules(rules, predictions)
    self.amountOfReduction = [len(reduced_rules), len(rules)]

    # compose the final rule
    return self.composeRule(instance, reduced_rules, local_error)



  def label_paths(self, instance):
    """label_paths function finds the ranges and predictions for each label
    Args:
        instance: The instance we want to find the paths
    Return:
        a list which contains a dictionary with features as keys and their min max ranges as values per tree
        and a list with the predictions of each tree for the examined instance
    """
    ranges = []
    predictions = []
    for tree in self.trees:
      n_tree_prediction = []
      for np in tree.predict([instance]):
        n_tree_prediction.append(np) 
      tree_prediction = n_tree_prediction
      path = tree.decision_path([instance])
      leq = {}  # leq: less equal ex: x <= 1
      b = {}  # b: bigger ex: x > 0.6
      local_range = {}
      for node in path.indices:
        feature_id = tree.tree_.feature[node]
        feature = self.feature_names[feature_id]
        threshold = tree.tree_.threshold[node]
        if threshold != -2.0:
          if instance[feature_id] <= threshold:
            leq.setdefault(feature, []).append(threshold)
          else:
            b.setdefault(feature, []).append(threshold)
      for k in leq:
        local_range.setdefault(k, []).append(['<=', min(leq[k])])  # !!
      for k in b:
        local_range.setdefault(k, []).append(['>', max(b[k])])  # !!
      ranges.append(local_range)
      predictions.append(list(tree_prediction[0]))
    return ranges, predictions


  def tree_to_code(self, tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] for i in tree_.feature]
    leaf_nodes = []
    def recurse(node, depth):
      indent = "  " * depth
      if tree_.feature[node] != -2:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        temp = []
        [temp.append(t) for t in recurse(tree_.children_left[node], depth + 1)]
        [temp.append(t) for t in recurse(tree_.children_right[node], depth + 1)]
        return temp
      else:
          return([node])

    leaf_nodes.append(recurse(0, 1))
    return(leaf_nodes[0])


  # we want ths to find the minmax of excluded trees
  def find_regression_trees_min_maxes(self, feature_names):
    """ finds min max of the leaves for each tree
    Args:
        trees: list of estimators, the examined trees
        num_of_targets: the number of the targets in the regression
    Return:
        A dict that contains a list of num_of_targets*2 values, first half are the min
        of each target, last half are maxes.
        example: [array([0.]), array([20.]), array([17.19]), | array([29.]), array([78.]), array([58.53])]
    """
    trees = self.trees
    min_max_leaf_prediction_per_tree = {}
    for i in range(len(trees)):
      tree = trees[i]
      min_max_leaf_prediction_per_tree[i] = [None for i in range(self.targets*2)]
      leaf_nodes = self.tree_to_code(tree, feature_names)
      for l in leaf_nodes:
        # here value returns the 3 targets per leaf and we want their minmax
        value = tree.tree_.value[l]
        for target in range(len(value)):
          if min_max_leaf_prediction_per_tree[i][target+len(value)] is None or value[target] > min_max_leaf_prediction_per_tree[i][target+len(value)]:
              min_max_leaf_prediction_per_tree[i][target+len(value)] = value[target]
          if  min_max_leaf_prediction_per_tree[i][target] is None or value[target] < min_max_leaf_prediction_per_tree[i][target]:
              min_max_leaf_prediction_per_tree[i][target] = value[target]
    self.min_max_leaf_prediction_per_tree = min_max_leaf_prediction_per_tree
    return min_max_leaf_prediction_per_tree



  # Algorithm 6
  #https://github.com/intelligence-csd-auth-gr/LionLearn/blob/1931ed50a2ca47f80243ce31e258d3f5fa9e701f/LionForests/lionforests.py#L1022
  def _reduce_through_distribution_multi(self, instance, rules, predictions, instance_qe, method, targets):
    """ path reduction
    Args:
        instance: used for the random seed
        rules: we get them from func label_paths
        predictions: predictions per tree, for the given target. we get them from func label_paths
        instance_qe: allowed error
        method: R2 for inner and R3 for outter
        target: a list with index of targets
    Return:
        the reduced rules and reduced predictions for the predifined targets
    """
    # the final results
    local_error_per_target = []
    reduced_rules_per_target = []
    reduced_predictions_per_target = []

    # loop for each selected target
    for target in targets:
      reduced_rules = rules
      reduced_predictions = predictions[:,target]
      real_prediction = np.array(predictions[:,target]).mean()
      min_errors = abs(instance_qe)
      min_s = 0

      for s in [.1, .2, .5, 1, 2, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]:
        np.random.seed(42)
        normal_dis = np.random.normal(real_prediction, np.array(predictions[:,target]).std()/s, 100)
        c = 0
        l_reduced_rules = []
        l_reduced_predictions = []


        # for each tree basically
        for i in predictions[:,target]:
          # R2 is inner and R3 is outter
          if (method == 'R2' and not (i < normal_dis.min() or i > normal_dis.max())) or (method == 'R3' and (i < normal_dis.min() or i > normal_dis.max())):
              l_reduced_rules.append(rules[c])
              l_reduced_predictions.append(predictions[:,target][c])
          else:
              # the min
              dis_a = abs(i - self.min_max_leaf_prediction_per_tree[c][target][0])
              # the max
              dis_b = abs(i - self.min_max_leaf_prediction_per_tree[c][target+self.targets][0])

              if dis_a < dis_b:
                  l_reduced_predictions.append(self.min_max_leaf_prediction_per_tree[c][target+self.targets][0])
              else:
                  l_reduced_predictions.append(self.min_max_leaf_prediction_per_tree[c][target][0])
          c = c + 1

        l_error = abs(np.array(l_reduced_predictions).mean() - real_prediction)
        if l_error < abs(instance_qe) and len(l_reduced_rules) < len(reduced_rules):
            reduced_rules = l_reduced_rules
            reduced_predictions = l_reduced_predictions
            min_errors = l_error
            min_s = s

      local_error_per_target.append(l_error)
      reduced_rules_per_target.append(reduced_rules)
      reduced_predictions_per_target.append(reduced_predictions)
    return local_error_per_target, reduced_rules_per_target, reduced_predictions_per_target


  #https://github.com/intelligence-csd-auth-gr/LionLearn/blob/1931ed50a2ca47f80243ce31e258d3f5fa9e701f/LionForests/lionforests.py#L715
  def _reduce_through_association_rules(self, rules, probabilities):
    """ path reduction
    Args:
        rules: we get them from func label_paths
        probabilities: predictions per tree, for the given target. we get them from func label_paths
    Return:
        reduced rules and the reduced predictions
    """
    reduced_rules = rules
    reduced_probabilities = probabilities

    # create itemsets of features per rule create from estimators
    get_itemsets = []
    items = set()
    # for each rule of the estimators
    for rule in rules:
      itemset = []
      # for each feature of the rule
      for p in rule:
        itemset.append(p)
        items.add(p)
      get_itemsets.append(itemset)
    max_number_of_features = len(items) # all distinct features
    del items

    # one-hot transform into boolean array and put in df
    tEncoder = TransactionEncoder()
    oneHotItemset = tEncoder.fit(get_itemsets).transform(get_itemsets)
    df = pd.DataFrame(oneHotItemset, columns=tEncoder.columns_)
    
    # run association rules and get frequent itemsets (fi) (ADD fpgrowth)
    temp_fi = apriori(df, min_support=0.1, use_colnames=True)
    if len(temp_fi.values) == 0:
        return rules, probabilities

    # get the frequent itemsets 
    frequent_itemsets = (
      association_rules(temp_fi, metric="support", min_threshold=0.1)
      .sort_values(by="confidence",ascending=True)
    )
    
    # Collect features and their importance from the association rules
    probability = 0
    k = 1
    antecedents = []
    antecedents_weights = {}
    antecedents_set = set()
    wcounter = 0
    for antecedent in list(frequent_itemsets['antecedents']):
      if tuple(antecedent) not in antecedents_set:
        antecedents_set.add(tuple(antecedent))
        for antecedent_i in list(antecedent):
          if antecedent_i not in antecedents:
            antecedents.append(antecedent_i)
      for antecedent_i in list(antecedent):
        wcounter = wcounter + 1
        if antecedent_i not in antecedents_weights:
          antecedents_weights[antecedent_i] = 1/wcounter
        else:
          antecedents_weights[antecedent_i] = antecedents_weights[antecedent_i] + 1/wcounter
    self.silly_local_importance = antecedents_weights # dict {feature: importance}
    size_of_ar = len(antecedents)

    
    items = set() # may be redundant since it was calculated/deleted previously
    new_feature_list = []
    for pr in reduced_rules:
      for p in pr:
        items.add(p)
    new_feature_list = list(items)

    reduced_rules = []
    #reduced_probabilities = []
    local_error = 2 * abs(self.allowed_error)
    keep_pids = []

    while np.any(local_error > abs(self.allowed_error)) and k <= size_of_ar:
      feature_set = set()
      for i in range(0, k):
        feature_set.add(antecedents[i])

      new_feature_list = list(feature_set)
      redundant_features = [
          i for i in self.feature_names if i not in new_feature_list]
      reduced_rules = []
      pid = 0
      keep_pids = []
      reduced_probabilities = []
      
      # for each rule, for each target
      for rule in rules:
        reduced_probabilities_per_target = []
        flag = True
        for target in range(self.targets):
          # in case of no redundant features in the rule
          if sum([1 for j in redundant_features if j in rule]) == 0:
            if flag: # this so the rule is added once per target
              reduced_rules.append(rule) # will get this using keep_pids
              flag=False
            reduced_probabilities_per_target.append(probabilities[pid][target])
            keep_pids.append(pid)
          else:
            dis_a = abs(probabilities[pid][target]- self.min_max_leaf_prediction_per_tree[pid][target][0])
            dis_b = abs(probabilities[pid][target] - self.min_max_leaf_prediction_per_tree[pid][target+self.targets][0])
            if dis_a < dis_b:
              reduced_probabilities_per_target.append(self.min_max_leaf_prediction_per_tree[pid][target+self.targets][0])
            else:
              reduced_probabilities_per_target.append(self.min_max_leaf_prediction_per_tree[pid][target][0])
        pid = pid + 1
        reduced_probabilities.append(reduced_probabilities_per_target)
      local_error = mean_absolute_error(probabilities, reduced_probabilities, multioutput="raw_values")
      k += 1

    # reset
    if np.any(local_error > abs(self.allowed_error)) and k > size_of_ar:  
      keep_pids = []
      reduced_rules = []
      reduced_probabilities = []
      pid = 0
      for i in rules:
        reduced_rules.append(i)
        reduced_probabilities.append(probabilities[pid])
        keep_pids.append(pid)
        pid = pid + 1
    temp_pids = keep_pids.copy()
    last_pid = None

    while np.all(local_error < abs(self.allowed_error)) and len(temp_pids) > 2:
      reduced_rules = []
      pid = 0
      reduced_probabilities = []
      last_pid = temp_pids[-1]
      temp_pids = temp_pids[:-1]

      for rule in rules:
        reduced_probabilities_per_target = []
        if pid in temp_pids:
          reduced_rules.append(rule)
          for target in range(self.targets): 
            reduced_probabilities_per_target.append(probabilities[pid][target])
        else:
          for target in range(self.targets):
            dis_a = abs(probabilities[pid][target] - self.min_max_leaf_prediction_per_tree[pid][target][0])
            dis_b = abs(probabilities[pid][target] - self.min_max_leaf_prediction_per_tree[pid][target+self.targets][0])
            if dis_a < dis_b:
                reduced_probabilities_per_target.append(self.min_max_leaf_prediction_per_tree[pid][target+self.targets][0])
            else:
                reduced_probabilities_per_target.append(self.min_max_leaf_prediction_per_tree[pid][target][0])
        pid = pid + 1
        reduced_probabilities.append(reduced_probabilities_per_target)

      local_error = mean_absolute_error(probabilities, reduced_probabilities, multioutput="raw_values")

    if last_pid is not None:
      temp_pids.append(last_pid)
      reduced_rules = []
      pid = 0
      reduced_probabilities = []

      for rule in rules:
        reduced_probabilities_per_target = []
        if pid in temp_pids:
          reduced_rules.append(rule)
          for target in range(self.targets): 
            reduced_probabilities_per_target.append(probabilities[pid][target])
        else:
          for target in range(self.targets):
            dis_a = abs(probabilities[pid][target] - self.min_max_leaf_prediction_per_tree[pid][target][0])
            dis_b = abs(probabilities[pid][target] - self.min_max_leaf_prediction_per_tree[pid][target+self.targets][0])
            if dis_a < dis_b:
                reduced_probabilities_per_target.append(self.min_max_leaf_prediction_per_tree[pid][target+self.targets][0])
            else:
                reduced_probabilities_per_target.append(self.min_max_leaf_prediction_per_tree[pid][target][0])
        pid = pid + 1
        reduced_probabilities.append(reduced_probabilities_per_target)

      local_error = mean_absolute_error(probabilities, reduced_probabilities, multioutput="raw_values")
    local_error = mean_absolute_error(probabilities, reduced_probabilities, multioutput="raw_values")
    return reduced_rules, reduced_probabilities, local_error


  def _pre_feature_range_caluclation(self, rules, feature):
    ''' function that return the min and max values that a feature from the 
        reduced rules can get
      args:
        rules: the rules we got after the reduction
        features: a particular feature out of the reduced feature set
    '''
    for i in range(len(self.feature_names)):
      self.min_max_feature_values[self.feature_names[i]] = [min(self.trainData[:, i]), max(self.trainData[:, i])]

    mi = None
    ma = None
    for i in rules:
      if feature in i:
        if len(i[feature]) == 1:
          if i[feature][0][0] == "<=":
            if ma is None or ma >= i[feature][0][1]:
              ma = i[feature][0][1]
          else:
            if mi == None or mi <= i[feature][0][1]:
              mi = i[feature][0][1]
        else:
          if mi == None or mi <= i[feature][1][1]:
            mi = i[feature][1][1]
          if ma == None or ma >= i[feature][0][1]:
            ma = i[feature][0][1]
    if mi is None:
      mi = self.min_max_feature_values[feature][0]
    if ma is None:
      ma = self.min_max_feature_values[feature][1]
    return [mi, ma]


  # https://github.com/intelligence-csd-auth-gr/LionLearn/blob/1931ed50a2ca47f80243ce31e258d3f5fa9e701f/LionForests/lionforests.py#L522
  def composeRule(self, instance, reduced_rules, local_error):
    ''' function used to compose the final rule
    '''
    rule = "if "
    temp_f_mins = {}
    temp_f_maxs = {}
    self.feature_rule_limits = {}
    self.decisions_and_erros = []

    # get the features that appear on the reduced rules
    items = set()
    for r in reduced_rules:
      for feature in r:
        items.add(feature)
    local_feature_names = list(items)


    for feature in self.feature_names:
      if feature in local_feature_names:
        mi, ma = self._pre_feature_range_caluclation(reduced_rules, feature)
        temp_f_mins[feature] = mi
        temp_f_maxs[feature] = ma

    f_mins = []
    f_maxs = []
    for feature in self.feature_names:
      if feature in temp_f_mins:
        f_mins.append(temp_f_mins[feature])
      else:
        f_mins.append(0)
      if feature in temp_f_maxs:
        f_maxs.append(temp_f_maxs[feature])
      else:
        f_maxs.append(0)

    # create the decision for all target values
    decision = {}
    pred = self.model.predict([instance])[0]
    if local_error is not None:
      for tar in range(len(self.target_names)):
        decision[self.target_names[tar]] = self.target_names[tar] + ': ' + str(round(pred[tar], 4)) + " +/- " + str(round(local_error[tar], 4)) + " error"
        self.decisions_and_erros.append([pred[tar], local_error[tar]])
    else:
      for tar in range(len(self.target_names)):
        decision[self.target_names[tar]] = self.target_names[tar] + ': ' + str(round(pred[tar], 4))
        self.decisions_and_erros.append([pred[tar], 0])
    # we only use this for reference on the ranked features below, its the same for all targets
    target_name = self.target_names[0]


    d = {'Feature': self.feature_names,
          'Importance': self.model.feature_importances_}
    for ind in range(len(self.target_names)):
      self.ranked_features[self.target_names[ind]] = \
          pd.DataFrame(data=d).sort_values(
              by=['Importance'], ascending=False)['Feature'].values

    # create the rule containing mins and maxes of the reduced features
    for ranked_f in self.ranked_features[target_name]:
      f = self.feature_names.get_loc(ranked_f)
      if self.feature_names[f] in local_feature_names:
        mmi = np.array([f_mins, f_mins])[0][f]
        mma = np.array([f_maxs, f_maxs])[0][f]  # ena tab mesa
        self.feature_rule_limits[self.feature_names[f]] = [mmi, mma]
        rule = rule + str(round(mmi, 3)) + "<=" + self.feature_names[f] + "<=" + str(round(mma, 3)) + " & "

    rule = rule[:-3] + " then "
    for key in decision.keys():
      rule += decision[key] + ", "
    return rule[:-2]
