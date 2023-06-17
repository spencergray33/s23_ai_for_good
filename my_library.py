def test_it():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(your_table, your_evidence, your_evidence_value, your_target, your_target_value):
  target_subset = up_table_subset(your_table, your_target, 'equals', your_target_value)
  evidence_list = up_get_column(target_subset, your_evidence)
  p_b_a = sum([1 if v==your_evidence_value else 0 for v in evidence_list])/len(evidence_list)
  return p_b_a + .01

def cond_probs_product(your_table, your_evidence_values, your_target_column, your_target_value):
  your_table_columns = up_list_column_names(your_table)
  your_evidence_columns = your_table_columns[:-1]
  your_evidence_complete = up_zip_lists(your_evidence_columns, your_evidence_values)
  cond_prob_list = []
  for item in your_evidence_complete:
    all_probs = cond_prob(your_table, item[0], item[1], your_target_column, your_target_value)
    cond_prob_list += [all_probs]
  return up_product(cond_prob_list)

def prior_prob(your_table, your_target, your_target_value):
  t_list = up_get_column(your_table, your_target)
  p_a = sum([1 if v==your_target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  neg_cond_prob = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)
  pos_cond_prob = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)
  neg, pos = compute_probs(neg_cond_prob, pos_cond_prob)
  return [neg, pos]



from sklearn.ensemble import RandomForestClassifier

def run_random_forest(train, test, target, n):

  X = up_drop_column(train, target)
  y = up_get_column(train, target)
  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)
  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
  clf.fit(X, y)
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]
  pos_probs[:5]

  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  all_mets[:2]
  metrics_table = up_metrics_table(all_mets)
  metrics_table

  print(metrics_table)
  return None




def metrics(your_list):
  assert isinstance(your_list, list), "Parameter must be a list."
  assert all(isinstance(pair, list) for pair in your_list), "Parameter must be a list of lists."
  assert all(len(pair) == 2 for pair in your_list), "Value must be a zipped list of pairs."
  assert all(isinstance(item, (int, float)) and item >=0 for pair in your_list for item in pair), "Each value in the pair must be an integer >= 0."
  tn = sum([1 if pair==[0,0] else 0 for pair in your_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in your_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in your_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in your_list])
  precision = tp / (tp + fp) if tp+fp !=0 else 0
  recall = tp / (tp + fn) if tp + fn !=0 else 0
  f1 = 2 * ((precision*recall)/(precision+recall)) if precision+recall !=0 else 0
  accuracy = (tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn !=0 else 0
  return {'Precision': precision, 'Recall': recall, 'F1':f1, 'Accuracy':accuracy}



def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)

  for arch in architectures:
    all_results = up_neural_net(train_table, test_table, arch, target)

    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))

  return None

