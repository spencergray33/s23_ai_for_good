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

def test_it():
  return 'loaded'
