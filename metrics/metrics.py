# from metrics.f1_score_f1_pa import *
# from metrics.fc_score import *
# from metrics.precision_at_k import *
# from metrics.customizable_f1_score import *
# from metrics.AUC import *
# from metrics.Matthews_correlation_coefficient import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc
import numpy as np

def combine_all_evaluation_scores(y_test, pred_labels):
    events_pred = convert_vector_to_events(y_test) 
    events_gt = convert_vector_to_events(pred_labels)
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    aff_p, aff_r = affiliation['Affiliation_Precision'], affiliation['Affiliation_Recall']
    aff_f1 = 2 * (aff_p * aff_r) / (aff_p + aff_r)
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(y_test, pred_labels)
    vus_results = get_range_vus_roc(y_test, pred_labels, 100) # default slidingWindow = 100
    
    score_list_simple = {
                  "Affiliation precision": aff_p,
                  "Affiliation recall": aff_r,
                  "Affiliation f1 score": aff_f1,
                  "R_AUC_ROC": vus_results["R_AUC_ROC"], 
                  "R_AUC_PR": vus_results["R_AUC_PR"],
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"]
                  }

    return score_list_simple


def get_adjust_F1PA(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    return accuracy, precision, recall, f_score

