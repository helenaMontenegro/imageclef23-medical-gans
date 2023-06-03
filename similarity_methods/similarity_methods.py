import cv2, numpy as np, h5py, os, pandas as pd
from skimage.metrics import structural_similarity
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--infer', action='store_true', help="Generate submission file.")
parser.add_argument('--method', type=str, default='all', help="Method to apply: threshold_max, threshold_avg, retrieval, ranking, clustering or ensemble.")
parser.add_argument('--matrix_file', type=str, default='ssim_matrices.hdf5', help="File that holds the similarity matrices.")
parser.add_argument('--submission_file', type=str, default='submission.csv', help="File where the results will be saved.")
parser.add_argument('--real_dir', type=str, default='real_unknown_1', help="Directory of the real data required on inference.")
args = parser.parse_args()

infer = args.infer
method = args.method
real_folder = args.real_dir
matrix_file = args.matrix_file
submission_file = args.submission_file
admissible_methods = ['all', 'threshold_max', 'threshold_avg', 'retrieval', 'ranking', 'clustering', 'ensemble']
if method not in admissible_methods:
    print('The method ' + method +  ' does not exist. Try one of the following: all, threshold_max, threshold_avg, retrieval, ranking, clustering or ensemble')

f = h5py.File(matrix_file)  # open the file in append mode

gen_real_matrix = f['gen_real_matrix']
gen_real_matrix = np.asarray(gen_real_matrix)
real_matrix = f['real_matrix']
real_matrix = np.asarray(real_matrix)
gen_gen_matrix = f['gen_matrix']
gen_gen_matrix = np.asarray(gen_gen_matrix)
f.close()

r_matrix = np.concatenate((real_matrix, gen_real_matrix.T), axis=1)
gen_matrix = np.concatenate((gen_real_matrix, gen_gen_matrix), axis=1)
whole_matrix = np.concatenate((r_matrix, gen_matrix))
n = int(len(real_matrix)/2)

def evaluate(classification_used, classification_nused):
  tp = np.sum(classification_used)
  fp = np.sum(classification_nused)
  fn = len(classification_used) - np.sum(classification_used)
  tn = len(classification_nused) - np.sum(classification_nused)
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  specificity = tn / (tn + fp)
  recall = tp / (tp + fn)
  fscore = 2 * precision * recall / (precision + recall)
  print('          |      Real     |')
  print('Predicted | Used | Unused |')
  print('Used      |  ' + str("{:02d}".format(tp)) + '  |   ' + str("{:02d}".format(fp)) + '   |')
  print('Unused    |  ' + str("{:02d}".format(fn)) + '  |   ' + str("{:02d}".format(tn)) + '   |')
  print('\nAccuracy: ' + str(accuracy))
  print('Precision: ' + str(precision))
  print('Specificity: ' + str(specificity))
  print('Recall: ' + str(recall))
  print('F1-score: ' + str(fscore))

def threshold(real_matrix, gen_real_matrix, threshold_type='max', threshold=None):
    if threshold is not None:
        threshold = threshold
    elif threshold_type == 'avg3std':
        threshold = np.average(real_matrix) + np.std(real_matrix) * 3
    elif threshold_type == 'avg2std':
        threshold = np.average(real_matrix) + np.std(real_matrix) * 2
    elif threshold_type == 'avg':
        threshold = np.average(real_matrix)
    elif threshold_type == 'max':
        threshold = np.amax(real_matrix)
    print('Threshold: ' + str(threshold))
    classification = np.where(np.amax(gen_real_matrix.T, axis=1) > threshold, 1, 0)
    return classification

def retrieval(real_matrix, gen_real_matrix):
    closest_images = np.argmax(gen_real_matrix, axis=1)
    classification = np.zeros((real_matrix.shape[0],))
    classification[closest_images] = 1
    return classification.astype(int)

def ranking(real_matrix, gen_real_matrix):
    real_ranking_matrix = real_matrix.shape[1] - np.argsort(real_matrix) + 1
    average_ranking = np.average(real_ranking_matrix.T, axis=1)
    overall_avg_ranking = np.average(average_ranking)
    gen_ranking_matrix = gen_real_matrix.shape[1] - np.argsort(gen_real_matrix) + 1
    ranking = np.average(gen_ranking_matrix.T, axis=1)
    classification = np.where(ranking < overall_avg_ranking, 1, 0)
    return classification

def ensemble(base, used=None, not_used=None):
    classification = base
    if used is not None:
        classification[np.where(used == 1)[0]] = 1
    if not_used is not None:
        classification[np.where(not_used == 0)[0]] = 0
    return classification

def clustering(whole_matrix):
    avg_dist = np.average(whole_matrix) + np.std(whole_matrix) * 3
    classification = np.where(np.amax(whole_matrix, axis=1) >= avg_dist, 1, 0)[:len(real_matrix)]
    return classification

save_results = None
if method == 'threshold_max' or method == 'all' or method == 'ensemble':
    threshold_max_results = threshold(real_matrix, gen_real_matrix, threshold_type='max')
    save_results = threshold_max_results
if method == 'threshold_avg' or method == 'all':
    threshold_avg_results = threshold(real_matrix, gen_real_matrix, threshold_type='avg3std')
    save_results = threshold_avg_results
if method == 'retrieval' or method == 'all' or method == 'ensemble':
    retrieval_results = retrieval(real_matrix, gen_real_matrix)
    save_results = retrieval_results
if method == 'ranking' or method == 'all' or method == 'ensemble':
    ranking_results = ranking(real_matrix, gen_real_matrix)
    save_results = ranking_results
if method == 'clustering' or method == 'all':
    clustering_results = clustering(whole_matrix)
    save_results = clustering_results
if method == 'ensemble' or method == 'all':
    ensemble_results = ensemble(ranking_results, threshold_max_results, retrieval_results)
    save_results = ensemble_results

if method != 'all' and not infer:
    evaluate(save_results[:n], save_results[n:])
elif not infer:
    evaluate(threshold_max_results[n:], threshold_max_results[:n])
    evaluate(threshold_avg_results[n:], threshold_avg_results[:n])
    evaluate(retrieval_results[n:], retrieval_results[:n])
    evaluate(ranking_results[n:], ranking_results[:n])
    evaluate(clustering_results[n:], clustering_results[:n])
    evaluate(ensemble_results[n:], ensemble_results[:n])
else:
    real_names = os.listdir(real_folder)
    real_names.sort()

    eval_set = dict()
    eval_set["id"] = real_names
    eval_set["pred"] = save_results

    evaluation_df = pd.DataFrame(data=eval_set)
    evaluation_df.to_csv(submission_file, sep=',', index=False, header=False)
