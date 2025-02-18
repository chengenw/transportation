from termcolor import cprint
import numpy as np
from utils import minmax_scale_dummy
import ot as pot
import sklearn
from sdmetrics.reports.single_table import QualityReport
from sklearn import metrics
import time

def pred_score(train_data, syn_data, test_data, y_name, model):
    train_X = train_data.drop(y_name, axis=1).to_numpy()
    syn_X = syn_data.drop(y_name, axis=1).to_numpy()
    test_X = test_data.drop(y_name, axis=1).to_numpy()
    train_y = train_data[y_name].to_numpy()
    syn_y = syn_data[y_name].to_numpy()
    test_y = test_data[y_name].to_numpy()

    model.fit(train_X, train_y) # train on real training data
    tr_tr_score = model.score(train_X, train_y)
    tr_syn_score = model.score(syn_X, syn_y)
    tr_te_score = model.score(test_X, test_y)

    model.fit(syn_X, syn_y) # train on syn data
    syn_syn_score = model.score(syn_X, syn_y)
    syn_tr_score = model.score(train_X, train_y)
    syn_te_score = model.score(test_X, test_y)

    return tr_tr_score, tr_syn_score, tr_te_score, syn_syn_score, syn_tr_score, syn_te_score

def dwn_score(taxi_data_train, synth_data_all, taxi_data_test, model, score_k):
    cols = ['tip_amount', 'total_amount', 'dropoff_time', 'dropoff_longitude']
    y_name = cols[1]

    score_d = {k: [] for k in score_k}
    for i in range(len(synth_data_all)):
        synth_data_df = synth_data_all[i]
        scores = pred_score(taxi_data_train, synth_data_df, taxi_data_test, y_name, model()) # tr_train_score, tr_test_score, tr_val_score
        for j in range(len(scores)):
            score_d[score_k[j]].append(scores[j])
        cprint(
            f'Trained on train, col <{y_name}>: train score {scores[0]:.4f}, test score {scores[1]:.4f}, val score {scores[2]:.4f}',
            'green')
        scores = pred_score(synth_data_df, taxi_data_train, taxi_data_test, y_name,
                                                                    model())
        for j in range(len(scores)):
            score_d[score_k[j + 3]].append(scores[j])
        cprint(
            f'Trained on synthetic, col <{y_name}>: train score {scores[0]:.4f}, test score {scores[1]:.4f}, val score {scores[2]:.4f}',
            'green')

    return score_d

def sdv_score(taxi_data_train, synth_data_df, metadata):
    report = QualityReport()
    start_time = time.time()
    report.generate(taxi_data_train,
                    synth_data_df,
                    metadata.to_dict(), verbose=False)
    overall_score = report.get_score()
    cprint(f'sdv overall score {overall_score:.4f}, time elapsed {(time.time() - start_time) / 60:.4f}')

    return overall_score

def w1_distance(taxi_data_train, taxi_data_test, synth_data_all, args, w1_k):
    if not args.check_WD:
        return {k:[0] for k in w1_k}

    w1_d = {k:[] for k in w1_k}
    train_data_np, test_data_np = taxi_data_train.to_numpy(), taxi_data_test.to_numpy()
    for i in range(args.ngen):
        synth_data_df = synth_data_all[i]
        synth_data_np = synth_data_df.to_numpy()
        train_scaled, synth_scaled, _, _, _ = minmax_scale_dummy(train_data_np, synth_data_np, [], divide_by=2)
        _, test_scaled, _, _, _ = minmax_scale_dummy(train_data_np, test_data_np, [], divide_by=2)

        if args.check_WD:
            cprint('overall distance', 'green')
            cur = [0] * 3
            cur[0] = pot.emd2(pot.unif(train_scaled.shape[0]), pot.unif(test_scaled.shape[0]),
                                     M=pot.dist(train_scaled, test_scaled, metric='cityblock')) # train test distance
            cur[1] = pot.emd2(pot.unif(train_scaled.shape[0]), pot.unif(synth_scaled.shape[0]),
                                      M=pot.dist(train_scaled, synth_scaled, metric='cityblock')) # train syn distance
            cur[2] = pot.emd2(pot.unif(test_scaled.shape[0]), pot.unif(synth_scaled.shape[0]),
                                     M=pot.dist(test_scaled, synth_scaled, metric='cityblock')) # test syn distance
            cprint(
                f'Wasserstein distance: train_test {cur[0]:.4f}, train_synth {cur[1]:.4f},test_synth {cur[2]:.4f}',
                'blue')
            for i in range(3):
                w1_d[w1_k[i]].append(cur[i])

    return w1_d


## Below is coverage from https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py

def compute_pairwise_distance(data_x, data_y=None): # Changed to L1 instead of L2 to better handle mixed data
	"""
	Args:
		data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
		data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
	Returns:
		numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
	"""
	if data_y is None:
		data_y = data_x
	dists = sklearn.metrics.pairwise_distances(
		data_x, data_y, metric='cityblock', n_jobs=-1)
	return dists


def get_kth_value(unsorted, k, axis=-1):
	"""
	Args:
		unsorted: numpy.ndarray of any dimensionality.
		k: int
	Returns:
		kth values along the designated axis.
	"""
	indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
	k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
	kth_values = k_smallests.max(axis=axis)
	return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
	"""
	Args:
		input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
		nearest_k: int
	Returns:
		Distances to kth nearest neighbours.
	"""
	distances = compute_pairwise_distance(input_features)
	radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
	return radii


# Automatically finding the best k as per https://openreview.net/pdf?id=1mNssCWt_v
def compute_coverage(real_features, fake_features, nearest_k=None):
	"""
	Computes precision, recall, density, and coverage given two manifolds.

	Args:
		real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
		fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
		nearest_k: int.
	Returns:
		dict of precision, recall, density, and coverage.
	"""

	if nearest_k is None: # we choose k to be the smallest such that we have 95% coverage with real data
		coverage_ = 0
		nearest_k = 1
		while coverage_ < 0.95:
			coverage_ = compute_coverage(real_features, real_features, nearest_k)
			nearest_k += 1

	real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
		real_features, nearest_k)
	distance_real_fake = compute_pairwise_distance(
		real_features, fake_features)

	coverage = (
			distance_real_fake.min(axis=1) <
			real_nearest_neighbour_distances
	).mean()

	return coverage

def privacy_metrics(np_real_scaled, np_fake_scaled, np_holdout=None, metric='euclidean', penc_rr=5, perc_rs=0.01):
    # Computing pair-wise distances between real/training and synthetic
    dist_rf = metrics.pairwise_distances(np_real_scaled, Y=np_fake_scaled, metric=metric, n_jobs=8)
    # Computing pair-wise distances between holdout and synthetic
    dist_hf = metrics.pairwise_distances(np_holdout, Y=np_fake_scaled, metric=metric, n_jobs=8)
    # Computing pair-wise distances within real
    dist_rr = metrics.pairwise_distances(np_real_scaled, Y=None, metric=metric, n_jobs=8)
    # Computing pair-wise distances within synthetic
    dist_ff = metrics.pairwise_distances(np_fake_scaled, Y=None, metric=metric, n_jobs=8)

    # Removes distances of data points to themselves to avoid 0s within real and synthetic
    dist_rr = dist_rr[~np.eye(dist_rr.shape[0], dtype=bool)].reshape(dist_rr.shape[0], -1)
    dist_ff = dist_ff[~np.eye(dist_ff.shape[0], dtype=bool)].reshape(dist_ff.shape[0], -1)

    min_dist_rf = np.min(dist_rf, axis=1)
    min_dist_hf = np.min(dist_hf, axis=1)
    min_dist_rr = np.min(dist_rr, axis=1)
    min_dist_ff = np.min(dist_ff, axis=1)

    perc_rf = np.percentile(min_dist_rf, perc_rs)
    perc_hf = np.percentile(min_dist_hf, perc_rs)
    perc_rr = np.percentile(min_dist_rr, penc_rr)
    perc_ff = np.percentile(min_dist_ff, 5)

    print(f'rf {perc_rf:.4f}, rr {perc_rr:.4f}, ff {perc_ff:.4f}, hf {perc_hf:.4f}')
    print(f'*** ratio: is {perc_rf/perc_hf:.6f}')

    return perc_rf, perc_hf, perc_rf/perc_hf, perc_rs, perc_rr, perc_ff