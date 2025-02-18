import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# export CUDA_VISIBLE_DEVICES=1
#
import setGPU
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer
from ctgan import CTGAN, TVAE
from termcolor import cprint
import ot as pot
from utils import minmax_scale_dummy, set_seed, get_args, get_metadata, to_graph, diff_graph, get_col_idx, make_dataset, get_metadata_CA
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from metrics import sdv_score, compute_coverage, pred_score, privacy_metrics
from torch.utils.tensorboard import SummaryWriter
import torch
import data_util
from ctgan.synthesizers.zonestat import ZoneStatSynthesizer
from TabDDPM.scripts.pipeline import main_fn as tab_ddpm_fn
from TabDDPM.lib.dataset_prep import my_data_prep
from STaSy.stasy import STaSy_model
from CTABGANPlus.ctabgan import CTABGAN

seed = 42
set_seed(seed=seed)

args = get_args()

taxi_datasets = ['green', 'zone', 'yellow']
score_k = ['tr_tr', 'tr_syn', 'tr_te', 'syn_syn', 'syn_tr', 'syn_te']
if args.dwn_base:
    score_k_b = ['tr_tr_b', 'tr_syn_b', 'tr_te_b', 'syn_syn_b', 'syn_tr_b', 'syn_te_b']

keys = ['tr_te', 'tr_syn', 'te_syn']
dispatcher = {'CTGAN': CTGANSynthesizer, 'TVAE': TVAESynthesizer, 'GaussianCopula': GaussianCopulaSynthesizer,
              'CopulaGAN': CopulaGANSynthesizer, 'HCTGAN': CTGANSynthesizer, 'HTVAE': TVAESynthesizer, 'zoneStat':ZoneStatSynthesizer}

method_to_model = {'CTGAN': CTGAN, 'TVAE': TVAE}
MAXNUM = 20000

bz_sample = None

model_path = 'models'
if not os.path.exists(model_path):
    os.mkdir(model_path)

if __name__ == '__main__':
    start_time_all = time.time()

    dwn_model = LinearRegression() if args.dwn_model == 'LR' else LinearSVR() if args.dwn_model == 'SVM' else GradientBoostingRegressor()
    cprint(f'dwn_model is {dwn_model.__class__.__name__}', 'green')

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    f_speed = f'{output_dir}/speed.csv' # save training+sampling speed
    f_dcr = f'{output_dir}/dcr.csv' # save rDCR values
    if not os.path.isfile(f_speed):
        with open(f_speed, 'w') as f:
            f.write('method, num, nexp, minutes\n')
    if not os.path.isfile(f_dcr):
        with open(f_dcr, 'w') as f:
            f.write('method, num, dcr_rs, dcr_hs, rDCR, perc, dcr_rr, dcr_ss\n')

    for num in args.nums:  # number of training samples
        path = args.path
        num_test = num if num < MAXNUM else MAXNUM
        num_load = num + num_test

        if args.dataset in ['california', 'buddy', 'qsar_biodegradation']:
            y_column = 'target'
            Xy, cat_idx, int_idx, cat_y = data_util.data_loader(args.dataset, y_column)
            taxi_data_train, taxi_data_test = train_test_split(Xy, test_size=0.2)
            metadata = get_metadata_CA(taxi_data_train, cat_idx)
            pd_columns = taxi_data_train.columns
        else:
            assert args.dataset in taxi_datasets
            if args.dataset == 'green': # green taxi trips
                taxi_data = data_util.load_green(path, num_load)
            elif args.dataset == 'yellow':
                taxi_data = data_util.load_yellow(path, num_load)
            elif args.dataset == 'zone':
                taxi_data = data_util.load_zone(path, num_load, seed)
            else:
                raise NotImplementedError(f'args.dataset {args.dataset} not implemented')

            cprint(f'{num} samples loaded', 'green')

            taxi_data = data_util.preprocess_tab(taxi_data)
            y_column = args.col_name
            X = taxi_data.loc[:, taxi_data.columns != y_column].reset_index(drop=True)
            y = taxi_data[y_column].reset_index(drop=True)
            cat_idx, int_idx = data_util.get_idx(X.columns)
            Xy = pd.concat([X, y.rename(y_column)], axis=1)
            cat_y = True if y.dtype == 'int32' or y.dtype == 'int64' else False

            test_size = num_test / num_load
            taxi_data_train, taxi_data_test = train_test_split(Xy, test_size=test_size,  random_state=0)

            constraints = None

            metadata = get_metadata(taxi_data_train)
            pd_columns = [col for col in taxi_data_train.columns]
            idx_columns = [str(i) for i in range(taxi_data_train.shape[1])]
            cat_idx, int_idx = get_col_idx(pd_columns)
            if args.dataset == 'zone':
                graph_train = to_graph(taxi_data_train, args)
                graph_test = to_graph(taxi_data_test, args)


        for method in args.methods:  # method: generative model
            use_synthesizer = args.use_synthesizer
            cprint(f'dataset {args.dataset}, method {method}, use_synthesizer {use_synthesizer}, train_base_info {args.train_base_info}, metric_base_info {args.metric_base_info}, num {num}', 'green')

            score_d = {k: [] for k in score_k}
            if args.dwn_base:
                score_d_b = {k: [] for k in score_k_b}
            w1_d = {k: [] for k in keys}
            graphScore = {k: [] for k in keys}
            sdv_d = {k: [] for k in keys}
            cover_d = {k: [] for k in keys}
            privacy_a = []

            for i_nexp in range(args.nexp):
                run_s_base = 'fullInfo'
                run_name_base = f'{num}_{method}_{run_s_base}_{args.seed}'
                run_name = f'{run_name_base}_{time.strftime("%m-%d %H-%M-%S")}'
                writer = SummaryWriter(f'runs/{run_name}')
                writer.add_text(
                    "hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
                )

                model_file = f'{model_path}/{run_name_base}.pt'
                model_args = dict(verbose=args.verbose, writer=writer, epochs=args.epochs,
                                  discriminator_steps=args.d_steps, batch_size=args.batch_size,
                                  discriminator_lr=args.d_lr, generator_lr=args.g_lr)

                start_time = time.time() # start training
                if method == 'CTABGAN':
                    cat_col = [str(i) for i in cat_idx]
                    general_col = [str(i) for i in range(taxi_data_train.shape[1]) if i not in cat_idx + int_idx]
                    int_col = [str(i) for i in int_idx]
                    taxi_data_train.columns = [str(i) for i in range(taxi_data_train.shape[1])]
                    synthesizer = CTABGAN(pd_data=taxi_data_train,
                                          categorical_columns=cat_col,
                                          general_columns=general_col,
                                          integer_columns=int_col)
                    synthesizer.fit()
                    taxi_data_train.columns = pd_columns
                elif method == 'STaSy':
                    num_samples = num if num < MAXNUM else MAXNUM
                    Xy_fake = STaSy_model(taxi_data_train.to_numpy().astype('float'),
                                          categorical_columns=cat_idx,
                                          ordinal_columns=int_idx,
                                          seed=args.nexp,
                                          epochs=10000,
                                          ngen=args.ngen,
                                          activation='elu', layer_type='concatsquash', sde='vesde', lr=0.002,
                                          num_scales=50, num_samples=num_samples, dataset=args.dataset)
                    Xy_fake = Xy_fake.reshape(args.ngen, num_samples, taxi_data_train.shape[1])
                elif method == 'TabDDPM':
                    X_pd = taxi_data_train.loc[:, taxi_data_train.columns != y_column].reset_index(drop=True)
                    y_pd = taxi_data_train[y_column].reset_index(drop=True)

                    pd_columns = list(X_pd.columns) + [y_column]
                    cat_idx, int_idx = get_col_idx(X_pd.columns) # recalculate idx due to reorder of X_pd
                    X_pd.columns = [str(i) for i in range(X_pd.shape[1])]
                    cat_ind = [str(i) for i in cat_idx]
                    noncat_ind = [str(i) for i in range(X_pd.shape[1]) if str(i) not in cat_ind]

                    if y_pd.dtype == 'int32' or y_pd.dtype == 'int64':
                        task_type = 'multiclass'
                    else:
                        task_type = 'regression'
                    my_data_prep(X_pd, y_pd, task=task_type, cat_ind=cat_ind, noncat_ind=noncat_ind)
                    config = 'TabDDPM/config/config.toml'
                    synthetic_data = tab_ddpm_fn(config='TabDDPM/config/config.toml',
                                                 cat_indexes=cat_idx,
                                                 num_classes=len(np.unique(y_pd)) if y_pd.dtype == 'int32' or y_pd.dtype == 'int64' else 0,
                                                 num_samples=taxi_data_train.shape[0],
                                                 num_numerical_features=len(noncat_ind), seed=i_nexp, ngen=args.ngen)
                    Xy_fake = synthetic_data.astype('float')
                    Xy_fake = Xy_fake.reshape(args.ngen, taxi_data_train.shape[0], taxi_data_train.shape[1])  # [ngen, n, p]
                elif use_synthesizer or method not in ['CTGAN', 'TVAE']:
                    model = dispatcher[method](metadata, **model_args)
                    if args.load_model:
                        model = torch.load(model_file)
                        cprint(f'{method} {run_name_base} loaded!', 'green')
                    else:
                        cprint(f'{method} Synthesizer initialized. fit...', 'green')
                        model.fit(taxi_data_train, args=args)
                else:
                    raise NotImplementedError

                writer.close()
                elapsed = (time.time() - start_time) / 60
                cprint(f'{method} fitted successfully. Time elapsed {elapsed:.4f}', 'green')
                with open(f_speed, 'a') as f:
                    f.write( f'{method}, {num}, {i_nexp}, {elapsed:.4f}\n')

                taxi_data_train_mt = taxi_data_train
                taxi_data_test_mt = taxi_data_test
                metadata_mt = metadata

                for i_ngen in range(args.ngen):
                    # get samples
                    if method == 'CTABGAN':
                        synth_data_mt = synthesizer.generate_samples()
                        synth_data_mt.columns = pd_columns
                    elif method == 'STaSy':
                        synth_data_mt = pd.DataFrame(Xy_fake[i_ngen], columns = pd_columns)
                    elif method == 'TabDDPM':
                        synth_data_mt = pd.DataFrame(Xy_fake[i_ngen], columns = pd_columns)
                    else:
                        synth_data_mt = model.sample(taxi_data_train_mt.shape[0], batch_size=bz_sample)

                    # Downstream tasks
                    if args.dataset not in taxi_datasets:
                        args.col_name = y_column
                    cprint(f'Downstream tasks, predicting {args.col_name}...', 'green')
                    start_time = time.time()

                    cur = pred_score(taxi_data_train_mt, synth_data_mt, taxi_data_test_mt, args.col_name, dwn_model)
                    for i_dwn in range(len(score_k)):
                        score_d[score_k[i_dwn]].append(cur[i_dwn])
                        cprint(f'{score_k[i_dwn]} is {cur[i_dwn]*100:.2f}', 'green')

                    end_time = time.time()
                    print(f'Downstream task: Time elapsed {(end_time - start_time) / 60:.4f}')

                    # Graph Similarity
                    cprint(f'Graph Similarity', 'green')
                    cur = [0] * 3
                    if args.dataset == 'zone': # only applicable to zone dataset
                        graph_gen = to_graph(synth_data_mt, args)
                        cur[0] = 1 - diff_graph(graph_train, graph_test, args)
                        cur[1] = 1 - diff_graph(graph_train, graph_gen, args)
                        cur[2] = 1 - diff_graph(graph_test, graph_gen, args)
                    for i_graphScore in range(3):
                        graphScore[keys[i_graphScore]].append(cur[i_graphScore])

                    # SDV score
                    cprint('quality report:', 'green')
                    start_time = time.time()

                    cur = [0] * 3
                    if not args.no_sdv:
                        cur[0] = sdv_score(taxi_data_train_mt, taxi_data_test_mt, metadata_mt)
                        cur[1] = sdv_score(taxi_data_train_mt, synth_data_mt, metadata_mt)
                        cur[2] = sdv_score(taxi_data_test_mt, synth_data_mt, metadata_mt)
                    for i_sdv in range(3):
                        sdv_d[keys[i_sdv]].append(cur[i_sdv])

                    end_time = time.time()
                    print(f'SDV score: Time elapsed {(end_time - start_time)/ 60:.4f}')

                    if num > MAXNUM: # reduce dataset size for fast computation
                        taxi_data_train_mt, taxi_data_test_mt = taxi_data_train_mt.sample(min(MAXNUM, len(taxi_data_train_mt))), taxi_data_test_mt.sample(min(MAXNUM, len(taxi_data_test_mt)))
                        synth_data_mt = synth_data_mt.sample(min(MAXNUM, len(synth_data_mt)))
                    taxi_data_train_mt_np = taxi_data_train_mt.to_numpy()
                    taxi_data_test_mt_np = taxi_data_test_mt.to_numpy()
                    synth_data_mt_np = synth_data_mt.to_numpy().astype('float')
                    train_scaled, synth_scaled, _, _, _ = minmax_scale_dummy(taxi_data_train_mt_np, synth_data_mt_np, [], divide_by=2)
                    _, test_scaled, _, _, _ = minmax_scale_dummy(taxi_data_train_mt_np, taxi_data_test_mt_np, [], divide_by=2)

                    # Wasserstein distance
                    cprint('Wasserstein distance...', 'green')
                    start_time = time.time()

                    cur = [0] * 3
                    if not args.no_wd:
                        cur[0] = pot.emd2(pot.unif(train_scaled.shape[0]), pot.unif(test_scaled.shape[0]),
                                          M=pot.dist(train_scaled, test_scaled, metric='cityblock'))  # train test distance
                        cur[1] = pot.emd2(pot.unif(train_scaled.shape[0]), pot.unif(synth_scaled.shape[0]),
                                          M=pot.dist(train_scaled, synth_scaled, metric='cityblock'))  # train syn distance
                        cur[2] = pot.emd2(pot.unif(test_scaled.shape[0]), pot.unif(synth_scaled.shape[0]),
                                      M=pot.dist(test_scaled, synth_scaled, metric='cityblock'))  # test syn distance

                    end_time = time.time()
                    cprint(f'Wasserstein distance: {cur[0]:.4f}, {cur[1]:.4f}, {cur[2]:.4f}. {min(num, MAXNUM)} ({num} training) samples, Time elapsed {(end_time - start_time)/ 60:.4f}',
                        'blue')
                    for i_w1 in range(3):
                        w1_d[keys[i_w1]].append(cur[i_w1])

                    # diversity
                    cprint(f'coverage test...', 'green')
                    start_time = time.time()

                    cur = [0] * 3
                    if not args.no_cov:
                        cur[0] = compute_coverage(train_scaled, test_scaled)
                        cur[1] = compute_coverage(train_scaled, synth_scaled)
                        cur[2] = compute_coverage(test_scaled, synth_scaled)

                    end_time = time.time()
                    cprint(f'tr_te {cur[0]:.4f}, tr_syn {cur[1]:.4f}, te_syn {cur[2]:.4f}. {min(num, MAXNUM)} ({num} training) samples, Time elapsed {(end_time - start_time)/ 60:.4f}')
                    for i_cov in range(3):
                        cover_d[keys[i_cov]].append(cur[i_cov])

                    # Privacy test
                    cprint(f'Privacy test...', 'green')
                    start_time = time.time()

                    n_prv_col = 6 # columns: dcr_rs, dcr_hs,rDCR, perc, dcr_rr, dcr_ss
                    cur = [0] * n_prv_col
                    if not args.no_prv:
                        for perc in args.percs: # calculate rDCR for different percentile/alpha
                            cur = privacy_metrics(train_scaled, synth_scaled, test_scaled, perc_rs=perc)
                            with open(f_dcr, 'a') as f:
                                cur_s = f'{method}, {num}'
                                for i in range(n_prv_col):
                                    cur_s += f', {cur[i]:.4f}'
                                cur_s += '\n'
                                f.write(cur_s)
                    end_time = time.time()

                    cprint(f'{min(num, MAXNUM)} ({num} training) samples, Time elapsed {(end_time - start_time)/ 60:.4f}\n{cur}')
                    privacy_a.append(cur)

            privacy_a = np.array(privacy_a)
            assert privacy_a.shape == (args.nexp * args.ngen, n_prv_col)
            cprint(f'sdv score {sdv_d}\nw1 distance {w1_d}\ncoverage {cover_d}', 'green')

            # save results
            csv_str = ''
            s_base = ''
            output_dir, output_file = 'output', f'{args.outfile}{s_base}_{"syn" if args.use_synthesizer else "nosyn"}_{args.col_name}_{args.seed}_{args.dataset}.csv'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            f_path = f'{output_dir}/{output_file}'
            s_dwn = f'dwn_tr_tr, dwn_tr_syn, dwn_tr_te, dwn_syn_syn, dwn_syn_tr, dwn_syn_te, '
            if args.dwn_base:
                s_dwn += f'b_tr_tr, b_tr_syn, b_tr_te, b_syn_syn, b_syn_tr, b_syn_te, '
            s_gs = f'G_tr_te, G_tr_syn, G_te_syn, '
            s_privacy = f', dcr_rs, dcr_hs, rDCR, perc, dcr_rr, dcr_ss'
            if not os.path.isfile(f_path):
                if args.more_cols:
                    csv_str = f'num, model, {s_dwn}{s_gs}sdv_tr_te, sdv_tr_syn, sdv_te_syn, w1_tr_te, w1_tr_syn, w1_te_syn, cov_tr_te, cov_tr_syn, cov_te_syn{s_privacy}\n'
                else:
                    csv_str = f'num, model, dwn_syn_syn, dwn_syn_tr, dwn_syn_te, sdv_tr_te, sdv_tr_syn, w1_tr_te, w1_tr_syn, cov_tr_te, cov_tr_syn, cov_te_syn\n'
            cur_model_ = f'{method}'
            csv_str += f'{num}, {cur_model_}, '
            if args.more_cols:
                for i_s in score_k:
                    csv_str += f'{np.mean(score_d[i_s]) * 100:.2f} ({np.std(score_d[i_s]) * 100:.2f}), '
                if args.dwn_base:
                    for i_s in score_k_b:
                        csv_str += f'{np.mean(score_d_b[i_s]) * 100:.2f} ({np.std(score_d_b[i_s]) * 100:.2f}), '
                for key in keys:
                    csv_str += f'{np.mean(graphScore[key]) * 100:.2f} ({np.std(graphScore[key]) * 100:.2f}), '
                for i_w in keys:
                    csv_str += f'{np.mean(sdv_d[i_w]) * 100:.2f} ({np.std(sdv_d[i_w]) * 100:.2f}), '
                for i_w in keys:
                    csv_str += f'{np.mean(w1_d[i_w]):.4f} ({np.std(w1_d[i_w]):.4f}), '
            else:
                for i_s in score_k[3:6]:
                    csv_str += f'{np.mean(score_d[i_s]) * 100:.2f} ({np.std(score_d[i_s]) * 100:.2f}), '
                for i_sdv in keys[:-1]:
                    csv_str += f'{np.mean(sdv_d[i_sdv]) * 100:.2f} ({np.std(sdv_d[i_sdv]) * 100:.2f}), '
                for i_w in keys[:-1]:
                    csv_str += f'{np.mean(w1_d[i_w]):.4f} ({np.std(w1_d[i_w]):.4f}), '
            for i_cov in keys:
                csv_str += f'{np.mean(cover_d[i_cov]) * 100:.2f} ({np.std(cover_d[i_cov]) * 100:.2f}), '
            prv_a_m, prv_a_std = np.mean(privacy_a, axis=0), np.std(privacy_a, axis=0)
            for i_prv in range(len(prv_a_m)):
                csv_str += f'{prv_a_m[i_prv]:.3f} ({prv_a_std[i_prv]:.3f}), '
            csv_str += f'\n'
            cprint(csv_str, 'green')
            with open(f_path, 'a+') as f:  #$ write early??
                f.write(csv_str)

    end_time_all = time.time()
    cprint(f'Total time elapsed {(end_time - start_time_all)/ 60:.4f}', 'green')

