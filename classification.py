# extract statisctical values of all subcarriers and average them
# try this to select subcarriers -> weighting: w_i = power(0.1–0.7 Hz band) / power(out-of-band) before filtering

import os, pickle, warnings
import scipy.io as io
import numpy as np
from scipy.signal import ellip, filtfilt, find_peaks
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pylab as plt
import matplotlib.pylab as plt

warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.linear_model._sag')
SAVE_PATH = 'ml_preprocessed_dataset_v3_ws11.pkl'
DATA_DIRS = "D:/UT_EE/3_year/module_12_BSc_Thesis/data/ba_breathing_dataset/S1/W1/train"
FS = 20.0
NYQ = 0.5*FS
RP = 0.1
RS = 40
WP = 0.1
WS = 1.1
TOP_K = 50
WIN_LEN = 300
STRIDE = 150
TRIM = 20
def plot_metric_bars(metric_scores, model_names):
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    label_font = 16
    tick_font = 14
    legend_font = 14
    bar_value_font = 9

    plt.figure(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        values = [metric_scores[model_name][metric] for metric in metrics]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = plt.bar(x + offset, values, width, label=model_name)

        plt.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=bar_value_font)

    plt.ylabel('Score (%)', fontsize=label_font)
    plt.xticks(x, metrics, fontsize=tick_font)
    plt.yticks(fontsize=tick_font)
    plt.legend(fontsize=legend_font)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.show()

def ellip_bp(data):
    bh, ah = ellip(8, RP, RS, WP/NYQ, btype='high')
    bl, al = ellip(5, RP, RS, WS/NYQ, btype='low')
    return filtfilt(bh, ah, filtfilt(bl, al, data))

def feature_extract(x):
    x_var = np.mean(np.var(x, axis=1))
    x_sk = np.mean(skew(x, axis=1))
    x_kurt = np.mean(kurtosis(x, axis=1))
    zcr = np.mean(np.diff(np.sign(x), axis=1) != 0, axis=1)
    x_zcr = np.mean(zcr)
    acf_peaks = []
    for row in x:
        acf = np.correlate(row, row, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf/acf[0] if acf[0] > 0 else acf
        peaks, _ = find_peaks(acf)
        valid = [p for p in peaks
                 if int(FS/0.7) <= p <= int(FS/0.1)]
        acf_peaks.append(acf[valid[0]] if valid else 0.0)
    x_acf = np.mean(acf_peaks)

    return np.array([x_var, x_sk, x_kurt, x_zcr, x_acf])

def process_channel(csi_matrix):
    # magnitude
    mag = np.abs(csi_matrix)
    mag = ellip_bp(mag)
    mag = mag - np.mean(mag, axis=1, keepdims=True)
    feats_amp = feature_extract(mag)

    # phase
    ph = np.angle(csi_matrix)
    ph = np.unwrap(ph, axis=1)
    ph = ellip_bp(ph)
    ph = ph - np.mean(ph, axis=1, keepdims=True)
    feats_ph = feature_extract(ph)

    return feats_amp, feats_ph

if os.path.exists(SAVE_PATH):
    print(f"Loading preprocessed dataset from {SAVE_PATH} ...")
    with open(SAVE_PATH, 'rb') as f:
        dataset = pickle.load(f)
        dataset_both_good = [
            d for d in dataset
            if d['features'][-2] == 1.0 and d['features'][-1] == 1.0
        ]
    print(f"Both-good windows: {len(dataset_both_good)} / {len(dataset)}")
    print(f"Loaded {len(dataset)} windows.\n")
else:
    dataset = []
    file_counter = 0
    print(f"Extracting (TRIM={TRIM}, WIN={WIN_LEN}, STRIDE={STRIDE}) ...")

    for file in [f for f in os.listdir(DATA_DIRS) if f.endswith('_dataset.mat')]:
        try:
            full_path = os.path.join(DATA_DIRS, file)
            mat_data = io.loadmat(full_path)
            meta = mat_data['metadata'][0, 0]
            key = meta['key'][0]
            activity = key.split('_')[3]
            ori = meta['orientation'][0]
            label = 1 if meta['breath_condition'][0] == 'BR' else 0
            off_c, off_s = float(meta['time_offset_comm'][0, 0]), float(meta['time_offset_sens'][0, 0])
            bad_comm = (off_c < 1e5) or (min(off_c % 5e6, 5e6 - (off_c % 5e6)) > 1e6)
            bad_sens = (off_s < 1e5) or (min(off_s % 5e6, 5e6 - (off_s % 5e6)) > 1e6)

            T_raw = min(mat_data['csi_comm_seg'].shape[1], mat_data['csi_sens_seg'].shape[1])
            t_s, t_e = TRIM, T_raw - TRIM
            if t_e - t_s < WIN_LEN:
                print(f"  Skip {file}: too short ({t_e - t_s} < {WIN_LEN})")
                continue

            csi_comm = mat_data['csi_comm_seg'][:, t_s:t_e]
            csi_sens = mat_data['csi_sens_seg'][:, t_s:t_e]
            T = t_e - t_s
            
            for start_idx in range(0, T - WIN_LEN + 1, STRIDE):
                end_idx = start_idx + WIN_LEN
                if not bad_comm:
                    comm_amp, comm_ph = process_channel(csi_comm[:, start_idx:end_idx])
                else:
                    comm_amp = comm_ph = np.zeros(5)
                if not bad_sens:
                    sens_amp, sens_ph = process_channel(csi_sens[:, start_idx:end_idx])
                else:
                    sens_amp = sens_ph = np.zeros(5)
                comm_valid = 0.0 if bad_comm else 1.0
                sens_valid  = 0.0 if bad_sens  else 1.0
                fused_features = np.concatenate([
                    comm_amp, comm_ph, sens_amp, sens_ph,
                    [comm_valid, sens_valid]
                ])
                dataset.append({
                    'group_id': file_counter,
                    'activity': activity,
                    'ori': ori,
                    'label': label,
                    'features': fused_features
                })
            file_counter += 1
        except Exception as e:
            pass
    print(f"Processed {file_counter} files into {len(dataset)} windows.\n")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {SAVE_PATH}")

param_grid_svm = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 5, 10, 50],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}
param_grid_rf = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
param_grid_hgb = {
    'max_iter':          [100, 200, 300],
    'max_depth':         [3, 5, None],
    'learning_rate':     [0.05, 0.1, 0.2],
    'min_samples_leaf':  [5, 10, 20],
    'l2_regularization': [0.0, 0.1, 1.0],
}
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0, 0.5, 1],     
    'solver': ['saga'],
    'max_iter': [10000, 20000]         
}

def run_evaluation(data_subset, title):
    print("=" * 60)
    print(f" INVESTIGATION: {title}")
    print("=" * 60)

    X = np.array([d['features'] for d in data_subset])
    y = np.array([d['label'] for d in data_subset])
    groups = np.array([d['group_id'] for d in data_subset])

    if len(np.unique(y)) < 2:
        print("Not enough class diversity in this subset to train.")
        return None

    print(f"Total Windows: {len(y)} | BR (1): {np.sum(y==1)} | NB (0): {np.sum(y==0)}")

    gkf = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)

    svm_accs, svm_precs, svm_recs, svm_aucs = [], [], [], []
    rf_accs, rf_precs, rf_recs, rf_aucs = [], [], [], []
    hg_accs, hg_precs, hg_recs, hg_aucs = [], [], [], []
    lr_accs, lr_precs, lr_recs, lr_aucs = [], [], [], []

    rf_importances = np.zeros(X.shape[1])
    fold = 1

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]

        train_br = np.mean(y_train == 1) * 100
        train_nb = np.mean(y_train == 0) * 100
        val_br = np.mean(y_val == 1) * 100
        val_nb = np.mean(y_val == 0) * 100

        print(f"\nFold {fold}")
        print(f"Train: BR={train_br:.1f}% NB={train_nb:.1f}% ({len(y_train)} windows)")
        print(f" Val : BR={val_br:.1f}% NB={val_nb:.1f}% ({len(y_val)} windows)")

        # Scaling: fit only on both-good windows
        both_good = (X_train[:, -2] == 1.0) & (X_train[:, -1] == 1.0)
        fit_data  = X_train[both_good] if both_good.any() else X_train
        scaler = StandardScaler().fit(fit_data)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)

        inner_cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X_train_s, y_train, groups=groups_train))

        # SVM
        svm_grid = GridSearchCV(SVC(random_state=42,probability=True), param_grid_svm,
                                cv=inner_cv, scoring='accuracy', n_jobs=-1, verbose=0)
        svm_grid.fit(X_train_s, y_train)
        svm_pred = svm_grid.best_estimator_.predict(X_val_s)
        svm_acc = accuracy_score(y_val, svm_pred)
        svm_accs.append(svm_acc)
        svm_prob = svm_grid.best_estimator_.predict_proba(X_val_s)[:, 1]
        svm_prec = precision_score(y_val, svm_pred)
        svm_precs.append(svm_prec)
        svm_rec = recall_score(y_val, svm_pred)
        svm_recs.append(svm_rec)
        svm_auc = roc_auc_score(y_val, svm_prob)
        svm_aucs.append(svm_auc)

        print(f"SVM best params: {svm_grid.best_params_}")
        print(f"SVM accuracy: {svm_acc*100:.1f}%  precision: {svm_prec:.3f}  recall: {svm_rec:.3f}  AUC: {svm_auc:.3f}")

        # RF
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf,
                               cv=inner_cv, scoring='accuracy', n_jobs=-1, verbose=0)
        rf_grid.fit(X_train_s, y_train)
        rf_pred = rf_grid.best_estimator_.predict(X_val_s)
        rf_acc = accuracy_score(y_val, rf_pred)
        rf_accs.append(rf_acc)
        rf_prec = precision_score(y_val, rf_pred)
        rf_precs.append(rf_prec)
        rf_rec = recall_score(y_val, rf_pred)
        rf_recs.append(rf_rec)
        rf_auc = roc_auc_score(y_val, rf_pred)
        rf_aucs.append(rf_auc)
        rf_importances += rf_grid.best_estimator_.feature_importances_

        print(f"RF best params: {rf_grid.best_params_}")
        print(f"RF accuracy: {rf_acc*100:.1f}%  precision: {rf_prec:.3f}  recall: {rf_rec:.3f}  AUC: {rf_auc:.3f}")

        # RF
        hg_grid = GridSearchCV(HistGradientBoostingClassifier(random_state=42),
                             param_grid_hgb, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        hg_grid.fit(X_train_s, y_train)
        hg_pred = hg_grid.best_estimator_.predict(X_val_s)
        hg_acc = accuracy_score(y_val, hg_pred)
        hg_accs.append(hg_acc)
        hg_prec = precision_score(y_val, hg_pred)
        hg_precs.append(hg_prec)
        hg_rec = recall_score(y_val, hg_pred)
        hg_recs.append(hg_rec)
        hg_auc = roc_auc_score(y_val, hg_pred)
        hg_aucs.append(hg_auc)

        print(f"HGB best params: {hg_grid.best_params_}")
        print(f"HGB accuracy: {hg_acc*100:.1f}%  precision: {hg_prec:.3f}  recall: {hg_rec:.3f}  AUC: {hg_auc:.3f}")

        lr_grid = GridSearchCV(LogisticRegression(random_state=42),
                               param_grid_lr,cv=inner_cv,scoring='accuracy',n_jobs=-1,verbose=0)
        lr_grid.fit(X_train_s, y_train)
        best_lr = lr_grid.best_estimator_
        lr_pred = best_lr.predict(X_val_s)
        lr_acc = accuracy_score(y_val, lr_pred)
        lr_accs.append(lr_acc)
        lr_prec = precision_score(y_val, lr_pred)
        lr_precs.append(lr_prec)
        lr_rec = recall_score(y_val, lr_pred)
        lr_recs.append(lr_rec)
        lr_auc = roc_auc_score(y_val, lr_pred)
        lr_aucs.append(lr_auc)

        print(f"Logistic Regression best params: {lr_grid.best_params_}")
        print(f"Logistic Regression accuracy: {lr_acc*100:.1f}%  precision: {lr_prec:.3f}  recall: {lr_rec:.3f}  AUC: {lr_auc:.3f}")

        fold += 1

    print(f"\nAverage SVM Accuracy : {np.mean(svm_accs)*100:.1f}%  "
      f"Precision: {np.mean(svm_precs):.3f}  "
      f"Recall: {np.mean(svm_recs):.3f}  "
      f"AUC: {np.mean(svm_aucs):.3f}")
    print(f"Average RF Accuracy : {np.mean(rf_accs)*100:.1f}%  "
      f"Precision: {np.mean(rf_precs):.3f}  "
      f"Recall: {np.mean(rf_recs):.3f}  "
      f"AUC: {np.mean(rf_aucs):.3f}")
    print(f"Average HGB Accuracy : {np.mean(hg_accs)*100:.1f}%  "
      f"Precision: {np.mean(hg_precs):.3f}  "
      f"Recall: {np.mean(hg_recs):.3f}  "
      f"AUC: {np.mean(hg_aucs):.3f}")
    print(f"Average LR Accuracy : {np.mean(lr_accs)*100:.1f}%  "
      f"Precision: {np.mean(lr_precs):.3f}  "
      f"Recall: {np.mean(lr_recs):.3f}  "
      f"AUC: {np.mean(lr_aucs):.3f}")

    metric_scores = {
        'SVM': {
            'Accuracy': np.mean(svm_accs) * 100,
            'Precision': np.mean(svm_precs) * 100,
            'Recall': np.mean(svm_recs) * 100,
            'AUC': np.mean(svm_aucs) * 100
        },
        'RF': {
            'Accuracy': np.mean(rf_accs) * 100,
            'Precision': np.mean(rf_precs) * 100,
            'Recall': np.mean(rf_recs) * 100,
            'AUC': np.mean(rf_aucs) * 100
        },
        'HGB': {
            'Accuracy': np.mean(hg_accs) * 100,
            'Precision': np.mean(hg_precs) * 100,
            'Recall': np.mean(hg_recs) * 100,
            'AUC': np.mean(hg_aucs) * 100
        },
        'LR': {
            'Accuracy': np.mean(lr_accs) * 100,
            'Precision': np.mean(lr_precs) * 100,
            'Recall': np.mean(lr_recs) * 100,
            'AUC': np.mean(lr_aucs) * 100
        }
    }
    plot_metric_bars(metric_scores, ['SVM', 'RF', 'HGB', 'LR'])

def make_comm_only_both_good_dataset(dataset):
    comm_only = []

    for d in dataset:
        features = d['features']
        comm_valid = features[-2]
        sens_valid = features[-1]
        if not (comm_valid == 1.0 and sens_valid == 1.0):
            continue
        new_d = d.copy()
        new_d['features'] = features[0:10]
        comm_only.append(new_d)
    return comm_only


def make_sens_only_both_good_dataset(dataset):
    sens_only = []
    for d in dataset:
        features = d['features']
        comm_valid = features[-2]
        sens_valid = features[-1]
        if not (comm_valid == 1.0 and sens_valid == 1.0):
            continue
        new_d = d.copy()
        new_d['features'] = features[10:20]
        sens_only.append(new_d)
    return sens_only


#stationary_subset = [d for d in sens_only_dataset if d['ori'] in ['O1','O3']]
#stationary_subset = [d for d in dataset_both_good if d['activity'] in ['A1','A2','A3'] and d['ori'] in ['O1']]
#run_evaluation(stationary_subset, "ALL ACTIVITIES - BOTH CHANNELS GOOD")
#run_evaluation(dataset, "ALL ACTIVITIES")
stationary_subset = [d for d in dataset if d['activity'] in ['A1','A3']]
run_evaluation(stationary_subset, "ALL ACTIVITIES")

comm_only_dataset = make_comm_only_both_good_dataset(stationary_subset)
sens_only_dataset = make_sens_only_both_good_dataset(stationary_subset)

#stationary_subset = [d for d in sens_only_dataset if d['ori'] in ['O4']]

run_evaluation(comm_only_dataset, "COMM ONLY - amp + phase")
run_evaluation(sens_only_dataset, "SENS ONLY - amp + phase")