import os, pickle, torch
import matplotlib
matplotlib.use('Agg')
import scipy.io as io
import scipy.fft as fft
import numpy as np
from scipy.signal import ellipord, ellip, filtfilt
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

DATA_DIR  = "D:/UT_EE/3_year/module_12_BSc_Thesis/data/ba_breathing_dataset/S1/W1/train"
SAVE_PATH = "new_w11.pkl"

WIN_LEN = 64
STRIDE = 32
FS  = 20.0
NYQ = FS/2.0
RP = 0.1
RS = 40
N = 8064
D = 2
SUB_D = 252
K = N//SUB_D
KV = 50
T_START = 20
T_END = 641
L = T_END - T_START
L_DS = len(range(0, L, D))
OFFSET_MIN  = 1e5;OFFSET_STEP = 5e6;OFFSET_TOL  = 1e6
DROP = 0.5
EPOCHS = 100
N_SPLITS = 5
N_PLOTS_PER_FOLD = 8
BATCH_SIZE = 32
LR = 1e-2
PURGE_GAP_WINDOWS = int(np.ceil(WIN_LEN / STRIDE)) - 1

class CSIDATASET(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        x = self.X[index,:,:]
        y = self.Y[index,:] 
        return x, y
    
class MSEWithOutOfBandLoss(nn.Module):
    def __init__(
        self,fs=10.0,low_hz=0.13,
        high_hz=0.5,lambda_oob=0.2,
        use_window=True,eps=1e-8,):
        super().__init__()
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.lambda_oob = lambda_oob
        self.use_window = use_window
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target, return_parts=False):
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"Expected pred and target to be [batch, time], "
                f"got pred={pred.shape}, target={target.shape}"
            )

        mse_loss = self.mse(pred, target)
        batch_size, T = pred.shape
        device = pred.device

        if self.use_window:
            window = torch.hann_window(T, device=device).unsqueeze(0)
            pred_fft_input = pred * window
        else:
            pred_fft_input = pred

        # FFT over time dimension
        pred_fft = torch.fft.rfft(pred_fft_input, dim=1)
        power = torch.abs(pred_fft) ** 2

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(device)

        in_band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        out_band = ~in_band

        out_band_power = power[:, out_band].sum(dim=1)
        total_power = power.sum(dim=1) + self.eps

        # normalize
        oob_loss = (out_band_power / total_power).mean()

        total_loss = mse_loss + self.lambda_oob * oob_loss

        if return_parts:
            return total_loss, {
                "total": total_loss.detach(),
                "mse": mse_loss.detach(),
                "oob": oob_loss.detach(),
            }

        return total_loss

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=160, out_channels=40, kernel_size=17, padding=8)
        self.bn1 = nn.BatchNorm1d(40)
        self.conv2 = nn.Conv1d(in_channels=40, out_channels=20, kernel_size=17, padding=8)
        self.bn2 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(in_features=20, out_features=10)
        self.fc = nn.Linear(in_features=10, out_features=1)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc(x)
        return x.squeeze(-1)

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=160, hidden_size=30, num_layers=2, dropout=0.9, batch_first=True)
        self.fc = nn.Linear(in_features=30, out_features=1)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x,_ = self.lstm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x.squeeze(-1)

n_hp, wn_hp = ellipord(0.1/NYQ, 0.08/NYQ, RP, RS)
bh, ah = ellip(n_hp, RP, RS, wn_hp, btype='high')
n_lp, wn_lp = ellipord(0.36 / NYQ, 0.42 / NYQ, RP, RS)
bl, al = ellip(n_lp, RP, RS, wn_lp, btype='low')

def process_channel(data):
    reshaped_data = data.reshape(K, SUB_D, -1)
    csi = np.mean(reshaped_data, axis=1)
    raw_features = []

    # phase
    ph = np.unwrap(np.angle(csi), axis=1)
    ph_bp = filtfilt(bl, al, filtfilt(bh, ah, ph, axis=1), axis=1)
    ph_d = peak_norm(ph_bp)[:, ::D]
    raw_features.append(ph_d)

    # amplitude
    mag0 = np.abs(csi)
    mag = filtfilt(bl, al, filtfilt(bh, ah, mag0, axis=1), axis=1)
    mag_d = peak_norm(mag)[:, ::D]
    raw_features.append(mag_d)

    # real
    re0 = np.real(csi)
    re = filtfilt(bl, al, filtfilt(bh, ah, re0, axis=1), axis=1)
    re_d = peak_norm(re)[:, ::D]
    raw_features.append(re_d)

    # imag
    im0 = np.imag(csi)
    im = filtfilt(bl, al, filtfilt(bh, ah, im0, axis=1), axis=1)
    im_d = peak_norm(im)[:, ::D]
    raw_features.append(im_d)

    # CIR
    cir = fft.ifft(csi, axis=0)
    cirph0 = np.unwrap(np.angle(cir), axis=1)
    cirph = filtfilt(bl, al, filtfilt(bh, ah, cirph0, axis=1), axis=1)
    cirph_d = peak_norm(cirph)[:, ::D]
    raw_features.append(cirph_d)

    feature_matrix_raw = np.concatenate(raw_features, axis=0)
    return feature_matrix_raw

def peak_norm(x, axis=1, eps=1e-8):
    x = x - np.mean(x, axis=axis, keepdims=True)
    scale = np.max(np.abs(x), axis=axis, keepdims=True)
    scale[scale < eps] = 1.0
    return x / scale

def process_label(mocap_raw):
    y = mocap_raw[T_START:T_END]
    y = filtfilt(bh, ah, y)
    y = filtfilt(bl, al, y)
    y = y[::D]
    y = (y - np.mean(y)) / np.max(np.abs(y))
    return y

def is_bad_channel(offset):
    off = float(offset)
    return (off < OFFSET_MIN) or \
           (min(off % OFFSET_STEP, OFFSET_STEP - off % OFFSET_STEP) > OFFSET_TOL)

# MAIN DATASET LOOP
if os.path.exists(SAVE_PATH):
    print(f"Loading preprocessed dataset from {SAVE_PATH} ...")
    with open(SAVE_PATH, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset)} samples.")
else:
    dataset = []
    file_counter = 0
    mat_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('_dataset.mat'))
    print(f"Found {len(mat_files)} .mat files. Skipping NB measurements ...\n")

    for fname in mat_files:
        mat  = io.loadmat(os.path.join(DATA_DIR, fname))
        meta = mat['metadata'][0, 0]
        if meta['breath_condition'][0] == 'NB':
            continue
        
        key = meta['key'][0]
        activity = key.split('_')[3]
        ori = meta['orientation'][0]
        bad_comm = is_bad_channel(meta['time_offset_comm'][0, 0])
        bad_sens = is_bad_channel(meta['time_offset_sens'][0, 0])

        csi_comm = mat['csi_comm_seg'][:, T_START:T_END]   
        csi_sens = mat['csi_sens_seg'][:, T_START:T_END]
        mocap_raw = mat['mocap_breathing'].squeeze()
        if np.isnan(mocap_raw).any():
            continue
        label_full = process_label(mocap_raw)

        if not bad_comm:
            features_comm = process_channel(csi_comm)
        else:
            continue
        
        L_DS = len(label_full) 

        window_count = 0
        for start_idx in range(0, L_DS - WIN_LEN + 1, STRIDE):
            end_idx = start_idx + WIN_LEN
            
            features_win = features_comm[:, start_idx:end_idx]
            label_win = label_full[start_idx:end_idx]
            features_win_n = peak_norm(features_win, axis=1)
            label_win_n = peak_norm(label_win, axis=0)
            dataset.append({
                'group_id':     file_counter,
                'activity':     activity,
                'ori':          ori,
                'comm_valid':   not bad_comm,
                'sens_valid':   not bad_sens,
                'window_index': window_count,
                'window_start': start_idx,
                'window_end':   end_idx,
                'source_file':   fname,
                'signal':       features_win_n,
                'label':        label_win_n,
            })
            window_count += 1

        n_ch = (not bad_comm) + (not bad_sens)
        print(f"  [{file_counter:3d}] {fname} | Windows created: {window_count} | L_DS={L_DS}")
        
        file_counter += 1

    print(f"\nProcessed {file_counter} files → Total generated windows: {len(dataset)}.")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved to {SAVE_PATH}")

def get_window_bounds(data_subset, groups_raw):
    starts = []
    ends = []
    local_counts = {}

    for d, group in zip(data_subset, groups_raw):
        local_idx = local_counts.get(group, 0)

        if 'window_start' in d and 'window_end' in d:
            start = int(d['window_start'])
            end = int(d['window_end'])
        else:
            start = local_idx * STRIDE
            end = start + WIN_LEN

        starts.append(start)
        ends.append(end)
        local_counts[group] = local_idx + 1

    return np.asarray(starts), np.asarray(ends)

def intervals_overlap(starts_a, ends_a, starts_b, ends_b):
    return (starts_a[:, None] < ends_b[None, :]) & (ends_a[:, None] > starts_b[None, :])

def assert_no_train_val_overlap(train_idx, val_idx, groups, starts, ends):
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    if not val_groups.issubset(train_groups):
        missing = sorted(val_groups - train_groups)
        raise AssertionError(f"Validation contains groups with no training windows: {missing}")

    for group in sorted(val_groups):
        tr = train_idx[groups[train_idx] == group]
        va = val_idx[groups[val_idx] == group]
        if len(tr) == 0 or len(va) == 0:
            raise AssertionError(f"Group {group} is not present in both train and validation.")
        if np.any(intervals_overlap(starts[tr], ends[tr], starts[va], ends[va])):
            raise AssertionError(f"Found train/validation window overlap in group {group}.")

def make_purged_intragroup_splits(groups, starts, ends, n_splits=N_SPLITS):
    groups = np.asarray(groups)
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    unique_groups = np.unique(groups)

    splits = []
    for fold_id in range(n_splits):
        train_parts = []
        val_parts = []
        n_purged = 0
        n_skipped_groups = 0

        for group in unique_groups:
            group_idx = np.where(groups == group)[0]
            if len(group_idx) < 2:
                n_skipped_groups += 1
                continue

            group_idx = group_idx[np.argsort(starts[group_idx])]
            local_blocks = np.array_split(np.arange(len(group_idx)), n_splits)
            val_local = local_blocks[fold_id]

            if len(val_local) == 0:
                n_skipped_groups += 1
                continue

            val_idx_group = group_idx[val_local]
            val_starts = starts[val_idx_group]
            val_ends = ends[val_idx_group]

            train_idx_group = []
            for local_pos, idx in enumerate(group_idx):
                if local_pos in val_local:
                    continue

                overlaps_val = np.any((starts[idx] < val_ends) & (ends[idx] > val_starts))
                if overlaps_val:
                    n_purged += 1
                else:
                    train_idx_group.append(idx)

            if len(train_idx_group) == 0:
                n_skipped_groups += 1
                continue

            train_parts.append(np.asarray(train_idx_group, dtype=int))
            val_parts.append(np.asarray(val_idx_group, dtype=int))

        if len(train_parts) == 0 or len(val_parts) == 0:
            raise ValueError(
                f"Fold {fold_id + 1} is empty after purging. "
                "Use fewer splits, shorter WIN_LEN, larger STRIDE, or longer measurements."
            )

        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
        assert_no_train_val_overlap(train_idx, val_idx, groups, starts, ends)
        splits.append((train_idx, val_idx, n_purged, n_skipped_groups))

    return splits

def run_evaluation_cnn(data_subset, title):
    print("=" * 60)
    print(f" INVESTIGATION: {title}")
    print("=" * 60)

    PLOT_DIR = os.path.join("plots", title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", ""))
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_raw = np.array([d['signal'] for d in data_subset])
    y_raw = np.array([d['label'] for d in data_subset])
    groups_raw = np.array([d['group_id'] for d in data_subset])
    starts_raw, ends_raw = get_window_bounds(data_subset, groups_raw)

    nan_in_X = np.isnan(X_raw).any(axis=(1, 2)) | np.isinf(X_raw).any(axis=(1, 2))
    nan_in_y = np.isnan(y_raw).any(axis=1) | np.isinf(y_raw).any(axis=1)
    valid_mask = ~(nan_in_X | nan_in_y)

    X = X_raw[valid_mask]
    y = y_raw[valid_mask]
    groups = groups_raw[valid_mask]
    starts = starts_raw[valid_mask]
    ends = ends_raw[valid_mask]

    n_dropped = len(data_subset) - len(y)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} windows out of {len(data_subset)} due to NaN/Inf values.")

    print(f"Cleaned Windows: {len(y)} | Feature Shape: {X.shape} | Target Shape: {y.shape}")
    print("CNN input shape is: batch x channels x time")

    splits = make_purged_intragroup_splits(groups, starts, ends, n_splits=N_SPLITS)
    fold = 1

    fold_mse = []
    fold_corr = []

    for train_idx, val_idx, n_purged, n_skipped_groups in splits:
        print(f"\n{'-'*20} Fold {fold} {'-'*20}")
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        print(f"Train windows: {len(train_idx)} | Val windows: {len(val_idx)}")
        print(f"Train groups: {len(train_groups)} | Val groups: {len(val_groups)} | Shared groups: {len(train_groups & val_groups)}")
        print(f"Purged overlapping train windows: {n_purged} | Skipped groups: {n_skipped_groups}")
        print(f"With WIN_LEN={WIN_LEN}, STRIDE={STRIDE}, up to {PURGE_GAP_WINDOWS} adjacent windows per side overlap a validation window.")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = CSIDATASET(X_train, y_train)
        val_dataset = CSIDATASET(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = CNN()
        criterion = MSEWithOutOfBandLoss(fs=FS / D,   low_hz=0.1,high_hz=0.4,lambda_oob=0.2,)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

        train_losses = []
        val_losses = []

        print("Training CNN...")

        for epoch in range(EPOCHS):
            model.train()
            running_train_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * xb.size(0)

            train_loss = running_train_loss / len(train_dataset)
            train_losses.append(train_loss)

            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    pred = model(xb)
                    loss = criterion(pred, yb)

                    running_val_loss += loss.item() * xb.size(0)

            val_loss = running_val_loss / len(val_dataset)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        model.eval()
        train_preds = []
        val_preds = []

        with torch.no_grad():
            for xb, _ in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False):
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                train_preds.append(pred)

            for xb, _ in DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False):
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)

        cnn_train_pred = np.concatenate(train_preds, axis=0)
        cnn_val_pred = np.concatenate(val_preds, axis=0)

        train_mse = mean_squared_error(y_train, cnn_train_pred)
        val_mse = mean_squared_error(y_val, cnn_val_pred)
        val_corr = pearsonr(y_val.flatten(), cnn_val_pred.flatten())[0]

        fold_mse.append(val_mse)
        fold_corr.append(val_corr)

        print(f"[CNN] Train MSE: {train_mse:.5f}")
        print(f"[CNN] Val   MSE: {val_mse:.5f}")
        print(f"[CNN] Val Pearson r: {val_corr:.5f}")

        # Save loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Fold {fold} - Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        save_name = f"fold_{fold}_loss_curve.png"
        plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150)
        plt.close()

        # Plot several training predictions
        n_train_plots = min(N_PLOTS_PER_FOLD, len(y_train))
        train_samples = np.random.choice(len(y_train), size=n_train_plots, replace=False)

        for i, sample_idx in enumerate(train_samples):
            plt.figure(figsize=(10, 4))
            plt.plot(y_train[sample_idx], label='True Mocap Label', color='black', linewidth=2)
            plt.plot(cnn_train_pred[sample_idx], label='CNN Prediction', linestyle='--', color='blue', alpha=0.8)

            plt.title(f"Fold {fold} - TRAIN Prediction vs Ground Truth - Window {sample_idx}")
            plt.xlabel("Time Samples Downsampled")
            plt.ylabel("Normalized Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

            save_name = f"fold_{fold}_train_{i}_window_{sample_idx}.png"
            plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150)
            plt.close()

        # Plot several validation predictions
        n_val_plots = min(N_PLOTS_PER_FOLD, len(y_val))
        val_samples = np.random.choice(len(y_val), size=n_val_plots, replace=False)

        for i, sample_idx in enumerate(val_samples):
            plt.figure(figsize=(10, 4))
            plt.plot(y_val[sample_idx], label='True Mocap Label', color='black', linewidth=2)
            plt.plot(cnn_val_pred[sample_idx], label='CNN Prediction', linestyle='--', color='blue', alpha=0.8)

            plt.title(f"Fold {fold} - VAL Prediction vs Ground Truth - Window {sample_idx}")
            plt.xlabel("Time Samples Downsampled")
            plt.ylabel("Normalized Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

            save_name = f"fold_{fold}_val_{i}_window_{sample_idx}.png"
            plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150)
            plt.close()

        fold += 1

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean Val MSE: {np.mean(fold_mse):.5f} ± {np.std(fold_mse):.5f}")
    print(f"Mean Val Pearson r: {np.mean(fold_corr):.5f} ± {np.std(fold_corr):.5f}")
    print(f"Plots saved in: {PLOT_DIR}")

def run_evaluation_ridge(data_subset, title):
    print("=" * 60)
    print(f" INVESTIGATION: {title}")
    print("=" * 60)

    PLOT_DIR = os.path.join("plots", title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", ""))
    os.makedirs(PLOT_DIR, exist_ok=True)

    N_PLOTS_PER_FOLD = 8

    X_raw = np.array([d['signal'] for d in data_subset])
    y_raw = np.array([d['label'] for d in data_subset])
    groups_raw = np.array([d['group_id'] for d in data_subset])
    print(np.shape(X_raw))
    starts_raw, ends_raw = get_window_bounds(data_subset, groups_raw)

    nan_in_X = np.isnan(X_raw).any(axis=(1, 2)) | np.isinf(X_raw).any(axis=(1, 2))
    nan_in_y = np.isnan(y_raw).any(axis=1) | np.isinf(y_raw).any(axis=1)
    valid_mask = ~(nan_in_X | nan_in_y)

    X = X_raw[valid_mask]
    y = y_raw[valid_mask]
    groups = groups_raw[valid_mask]
    starts = starts_raw[valid_mask]
    ends = ends_raw[valid_mask]

    n_dropped = len(data_subset) - len(y)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} windows out of {len(data_subset)} due to NaN/Inf values.")

    print(f"Cleaned Windows: {len(y)} | Feature Shape before flatten: {X.shape} | Target Shape: {y.shape}")

    # Flatten
    X = X.reshape(X.shape[0], -1)

    print(f"Feature Shape after flatten: {X.shape}")

    splits = make_purged_intragroup_splits(groups, starts, ends, n_splits=N_SPLITS)
    fold = 1

    fold_mse = []
    fold_corr = []

    for train_idx, val_idx, n_purged, n_skipped_groups in splits:
        print(f"\n{'-'*20} Fold {fold} {'-'*20}")
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        print(f"Train windows: {len(train_idx)} | Val windows: {len(val_idx)}")
        print(f"Train groups: {len(train_groups)} | Val groups: {len(val_groups)} | Shared groups: {len(train_groups & val_groups)}")
        print(f"Purged overlapping train windows: {n_purged} | Skipped groups: {n_skipped_groups}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]

        n_inner_groups = len(np.unique(groups_train))
        n_inner_splits = min(3, n_inner_groups)
        if n_inner_splits < 2:
            raise ValueError("Need at least 2 training groups for inner GroupKFold.")
        inner_cv = GroupKFold(n_splits=n_inner_splits)

        print("Training Ridge with GridSearchCV...")

        ridge_search = GridSearchCV(
            estimator=make_pipeline(StandardScaler(),Ridge()),
            param_grid={'ridge__alpha': [300.0, 1000.0, 2000.0, 3000,0, 4000,0]},
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        ridge_search.fit(X_train, y_train, groups=groups_train)

        ridge_train_pred = ridge_search.best_estimator_.predict(X_train)
        ridge_val_pred = ridge_search.best_estimator_.predict(X_val)

        train_mse = mean_squared_error(y_train, ridge_train_pred)
        val_mse = mean_squared_error(y_val, ridge_val_pred)

        val_corr = pearsonr(y_val.flatten(), ridge_val_pred.flatten())[0]

        fold_mse.append(val_mse)
        fold_corr.append(val_corr)

        print(f"[Ridge] Best Params: {ridge_search.best_params_}")
        print(f"[Ridge] Train MSE: {train_mse:.5f}")
        print(f"[Ridge] Val   MSE: {val_mse:.5f}")
        print(f"[Ridge] Val Pearson r: {val_corr:.5f}")

        # Plot several training predictions
        n_train_plots = min(N_PLOTS_PER_FOLD, len(y_train))
        train_samples = np.random.choice(len(y_train), size=n_train_plots, replace=False)

        for i, sample_idx in enumerate(train_samples):
            plt.figure(figsize=(10, 4))
            plt.plot(y_train[sample_idx], label='True Mocap Label', color='black', linewidth=2)
            plt.plot(ridge_train_pred[sample_idx], label='Ridge Prediction', linestyle='--', color='blue', alpha=0.8)

            plt.title(f"Fold {fold} - TRAIN Prediction vs Ground Truth - Window {sample_idx}")
            plt.xlabel("Time Samples (Downsampled)")
            plt.ylabel("Normalized Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

            save_name = f"fold_{fold}_train_{i}_window_{sample_idx}.png"
            plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150)
            plt.close()

        # Plot several validation predictions
        n_val_plots = min(N_PLOTS_PER_FOLD, len(y_val))
        val_samples = np.random.choice(len(y_val), size=n_val_plots, replace=False)

        for i, sample_idx in enumerate(val_samples):
            plt.figure(figsize=(10, 4))
            plt.plot(y_val[sample_idx], label='True Mocap Label', color='black', linewidth=2)
            plt.plot(ridge_val_pred[sample_idx], label='Ridge Prediction', linestyle='--', color='blue', alpha=0.8)

            plt.title(f"Fold {fold} - VAL Prediction vs Ground Truth - Window {sample_idx}")
            plt.xlabel("Time Samples (Downsampled)")
            plt.ylabel("Normalized Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

            save_name = f"fold_{fold}_val_{i}_window_{sample_idx}.png"
            plt.savefig(os.path.join(PLOT_DIR, save_name), dpi=150)
            plt.close()

        fold += 1

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean Val MSE: {np.mean(fold_mse):.5f} ± {np.std(fold_mse):.5f}")
    print(f"Mean Val Pearson r: {np.mean(fold_corr):.5f} ± {np.std(fold_corr):.5f}")
    print(f"Plots saved in: {PLOT_DIR}")

stationary_subset = [d for d in dataset if d['activity'] in ['A1','A2','A3'] and d['ori'] in ['O1']]
run_evaluation_cnn(stationary_subset, "CNN")
run_evaluation_ridge(stationary_subset, "Ridge")