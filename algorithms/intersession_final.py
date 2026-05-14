import numpy as np 
from einops import rearrange 
import os 
import os.path as osp
import re
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt 
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

w_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = osp.join(w_dir, '6_gest2')
subjects = [1, 2, 3, 4, 5, 6]
train_reps = [1, 2, 3, 5, 6, 8, 10]
val_reps = [4, 7]
test_reps = [9]

fs, WS, WI = 500, 50, 10
fe = FeatureExtractor()

sos_band = butter(N=4, Wn=[4, 200], btype='bandpass', fs=fs, output='sos')
b, a = iirnotch(50, 30.0, fs=fs)
sos_notch = tf2sos(b, a)

def process_emg(emg_data):
    filtered = sosfiltfilt(sos_band, emg_data, axis=0)
    filtered = sosfiltfilt(sos_notch, filtered, axis=0)
    return filtered[750:-750, :]


print("Loading data and feature extraction...")
files = sorted(os.listdir(dataset_dir))

all_x_train, all_y_train = [], []
all_x_val, all_y_val = [], []
subject_data = {s: {'train_x': [], 'train_y': [], 'val_x': [], 'val_y': [], 'test_x': [], 'test_y': []} for s in subjects}

for subj in subjects:
    subject_files = [f for f in files if f"subject{subj}_" in f]
    for f in subject_files:
        rep = int(re.search(r"rep(\d+)", f).group(1))
        rep_path = osp.join(dataset_dir, f, 'raw')
        gestures = sorted(os.listdir(rep_path))

        for ges_num, g in enumerate(gestures):
            data = loadmat(osp.join(rep_path, g))['emg_raw']
            emg_clean = process_emg(data)
            windows = get_windows(emg_clean, WS, WI)
            feat_dict = fe.extract_feature_group('HTD', windows)
            
            fv = np.array([feat_dict[k] for k in feat_dict.keys()])
            fv = rearrange(fv, 'f w c -> w (c f)')
            labels = np.repeat(ges_num, fv.shape[0])

            if rep in train_reps:
                all_x_train.append(fv); all_y_train.append(labels)
                subject_data[subj]['train_x'].append(fv); subject_data[subj]['train_y'].append(labels)
            elif rep in val_reps:
                all_x_val.append(fv); all_y_val.append(labels)
                subject_data[subj]['val_x'].append(fv); subject_data[subj]['val_y'].append(labels)
            elif rep in test_reps:
                subject_data[subj]['test_x'].append(fv); subject_data[subj]['test_y'].append(labels)

X_train_global = np.concatenate(all_x_train)
Y_train_global = np.concatenate(all_y_train)
X_val_global = np.concatenate(all_x_val)
Y_val_global = np.concatenate(all_y_val)

scaler = StandardScaler()
X_train_global_scaled = scaler.fit_transform(X_train_global)
X_val_global_scaled = scaler.transform(X_val_global)


models_to_test = [
    {'name': 'SVM', 'estimator': SVC(), 'params': {'C': [1, 10, 50], 'gamma': ['scale', 0.01], 'kernel': ['rbf']}},
    {'name': 'kNN', 'estimator': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}},
    {'name': 'RandomForest', 'estimator': RandomForestClassifier(), 'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}},
    {'name': 'LDA', 'estimator': LinearDiscriminantAnalysis(), 'params': {}}
]

X_grid = np.vstack((X_train_global_scaled, X_val_global_scaled))
Y_grid = np.concatenate((Y_train_global, Y_val_global))
ps = PredefinedSplit(np.concatenate([-1 * np.ones(len(Y_train_global)), np.zeros(len(Y_val_global))]))

final_summary = []

for m in models_to_test:
    print(f"\n{'='*30}\nAnalysis for: {m['name']}")
    grid = GridSearchCV(m['estimator'], m['params'], cv=ps, n_jobs=-1)
    grid.fit(X_grid, Y_grid)
    best_p = grid.best_params_
    
    m_train_accs, m_val_accs, m_test_accs = [], [], []

    for subj in subjects:
        x_tr = scaler.transform(np.concatenate(subject_data[subj]['train_x']))
        y_tr = np.concatenate(subject_data[subj]['train_y'])
        
        x_val = scaler.transform(np.concatenate(subject_data[subj]['val_x']))
        y_val = np.concatenate(subject_data[subj]['val_y'])
        
        x_te = scaler.transform(np.concatenate(subject_data[subj]['test_x']))
        y_te = np.concatenate(subject_data[subj]['test_y'])
        
        clf = m['estimator'].set_params(**best_p)
        clf.fit(x_tr, y_tr)
        
        # Accuracies
        tr_acc = clf.score(x_tr, y_tr)
        val_acc = clf.score(x_val, y_val)
        te_acc = clf.score(x_te, y_te)
        
        print(f"Subj {subj} | Train: {tr_acc*100:.1f}% | Val: {val_acc*100:.1f}% | Test: {te_acc*100:.1f}%")
        
        m_train_accs.append(tr_acc)
        m_val_accs.append(val_acc)
        m_test_accs.append(te_acc)

    final_summary.append({
        'Model': m['name'],
        'Best Params': str(best_p),
        'Avg Train': np.mean(m_train_accs),
        'Avg Val': np.mean(m_val_accs),
        'Avg Test': np.mean(m_test_accs)
    })

print(f"{'Model':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Best Params'}")
print("-" * 100)
for r in final_summary:
    print(f"{r['Model']:<12} | {r['Avg Train']*100:6.2f}% | {r['Avg Val']*100:6.2f}% | {r['Avg Test']*100:6.2f}% | {r['Best Params']}")
