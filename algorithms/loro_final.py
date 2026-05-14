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
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

w_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = osp.join(w_dir, '6_gest2')
subjects = [1, 2, 3, 4, 5, 6]
reps = list(range(1, 11))

train_reps_glob = [1, 2, 3, 5, 6, 8, 10]
val_reps_glob = [4, 7]

fs, WS, WI = 500, 50, 10
fe = FeatureExtractor()

sos_band = butter(N=4, Wn=[4, 200], btype='bandpass', fs=fs, output='sos')
b, a = iirnotch(50, 30.0, fs=fs)
sos_notch = tf2sos(b, a)

def process_emg(emg_data):
    filtered = sosfiltfilt(sos_band, emg_data, axis=0)
    filtered = sosfiltfilt(sos_notch, filtered, axis=0)
    return filtered[750:-750, :]

print("Load data...")
files = sorted(os.listdir(dataset_dir))
data_storage = {s: {r: {'x': None, 'y': None} for r in reps} for s in subjects}

X_train_glob, Y_train_glob = [], []
X_val_glob, Y_val_glob = [], []

for subj in subjects:
    subj_files = [f for f in files if f"subject{subj}_" in f]
    for f in subj_files:
        rep_num = int(re.search(r"rep(\d+)", f).group(1))
        rep_path = osp.join(dataset_dir, f, 'raw')
        gestures = sorted(os.listdir(rep_path))
        
        x_rep, y_rep = [], []
        for ges_idx, g in enumerate(gestures):
            raw = loadmat(osp.join(rep_path, g))['emg_raw']
            clean = process_emg(raw)
            windows = get_windows(clean, WS, WI)
            features = fe.extract_feature_group('HTD', windows)
            fv = rearrange(np.array(list(features.values())), 'f w c -> w (c f)')
            x_rep.append(fv)
            y_rep.append(np.repeat(ges_idx, fv.shape[0]))
        
        final_x = np.concatenate(x_rep)
        final_y = np.concatenate(y_rep)
        
        
        data_storage[subj][rep_num]['x'] = final_x
        data_storage[subj][rep_num]['y'] = final_y

        if rep_num in train_reps_glob:
            X_train_glob.append(final_x); Y_train_glob.append(final_y)
        elif rep_num in val_reps_glob:
            X_val_glob.append(final_x); Y_val_glob.append(final_y)

X_train_glob = np.concatenate(X_train_glob)
Y_train_glob = np.concatenate(Y_train_glob)
X_val_glob = np.concatenate(X_val_glob)
Y_val_glob = np.concatenate(Y_val_glob)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_glob)
X_val_sc = scaler.transform(X_val_glob)

X_grid = np.vstack((X_train_sc, X_val_sc))
Y_grid = np.concatenate((Y_train_glob, Y_val_glob))
ps = PredefinedSplit(np.concatenate([-1 * np.ones(len(Y_train_glob)), np.zeros(len(Y_val_glob))]))

models_to_test = [
    {'name': 'SVM', 'est': SVC(), 'params': {'C': [1, 10, 50], 'gamma': ['scale', 0.01]}},
    {'name': 'kNN', 'est': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}},
    {'name': 'RF', 'est': RandomForestClassifier(), 'params': {'n_estimators': [100], 'max_depth': [None, 10, 20]}},
    {'name': 'LDA', 'est': LinearDiscriminantAnalysis(), 'params': {}}
]

final_summary = []

for m_info in models_to_test:
    print(f"\n{'='*30}")
    print(f" MODEL: {m_info['name']}")
    
    # Global Grid Search
    grid = GridSearchCV(m_info['est'], m_info['params'], cv=ps, n_jobs=-1)
    grid.fit(X_grid, Y_grid)
    best_p = grid.best_params_
    print(f" Global Best Params: {best_p}\n")

    all_subj_train_accs, all_subj_test_accs = [], []

    for subj in subjects:
        print(f"--- Subject {subj} ---")
        subj_rep_train, subj_rep_test = [], []
        
        for out_rep in reps:
            x_train = np.concatenate([data_storage[subj][r]['x'] for r in reps if r != out_rep])
            y_train = np.concatenate([data_storage[subj][r]['y'] for r in reps if r != out_rep])
            x_test = data_storage[subj][out_rep]['x']
            y_test = data_storage[subj][out_rep]['y']          
           
            sc = StandardScaler()
            x_train_sc = sc.fit_transform(x_train)
            x_test_sc = sc.transform(x_test)
            
            clf = m_info['est'].set_params(**best_p)
            clf.fit(x_train_sc, y_train)
            
            tr_acc = clf.score(x_train_sc, y_train)
            te_acc = clf.score(x_test_sc, y_test)
            
            subj_rep_train.append(tr_acc)
            subj_rep_test.append(te_acc)
            
            print(f"  Repetition {out_rep:2} Test Accuracy: {te_acc*100:6.2f}%")
            
        avg_tr = np.mean(subj_rep_train)
        avg_te = np.mean(subj_rep_test)
        print(f">> Mean values for Subject {subj} | Train: {avg_tr*100:.1f}% | Test: {avg_te*100:.1f}%\n")
        
        all_subj_train_accs.append(avg_tr)
        all_subj_test_accs.append(avg_te)

    final_summary.append({
        'Model': m_info['name'],
        'Best Params': str(best_p),
        'Global LORO Train': np.mean(all_subj_train_accs),
        'Global LORO Test': np.mean(all_subj_test_accs)
    })
    
print("\n" + "!"*50)
print(f"{'Model':<12} | {'LORO Train':<12} | {'LORO Test':<12} | {'Params'}")
print("-" * 90)
for r in final_summary:
    print(f"{r['Model']:<12} | {r['Global LORO Train']*100:10.2f}% | {r['Global LORO Test']*100:10.2f}% | {r['Best Params']}")
