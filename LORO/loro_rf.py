# Numerical Operations
import numpy as np 
from einops import rearrange 
import os 
import os.path as osp
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# working directory
w_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = osp.join(w_dir, '6_gest2')

files = sorted(os.listdir(dataset_dir))

ges_path = osp.join(dataset_dir, files[0], 'raw', 'gesture25.mat')

data = loadmat(ges_path)
print(data.keys())

emg = rearrange(data['emg_raw'], 't c -> c t')
print(emg.shape)

fs = 500
Ts = 1/fs
t = np.arange(emg.shape[1]) * Ts

sos = butter(N=4, Wn=[4,200], btype='bandpass', fs=500, output='sos')
emg_f = sosfiltfilt(sos, emg, axis=1)

N = 4096

f0 = 50
Q = 30.0

b, a = iirnotch(f0, Q, fs=fs)
sos_notch = tf2sos(b, a)

emg_final = sosfiltfilt(sos_notch, emg_f, axis=1)

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

fe = FeatureExtractor()

WS = 50
WI = 10

print(f'Window Time {WS*Ts*1000}ms')
print(f'Overlap {(1-WI/WS)*100}%')

windows = get_windows(emg_final[:,750:-750].T, WS, WI)

features = fe.extract_feature_group('HTD', windows)
print("Feature keys:", list(features.keys()))

subjects = [1,2,3,4,5,6]

all_results = {}

for subj in subjects:

    reps = list(range(1, 11))
    rep_accuracies = []

    subject_files = [f for f in files if f"subject{subj}_" in f]

    for test_rep in reps:

        x_train, y_train = [], []
        x_test, y_test = [], []

        for f in subject_files:

            import re
            rep = int(re.search(r"rep(\d+)", f).group(1))

            rep_path = osp.join(dataset_dir, f, 'raw')
            gestures = sorted(os.listdir(rep_path))

            for ges_num, g in enumerate(gestures):

                emg = loadmat(osp.join(rep_path, g))['emg_raw']

                emg_f = sosfiltfilt(sos, emg, axis=0)
                emg_fn = sosfiltfilt(sos_notch, emg_f, axis=0)

                emg_fn = emg_fn[750:-750, :]

                windows = get_windows(emg_fn, WS, WI)
                emg_features = fe.extract_feature_group('HTD', windows)

                feature_vector = np.array([emg_features[k] for k in emg_features.keys()])
                feature_vector = rearrange(feature_vector, 'f w c -> w (c f)')

                n_samples, _ = feature_vector.shape

                if rep == test_rep:
                    x_test.append(feature_vector)
                    y_test.append(np.repeat(ges_num, n_samples))
                else:
                    x_train.append(feature_vector)
                    y_train.append(np.repeat(ges_num, n_samples))

        # flatten
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        x_train, y_train = shuffle(x_train, y_train, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = RandomForestClassifier()

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(x_train_scaled, y_train)

        clf = grid.best_estimator_

        y_pred = clf.predict(x_test_scaled)

        acc = accuracy_score(y_test, y_pred)

        rep_accuracies.append(acc)

        print(f"Subject {subj} | Rep {test_rep} acc: {acc*100:.2f}%")

    print(f"\nSubject {subj} FINAL LORO: {np.mean(rep_accuracies)*100:.2f}%")
    all_results[subj] = np.mean(rep_accuracies)
