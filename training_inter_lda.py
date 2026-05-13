#Numerical Operations
import numpy as np 
from einops import rearrange 
import os 
import os.path as osp
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt,iirnotch, tf2sos,sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#working directory
w_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = osp.join(w_dir, '6_gest2')

#Train-validation-test split
train_reps = [1,2,3,5,6,8,10]
val_reps = [4,7]
test_reps = [9]

#Load files
files = sorted(os.listdir(dataset_dir))
#Path to a reference gesture
ges_path = osp.join(dataset_dir,files[0],'raw','gesture25.mat')

#Load .mat file in python 
data  = loadmat(ges_path)
print(data.keys())

emg = rearrange(data['emg_raw'],'t c -> c t')
print(emg.shape)

#Sampling Frequency
fs = 500
Ts = 1/fs
#t=nTs (time)
t = np.arange(emg.shape[1])*Ts#Sampling Frequency
fs = 500
Ts = 1/fs
#t=nTs (time)
t = np.arange(emg.shape[1])*Ts

#Second Order Sections for Numerical stability
sos = butter(N=4,Wn=[4,200],btype='bandpass',fs=500,output='sos')
emg_f = sosfiltfilt(sos,emg,axis=1)

from scipy.fft import fft,fftfreq
#Zero padding 2500 -> 4096
N=4096
#Analog Frequency that corresponds to each FFT coefficient 
x_f = fftfreq(N,Ts)[:N//2]
y_f = fft(emg_f,n=N,axis=1)[:,:N//2]

#Notch Frequency
f0 = 50
#Quality Factor
Q = 30.0 
#Notch filter using sos represantion for numerical stability 
b,a = iirnotch(f0,Q,fs=fs)
sos_notch = tf2sos(b,a)

emg_final = sosfiltfilt(sos_notch,emg_f,axis=1)
print("EMG shape:", emg_final.shape)
y_f2 = fft(emg_final,n=N,axis=1)[:,:N//2]

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

fe = FeatureExtractor()
#Window size
WS = 50
T_window = WS*Ts
WI = 10
#Window increment 
print(f'Window Time {T_window*1000}ms')
print(f'Overlap {(1-WI/WS)*100}%')
#Apply windows to the middle 3s of the gesture
windows = get_windows(emg_final[:,750:-750].T,WS,WI)

print("Windows shape:", windows.shape)

features = fe.extract_feature_group('HTD',windows)
print("Feature keys:", list(features.keys()))

subjects = [1,2,3,4,5,6]

all_results = {}

for subj in subjects:

    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    subject_files = [f for f in files if f"subject{subj}_" in f]

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

            if rep in train_reps:
                x_train.append(feature_vector)
                y_train.append(np.repeat(ges_num, n_samples))

            elif rep in val_reps:
                x_val.append(feature_vector)
                y_val.append(np.repeat(ges_num, n_samples))

            else:
                x_test.append(feature_vector)
                y_test.append(np.repeat(ges_num, n_samples))

    # ===== IMPORTANT: INSIDE LOOP =====
    x_train = np.concatenate(x_train)
    x_val = np.concatenate(x_val)

    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)

    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LinearDiscriminantAnalysis()

    
    param_grid = {}

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,          # μέσα στο train set
        n_jobs=-1,
        verbose=1
    )

    grid.fit(x_train_scaled, y_train)

    clf = grid.best_estimator_

    print("Best params:", grid.best_params_)

    train_acc = clf.score(x_train_scaled, y_train)

    x_val_scaled = scaler.transform(x_val)
    y_pred = clf.predict(x_val_scaled)
    val_acc = accuracy_score(y_val, y_pred)

    x_test_scaled = scaler.transform(x_test)
    y_test_pred = clf.predict(x_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nSubject {subj}")
    print(f"Train Acc: {train_acc*100:.2f}%")
    print(f"Val Acc: {val_acc*100:.2f}%")
    print(f"Test Acc: {test_acc*100:.2f}%")

    all_results[subj] = val_acc

    # confusion matrix per subject
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred, normalize='true', cmap=plt.cm.Blues
    )
    plt.title(f"Subject {subj}")
    plt.show()


