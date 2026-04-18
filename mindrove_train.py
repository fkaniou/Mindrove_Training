import sys
import os
import numpy as np
import scipy.io
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Modules
from modules.DataGenerator import DataGenerator
from modules.AtzoriNet import AtzoriNet

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Softmax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def load_my_data(base_path):
    signals = []
    labels = []
    
    subjects_reps = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for folder in subjects_reps:
        path_to_preprocessed = os.path.join(base_path, folder, 'preprocessed')
        if not os.path.exists(path_to_preprocessed): continue
            
        print(f"Reading folder: {folder}...", end='\r')
        
        for file in os.listdir(path_to_preprocessed):
            if file.endswith(".mat") and "gesture" in file:
                file_path = os.path.join(path_to_preprocessed, file)
                mat_data = scipy.io.loadmat(file_path)
                
                # Φόρτωση EMG
                emg_signal = mat_data.get('emg') 
                if emg_signal is None:
                    keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    emg_signal = mat_data[keys[0]]

                # Λήψη Label (0-29)
                label = int(np.median(mat_data['stimulus'])) if 'stimulus' in mat_data else int(file.replace('gesture', '').replace('.mat', ''))

                # --- PREPROCESSING ---
                emg_signal = emg_signal.astype('float32')
                emg_signal = np.abs(emg_signal) # 1. Rectification
                
                # 2. Z-score Normalization
                mean_val = np.mean(emg_signal)
                std_val = np.std(emg_signal)
                if std_val > 0:
                    emg_signal = (emg_signal - mean_val) / std_val

                # --- 3. ΕΔΩ ΜΠΑΙΝΕΙ ΤΟ ΠΑΝΤΙΝΓΚ (ΤΟ "1") ---
                if emg_signal.shape[1] == 8:
                    # Φτιάχνουμε μια στήλη μηδενικά για την αρχή και μια για το τέλος
                    pad_col = np.zeros((emg_signal.shape[0], 1), dtype='float32')
                    # Ενώνουμε: [0] + [8 κανάλια] + [0]
                    emg_signal = np.hstack((pad_col, emg_signal, pad_col))

                # Τώρα το σήμα έχει 10 κανάλια και είναι έτοιμο
                signals.append(emg_signal)
                labels.append(label)
                
    return signals, labels

def run_pipeline():
    base_path = 'C:\\Users\\User\\Desktop\\semg_recording\\NaviFlame\\naviflame\\data\\recorded_ninapro'
    
    signals, labels = load_my_data(base_path)
    
    if len(signals) == 0:
        print("Error! No data found!")
        return

    # Split
    train_sig, temp_sig, train_lab, temp_lab = train_test_split(
        signals, labels, test_size=0.20, random_state=42, stratify=labels
    )
    val_sig, test_sig, val_lab, test_lab = train_test_split(
        temp_sig, temp_lab, test_size=0.50, random_state=42, stratify=temp_lab
    )

    W_SIZE = 150  # Αυξημένο παράθυρο για καλύτερη αναγνώριση 30 κινήσεων
    CHANNELS = 10  # MINDROVE CHANNELS
    NUM_CLASSES = 30
    
    # Generators
    train_gen = DataGenerator(train_sig, train_lab, batch_size=64, dim=(W_SIZE, CHANNELS, 1), classes=NUM_CLASSES, window_size=W_SIZE, shuffle=True)
    val_gen   = DataGenerator(val_sig, val_lab, batch_size=64, dim=(W_SIZE, CHANNELS, 1), classes=NUM_CLASSES, window_size=W_SIZE, shuffle=False)
    test_gen  = DataGenerator(test_sig, test_lab, batch_size=64, dim=(W_SIZE, CHANNELS, 1), classes=NUM_CLASSES, window_size=W_SIZE, shuffle=False)

    # Μοντέλο προσαρμοσμένο στα 8 κανάλια
    base_atzori = AtzoriNet(input_shape=(W_SIZE, CHANNELS, 1), classes=NUM_CLASSES, batch_norm=True, n_dropout=0.5)
    
    # Global Average Pooling Fix
    x = base_atzori.layers[-3].output
    x = GlobalAveragePooling2D(name='global_fix')(x)
    new_output = Softmax(name='final_softmax')(x)
    
    model = Model(inputs=base_atzori.input, outputs=new_output, name='AtzoriNet_8Ch_30Cl')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print(f"\nStarting training (Window: {W_SIZE}, Channels: {CHANNELS}, Classes: {NUM_CLASSES})...")
    model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[stop, reduce_lr])

    print("\n--- TEST EVALUATION ---")
    loss, acc = model.evaluate(test_gen)
    print(f"Final Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    run_pipeline()