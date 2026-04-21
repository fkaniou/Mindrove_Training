import pickle
import cv2
import os
import logging
import time
import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import json
from scipy.io import savemat

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def show_image(i, gesture_image_path):
    """Display the image for the gesture."""
    image_file = os.path.join(gesture_image_path, f"g{i}.png")
    if os.path.exists(image_file):
        img = cv2.imread(image_file)
        if img is None:
            logging.warning(f"Error reading image '{image_file}'.")
            return
        img = cv2.resize(img, (0, 0), fx=1.0, fy=1.0)
        name = f"Current Gesture"
        
        # Close the previous window (if it exists)
        cv2.destroyAllWindows()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)  # Always on top
        cv2.imshow(name, img)    
        cv2.resizeWindow(name, img.shape[1], img.shape[0])
        cv2.waitKey(1)
    else:
        logging.error(f"Image file not found: {image_file}")


def preprocess_data(data, filters):
    for i in range(len(data)):
        for ch in range(data.shape[1]):
            for filter_ in filters:
                data[i, ch] = filter_.process(data[i, ch], ch)
    return data

def record_gestures(
    filters, 
    data_path, 
    gesture_image_path = "gestures", 
    skip_gestures = [], 
    gestures_repeat = 1, 
    recording_time_sec = 8, 
    sampling_rate = 500, 
    model_input_len = 100, 
    overlap_frac = 10
):
    """
    Record gestures from the MindRove board and save them to a file.
    
    Args:
        filters (list): List of filters.
        data_path (str): Path to save the recorded data.
        gesture_image_path (str): Path to the gesture images.
        skip_gestures (list): List of gesture IDs to skip.
        gestures_repeat (int): Number of times to repeat the gestures.
        recording_time_sec (int): Duration to record each gesture.
        sampling_rate (int): Sampling rate of the MindRove board.
        model_input_len (int): Length of input data to the model.
        overlap_frac (int): Overlap fraction between samples.

    Returns:
        None
    """

    recorded_data = []
    recorded_labels = []
    fft_warmup = 10
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)

    try:
        # Prepare and start streaming from the MindRove board
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        logging.info("Starting streaming from MindRove board.")
        
        # Warm-up phase to ensure the board is ready
        start_time = time.time()
        while time.time() - start_time < fft_warmup:
            if board_shim.get_board_data_count() > 0:
                        raw_data = board_shim.get_board_data(sampling_rate)
                        emg_data = raw_data[:8]
                        _ = preprocess_data(emg_data.T, filters)

        # Start recording gestures
        for repeat in range(gestures_repeat):
            for gesture_id in range(7): 
                if gesture_id in skip_gestures:
                    continue

                show_image(gesture_id, gesture_image_path)
                input(f"{repeat}/{gestures_repeat} - Perform gesture id {gesture_id}. Press Enter to start recording for {recording_time_sec} seconds.")
                gesture_data = []
                board_shim.get_board_data()

                # Wait until the desired number of samples are received
                while board_shim.get_board_data_count() < recording_time_sec * sampling_rate:
                    pass
                
                # Collect the data
                raw_data = board_shim.get_board_data(recording_time_sec * sampling_rate)
                emg_data = raw_data[:8] # Only EMG data
                processed_data = preprocess_data(emg_data.T, filters)

                for i in range(0, len(processed_data) - model_input_len, overlap_frac):
                    sample = processed_data[i:i + model_input_len]
                    sample = np.expand_dims(sample.T, axis=2).astype(np.float32)
                    gesture_data.append(sample)

                recorded_data.extend(gesture_data)
                recorded_labels.extend([gesture_id] * len(gesture_data))

        # Save the recorded data to a file
        with open(data_path, "wb+") as f:
            pickle.dump((recorded_data, recorded_labels), f)
        logging.info(f"Gestures successfully recorded and saved to {data_path}.")

    except Exception as e:
        logging.error(f"Error recording gestures: {e}")
    finally:
        cv2.destroyAllWindows()
        board_shim.stop_stream()
        board_shim.release_session()
        logging.info("Disconnected from MindRove board.")

def record_ninapro_gestures(
    filters,
    base_save_dir="C:\\Users\\User\\Desktop\\semg_recording\\NaviFlame\\naviflame\\data\\recorded_ninapro",
    subject_id=1,
    repetition_id=1,
    gesture_image_path="ninapro_gestures",
    total_gestures=30,
    skip_gestures=[],
    recording_time_sec=8,
    sampling_rate=500
):

    # === CREATE SUBJECT/REPETITION DIRECTORY ===
    subject_folder = f"subject{subject_id}_rep{repetition_id}"
    save_dir = os.path.join(base_save_dir, subject_folder)

    raw_dir = os.path.join(save_dir, "raw")
    proc_dir = os.path.join(save_dir, "preprocessed")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    logging.info(f"Saving RAW signals to: {raw_dir}")
    logging.info(f"Saving PREPROCESSED signals to: {proc_dir}")

    # === MindRove Setup ===
    fft_warmup = 10
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        logging.info("MindRove stream started.")

        # === Warm-up phase ===
        start_time = time.time()
        while time.time() - start_time < fft_warmup:
            if board_shim.get_board_data_count() > 0:
                warm = board_shim.get_board_data(sampling_rate)
                emg_warm = warm[:8]
                _ = preprocess_data(emg_warm.T, filters)

        # === Gesture recording loop ===
        #for gesture_id in range(total_gestures):
        gesture_id = [17,18,25,26,27,28] 
        for gesture_id in gesture_id:

            if gesture_id in skip_gestures:
                continue

            show_image(gesture_id, gesture_image_path)

            input(
                f"[Subject {subject_id} | Rep {repetition_id}] "
                f"Gesture {gesture_id:02d} — press Enter to record {recording_time_sec}s..."
            )

            # Clear buffer before recording
            board_shim.get_board_data()

            # Wait until enough data arrives
            required_samples = recording_time_sec * sampling_rate
            while board_shim.get_board_data_count() < required_samples:
                pass

            # === Retrieve RAW data ===
            raw = board_shim.get_board_data(required_samples)
            raw_emg = raw[:8].T  # (samples, 8)

            # === Apply preprocessing ===
            processed_emg = preprocess_data(raw_emg.copy(), filters)

            # === Build stimulus vector ===
            stim_vec = np.full(
                (processed_emg.shape[0], 1),
                gesture_id,                 # gesture id starts from 0
                dtype=np.int32
            )

            # === Save RAW ===
            raw_path = os.path.join(raw_dir, f"gesture{gesture_id:02d}.mat")
            savemat(raw_path, {
                "emg_raw": raw_emg.astype(np.float32),
                "stimulus": stim_vec
            })
            logging.info(f"[SAVED RAW] {raw_path}")

            # === Save PREPROCESSED ===
            proc_path = os.path.join(proc_dir, f"gesture{gesture_id:02d}.mat")
            savemat(proc_path, {
                "emg": processed_emg.astype(np.float32),
                "stimulus": stim_vec
            })
            logging.info(f"[SAVED PROC] {proc_path}")

        logging.info("[INFO] All gestures recorded successfully.")

    except Exception as e:
        logging.error(f"[ERROR] During NiNapro recording: {e}")

    finally:
        cv2.destroyAllWindows()
        board_shim.stop_stream()
        board_shim.release_session()
        logging.info("[INFO] MindRove disconnected.")
