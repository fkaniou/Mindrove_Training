import os
import sys
import pickle
import cv2
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import FilterTypes, BiquadMultiChan
from record import record_ninapro_gestures

#This code is used for the recording of the gestures from the MindRove Armband
#It will save the gestures of exercises A and B of NiNapro dataset (raw and preprocessed data)

#config loading
def load_config():
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config_ninapro.json")),
        os.path.abspath("config_ninapro.json")
    ]
    config_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if config_path is None:
        raise FileNotFoundError("Configuration file not found in the expected locations.")

    with open(config_path, "r") as f:
        config = json.load(f)

    config_dir = os.path.dirname(config_path)
    for key in ["data_path", "feature_extractor_path", "mlp_model_path", "scaler_path", "gesture_image_path"]:
        if key in config:
            config[key] = os.path.abspath(os.path.join(config_dir, config[key]))

    return config


def main():
    config = load_config()
    # Paths from config
    data_path = config["data_path"]
    gesture_image_path = config["gesture_image_path"]
    subject_id = config["subject_id"]
    repetition_id = config["repetition_id"]
    total_gestures = config["total_gestures"]
    recording_time_sec = config["recording_time_sec"]
    skip_gestures = config["skip_gestures"]

    #Flags
    record = config["record"]

    # Sampling rate and model input length
    sampling_rate = 500

    # Define filters
    filters = [
        BiquadMultiChan(8, FilterTypes.bq_type_highpass, 4.5 / sampling_rate, 0.5, 0.0), # Dc filter
        BiquadMultiChan(8, FilterTypes.bq_type_notch, 50.0 / sampling_rate, 4.0, 0.0), # 50 Hz noise
        BiquadMultiChan(8, FilterTypes.bq_type_lowpass, 100.0 / sampling_rate, 0.5, 0.0), 
    ]

    #Record gestures
    if record:
        record_ninapro_gestures(
            filters=filters,
            base_save_dir=data_path,
            subject_id=subject_id,
            repetition_id=repetition_id,
            gesture_image_path=gesture_image_path,
            total_gestures=total_gestures,
            skip_gestures=skip_gestures,
            recording_time_sec=recording_time_sec,
            sampling_rate=sampling_rate 
        )

    else:
        print("Recording is disabled in the json file.") 

if __name__ == "__main__":
    main()
