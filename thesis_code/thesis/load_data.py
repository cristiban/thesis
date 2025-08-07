from hdf5storage import loadmat
import numpy as np
import os
import re
from .thesis_logger import logger


def load_response(path="data"):

    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(path)
        for f in fn
        if "response" in f and f.endswith(".npy") and f != "comb_response.npy"
    ]
    total_trials = 0
    animal_files = {}
    # get shape
    for file in files:
        sample = np.load(file, mmap_mode="r")
        total_trials += sample.shape[0]
        match = re.search(r'(mouse\d+)', file)
            
        if match:
            animal_name = match.group(1)
            if animal_name not in animal_files:
                animal_files[animal_name] = []
            animal_files[animal_name].append(file)

    shape = np.load(files[0], mmap_mode="r").shape

    comb_response = np.lib.format.open_memmap(f"{path}/comb_response.npy", mode="w+", shape=(total_trials, shape[1], shape[2], shape[3]), dtype=np.float32)
    animal_slices = {}
    index = 0
    for animal_name, files in animal_files.items():
        start_index = index
        for file in files:
            logger.debug(f"Adding {file}...")
            current_recording = np.load(file, mmap_mode="r")
            comb_response[index : index + current_recording.shape[0]] = current_recording
            logger.debug(f"This file contains {current_recording.shape[0]} trials")
            index += current_recording.shape[0]
        stop_index = index
        animal_slices[animal_name] = slice(start_index, stop_index)
    comb_response.flush()
    return comb_response, animal_slices


def load_stimulus(path="data"):

    raw_stimulus = []
    for dirpath, dirnames, filenames in os.walk(path):
        for x in filenames:
            if "stimulus" in x:
                current_stimulus = loadmat(os.path.join(dirpath, x))["values"]
                
                raw_stimulus.extend(current_stimulus[0])

    # print(raw_stimulus[49][0])
    stimulus = np.array(raw_stimulus)[:, 0]
    # print(stimulus)
    aux = []
    for index, trial in enumerate(stimulus):
        parseq_list = []
        for parseq in trial:
            parseq_list.append(list(parseq[1][0]))
            # print(parseq[1][0])
            # print(list(parseq_list))
        aux.append(list(parseq_list))
    # print(aux)
    # since each chord contains a varying number of tones, padding is added
    max_tones = np.max([len(chord) for trial in aux for chord in trial])

    for trial in aux:
        for chord in trial:
            to_extend = [-1 for _ in range(max_tones - len(chord))]
            chord = chord.extend(to_extend)

    stimulus = np.array(aux)

    drc_stimulus = []
    sr_audio = 250000
    tone_duration = 12500 / sr_audio
    amplitude = 0.1

    nr_steps = round(tone_duration * sr_audio)
    time = np.arange(0, nr_steps).T / sr_audio
    print(time)

    silence_duration = 12500 / sr_audio
    total_duration = 2.2
    intersilence = np.full_like(np.arange(0, silence_duration, 1 / sr_audio), 0)
    aftersilence = np.full_like(
        np.arange(
            0, total_duration - 2 * tone_duration - silence_duration, 1 / sr_audio
        ),
        0,
    )
    raw_stimulus = []
    for trial in stimulus:
        drc_stimulus = []
        for chord in trial:
            stim_sum = np.zeros_like(sr_audio)
            for tone in chord:
                if tone != -1:
                    signal = amplitude * np.sin(2 * np.pi * tone * time)
                    stim_sum = stim_sum + signal
            drc_stimulus.append(stim_sum)
        raw_stimulus.append(
            np.concatenate(
                (drc_stimulus[0], intersilence, drc_stimulus[1], aftersilence)
            )
        )
    return np.array(raw_stimulus)
