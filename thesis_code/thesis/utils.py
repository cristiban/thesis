from hdf5storage import loadmat
import numpy as np
import os


def convert_to_npy(type, path="data"):
    if type == "response":
        files = [
            os.path.join(dp, f)
            for dp, _, fn in os.walk(path)
            for f in fn
            if "response" in f and f.endswith(".mat")
        ]
        for file in files:
            response = loadmat(file)["response"]
            response = np.transpose(response, (3, 2, 0, 1))
            out_path = file.replace(".mat", ".npy")
            np.save(out_path, response)

    if type == "stimulus":
        for file in files:
            stimulus = loadmat(file)["values"][0]
            np.save(file.replace(".mat", ".npy"), stimulus)
