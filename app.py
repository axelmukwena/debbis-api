import json
import os
import time
from itertools import product

import biosppy
import numpy as np
from flask import Flask, request
from scipy.signal import filtfilt
from tensorflow.keras.models import load_model

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

app = Flask(__name__)
VERIFY_MODEL = load_model("model/", compile=False)
AUTHENTICATE_MODELS = {}
TEMPLATES = []
CLASSES = []
TEMPS = 0
TRIM = 20


def refine_r_peaks(sig, r_peaks):
    r_peaks2 = np.array(r_peaks)  # make a copy
    for i in range(len(r_peaks)):
        r = r_peaks[i]  # current R-peak
        small_segment = sig[max(0, r - 100):min(len(sig), r + 100)]  # consider the neighboring segment of R-peak
        r_peaks2[i] = np.argmax(small_segment) - 100 + r_peaks[i]  # picking the highest point
        r_peaks2[i] = min(r_peaks2[i], len(sig))  # the detected R-peak shouldn't be outside the signal
        r_peaks2[i] = max(r_peaks2[i], 0)  # checking if it goes before zero
    return r_peaks2  # returning the refined r-peak list


def segment_signals(sig, r_peaks_annot, bmd=True, normalization=True):
    segmented_signals = []
    r_peaks = np.array(r_peaks_annot)
    r_peaks = refine_r_peaks(sig, r_peaks)
    if bmd:
        win_len = 300
    else:
        win_len = 256
    win_len_1_4 = win_len // 4
    win_len_3_4 = 3 * (win_len // 4)
    for r in r_peaks:
        if ((r - win_len_1_4) < 0) or ((r + win_len_3_4) >= len(sig)):  # not enough signal to segment
            continue
        segmented_signal = np.array(sig[r - win_len_1_4:r + win_len_3_4])  # segmenting a heartbeat

        if normalization:  # Z-score normalization
            if abs(np.std(segmented_signal)) < 1e-6:  # flat line ECG, will cause zero division error
                continue
            segmented_signal = (segmented_signal - np.mean(segmented_signal)) / np.std(segmented_signal)

        if not np.isnan(segmented_signal).any():  # checking for nan, this will never happen
            segmented_signals.append(segmented_signal)

    return segmented_signals, r_peaks


def normalize(x):
    x = np.array(x)
    x = (x - x.min()) / (x.max() - x.min())
    return x


def filters(array, n):
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    array = filtfilt(b, a, array, padlen=50)
    return array


def load_templates():
    global TEMPLATES, TEMPS, CLASSES
    TEMPLATES = []
    TEMPS = 0
    CLASSES = []
    files = os.listdir("templates/")
    for file in files:
        if file.endswith(".json"):
            basename = file.split('.')[0]
            with open("templates/" + file) as f:
                data = json.load(f)
                waves = data["templates"][:TRIM]
                TEMPLATES += waves
                TEMPS += 1
                CLASSES.append(basename)


def check_enrolment(samples):
    pairs = [[t, s] for t, s in product(TEMPLATES, samples)]
    pairs = normalize(pairs)
    t = time.time()
    ps = VERIFY_MODEL.predict([pairs[:, 0], pairs[:, 1]])  # output looks like [[[1]], [[2]], [[3]]]
    ps = [i[0][0] for i in ps]
    print(time.time() - t)
    return ps


def process(data):
    data = data.splitlines()[1:]
    array = [int(m.split()[1]) for m in data]
    array = np.array(array, dtype="float32")
    peaks = biosppy.signals.ecg.hamilton_segmenter(signal=array, sampling_rate=500)[0]
    waves, _ = segment_signals(array, peaks, False, True)
    return waves


def create_template(waves, who):
    waves = [w.tolist() for w in waves[:200]]
    data = {"templates": waves}
    with open('templates/' + str(who) + '.json', 'w') as outfile:
        json.dump(data, outfile)


@app.route("/enroll", methods=["POST"])
def enroll():
    result = {"error": "started..."}
    # ensure the file was properly uploaded to our endpoint
    if request.method == "POST":
        if request.get_json(force=True):
            data = request.get_json(force=True)["data"]
            who = data[0]
            try:
                waves = process(data[1])
                create_template(waves, who)
                return json.dumps({"success": str(who) + " Template created"})
            except ValueError:
                print("error")
                return json.dumps({"error": "Failed to delete template"})
        else:
            return json.dumps({"error": "Json object posted not valid"})
    return result


@app.route("/unenroll", methods=["POST"])
def unenroll():
    result = {"error": "started..."}
    # ensure the file was properly uploaded to our endpoint
    if request.method == "POST":
        if request.get_json(force=True):
            data = request.get_json(force=True)["data"]
            try:
                os.remove("templates/" + str(data) + ".json")
                return json.dumps({"success": str(data) + " Template deleted"})
            except ValueError:
                print("error")
                return json.dumps({"error": "Failed to delete template"})
        else:
            return json.dumps({"error": "Json object posted not valid"})
    return result


@app.route("/verify", methods=["POST"])
def verify():
    result = {"error": "started..."}
    # ensure the file was properly uploaded to our endpoint
    if request.method == "POST":
        if request.get_json(force=True):
            data = request.get_json(force=True)["data"]
            try:
                waves = process(data)
                ps = check_enrolment(waves[:TRIM])
                pss = [ps[i:i + TRIM**2] for i in range(0, len(ps), TRIM**2)]
                en = []
                for p in pss:
                    enrolled = [i for i in p if i >= 0.00099999]
                    en.append(len(enrolled))

                cut = (len(ps) / TEMPS) / 2  # 80%
                which = max(en)
                idx = en.index(which)

                if which >= cut:
                    target_label = CLASSES[idx]
                    print(target_label)
                    return json.dumps({"success": str(target_label)})
                else:
                    print("failure")
                    return json.dumps({"failure": "Identity not enrolled"})
            except ValueError:
                return json.dumps({"error": "Couldn't successfully read content"})
        else:
            return json.dumps({"error": "Json object posted not valid"})

    return result


if __name__ == '__main__':
    load_templates()
    app.run()
