"""Test the behaviour of the heartrates under the same conditions as in filemapper.py.
It is used to find out how to adjust heartrate samples with wrong detected maxes.
"""

import numpy as np
from datetime import datetime
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from itertools import islice
from scipy.signal import argrelextrema, argrelmax, find_peaks_cwt
# import peakutils
from detect_peaks import detect_peaks
import environment as paths

# TODO: What to do with:
# Out:
# /homes/galberding/Dokumente/SS18/Ba/Database/heartrate_with_timestamps/2015-11-25-16-08-12.tide-heartrate.csv

# Edited
# 2015-11-25-12-11-07.tide-heartrate.csv

# r = csv.DictReader(datadialect=csv.Sniffer().sniff(data.read(1000)))
# data.seek(0)
# temp needs to be converted from a "list" into a numpy array...


def lowpass_filter(data, Wn=0.014, N=2):
    """ Smooths the gitven data according to the lopass filter.

    Parameters
    ----------
    data: array like
    Wn: default 0.014 - is the cutoff frequence
    N: default 5 - is the order of the filter

    Return
    ------
    array with smoothed data.
    """
    data = np.array(data)
    data = data.astype(np.float)
    # N  = 5  # Filter order
    # Wn = 0.02 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')

    # Second, apply the filter
    tempf = signal.filtfilt(B,A, data)
    # print(tempf)
    return tempf

def greater(x,y):

    print(x)
    print(y)
    print(np.array_equal(x,y))
    return True

def find_maxes(X):
    up = "up"
    down = "down"
    gradiets = np.gradient(X)
    current = gradiets[0]
    gradient_dir = None
    # print(X.shape)
    # print(gradiets.shape)
    if current > 0:
        gradient_dir = up
    else:
        gradient_dir = down
    print(gradient_dir)

    maxes = []
    for i, gradient in enumerate(gradiets, 0):
        # print("Index: {}, Dir: {}, Value: {}, Gradient: {}".format(i+30, gradient_dir,X[i], gradient))
        if gradient < 0 and gradient_dir is up:
            # print("Maxima found")
            maxes.append(i-1)
            gradient_dir = down
        # elif gradient == 0 and gradient_dir is up:
        #     maxes.append(i)
        #     gradient = down
        elif gradient > 0 and gradient_dir is down:
            gradient_dir = up
    return np.array(maxes)


def prep_frame(data, shift=True, window=[30,-20]):
    frame = data
    gap_diff = [0,0]
    all_gaps = []
    all_diffs = []
    current_row = 0
    current_value = 0
    for index, row in islice(frame.iterrows(), 30,frame.shape[0]-60):
        if (current_value == 0):
            current_value = row.global_time
            continue
        diff = row.global_time - current_value
        if gap_diff[1] < diff:
            print("found gap at {} with {}".format(index, diff))
            gap_diff[0] = index
            all_diffs.append(diff)
            all_gaps.append(index)
            gap_diff[1] = diff

        current_value = row.global_time
    # cut_indice[i.name] = current_row
    # print(gap_diff[0])
    # print(all_gaps)
    # print(all_diffs)

    # if "2016-03-29-08-18-08.tide.csv" in name:
    # gap_diff[0] = all_gaps[4]
    # gap_diff[0] = frame.shape[0]
    # gap_diff[0] = all_gaps[4]

    frame_tmpl = frame.iloc[:gap_diff[0]]
    # frame_tmpl = frame_tmpl[frame_tmpl.rate > 40]
    # frame_tmpl = frame_tmpl[frame_tmpl.rate < 190]
    # frame_tmpl = [frame_tmpl.global_time.apply(pd.Series.nunique) == 1]
    # frame_tmpl.rate = frame_tmpl.rate.rolling(window=21, center=True).mean()
    # frame_tmpl = frame_tmpl.reset_index()
    baseline = frame_tmpl.rate.iloc[:30].mean()
    # print(baseline)
    # Handle Maxima:
    print("Shift: {}, Window: {}".format(shift, window ))
    tmp = lowpass_filter(frame_tmpl.rate.iloc[window[0]:window[1]])
    # print(tmp.shape)
    # print(frame_tmpl.shape)
    # np.set_printoptions(threshold=np.nan)
    # print(np.gradient(tmp))

    # maxes = find_maxes(tmp)
    maxes = argrelmax(tmp)
    print("first max: {} {}".format(maxes, len(maxes)))
    adjusted = False
    for n in range(3,9):
        Wn=0.014
        for i in range(100):
            # print(Wn)
            if len(maxes[0]) > 5:
                Wn -= 0.001
                tmp = lowpass_filter(frame_tmpl.rate.values[window[0]:window[1]],Wn, N=n)
                maxes = argrelmax(tmp)
                # print(tmp.shape)
                # print(frame_tmpl.shape)
            elif len(maxes[0]) < 5:
                Wn += 0.001
                tmp = lowpass_filter(frame_tmpl.rate.values[window[0]:window[1]], Wn, N=n)
                maxes = argrelmax(tmp)
                # print(tmp.shape)
                # print(frame_tmpl.shape)
            else:
                # print(tmp.shape)
                # print(frame_tmpl.shape)
                frame_tmpl.rate.iloc[window[0]:window[1]] = tmp
                adjusted = True
                break
        if adjusted:
            break
    print(maxes)




    # [print(frame_tmpl.rate[window[0]:window[1]].iloc[[i-1,i,i+1,i+2,i+3]]) for i in maxes[0]]

    maxes = maxes[0] + window[0]
    # print(len(maxes))
    # print("maxes: {}".format((maxes)))
    maxes_frame = frame.rolling(window=21, center=True).mean().iloc[maxes]
    # print(maxes_frame)
    print((maxes))
    fst_max = frame.rolling(window=21, center=True).mean().iloc[maxes[0]]
    last_max = frame.rolling(window=21, center=True).mean().iloc[maxes[-1]]

    diff_start = fst_max.rate - baseline
    diff_end = last_max.rate - baseline
    # print(diff_end)
    # print("Name:, Diff_start: {} Diff_end: {}".format(diff_start, diff_end))
    # print(maxes[0])
    # print(baseline)
    # print(maxes_frame)
    # [print(frame.iloc[[i-1,i,i+1]]) for i in maxes]

    # #
    #
    if shift:
        return prep_frame(data, shift=False, window=[200, -20])


    if diff_start > 0 :
        shift = False
    else:
        if diff_start < 0:
            window[0] = maxes[0]
        # if diff_end < 0:
        #     window[1] = maxes[-1]
    # shift = False
    # print(window)
    # print(shift)
    # tmp_window = [maxes[0], -20]
    # # return prep_frame(data, shift=False, window=tmp_window)




    # if shift:
    #     return prep_frame(data, shift=False, window=window)


    return (frame_tmpl,maxes_frame, all_gaps)



# data = pd.read_csv("/home/schorschi/Dokumente/SS18/Ba/Database/heartrate_with_timestamps/2015-11-25-16-08-12.tide-heartrate.csv", header=None, names=["global_time", "port", "local_time", "hrv1", "hrv2", "rate"], verbose=True)
# Works:
# data = pd.read_csv("/homes/galberding/Dokumente/SS18/Ba/Database/heartrate_with_timestamps/2016-03-15-09-03-51.tide.csv", header=None, names=["global_time", "port", "local_time", "hrv1", "hrv2", "rate"], verbose=True)
# Weird gap diff with 646
# data = pd.read_csv("/homes/galberding/Dokumente/SS18/Ba/Database/heartrate_with_timestamps/2016-05-03-15-27-20.tide.csv", header=None, names=["global_time", "port", "local_time", "hrv1", "hrv2", "rate"], verbose=True)
# Adjust with window
data = pd.read_csv(paths.ba_path+"/Database/heartrate_with_timestamps/2016-04-20-14-08-04.tide.csv", header=None, names=["global_time", "port", "local_time", "hrv1", "hrv2", "rate"], verbose=True)
# Contains multiple exercises??
# data = pd.read_csv("/homes/galberding/Dokumente/SS18/Ba/Database/heartrate_with_timestamps/2016-03-16-16-18-25.tide.csv", header=None, names=["global_time", "port", "local_time", "hrv1", "hrv2", "rate"], verbose=True)

frame, maxes, all_gaps = prep_frame(data)

ax = data.reset_index().plot("index", "rate", marker="x", kind="scatter")
ax.set_xlabel("Index")
ax.set_ylabel("Heartrate")
ax.set_title("2016-04-20-14-08-04.tide.csv")
# ax = data.rate.hist(bins=data.shape[0])
# gaps = [31, 59, 434, 1356, 2221, 2256, 4141]
# print(data.global_time.iloc[all_gaps])


# frame.reset_index().plot("index", "rate", color="r", kind="scatter", ax=ax )
# maxes.reset_index().plot("index", "rate", color="g", ax=ax, kind="scatter")
# ax.vlines(data.reset_index().index[all_gaps], 0, 200)
print(frame.shape)
print(frame.rate.shape)


# plt.scatter(np.array(range(frame.rate.shape[0])), frame.rate.values)
plt.savefig(paths.BA_PICTURES+"hr_bsp.png")
plt.show()
