""" Main Module for handling the Datasets.
It handles loading of the Heartrates, applying the lopassfilter on them and detecting the maximaum values.
on top of that it manages the synchronization of the times between the heartrates and videos.
Finally it organizes where to cut the videos and to safe them.
It is also able to produce plots of the filtered heartrates.
"""

import os
from shutil import copy2
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, argrelmax, find_peaks_cwt
from itertools import islice
from cut_vid_fast import VideoHandler
# import environment_laptop as paths
import environment as paths
from open_face_wrapper import OpenFace
from multiprocessing import Pool
from scipy import signal
import scipy
import glob
import sys
from functools import partial




def lowpass_filter(data, Wn=0.014, N=3):
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
    try:
        data = data.astype(np.float)
    except ValueError as e:
        np.set_printoptions(threshold=np.nan)
        # print(data)
        raise e

    # N  = 5  # Filter order
    # Wn = 0.02 # Cutoff frequency
    # print(data.shape)
    B, A = signal.butter(N, Wn, output='ba')

    # Second, apply the filter
    tempf = signal.filtfilt(B,A, data)
    # print(tempf)
    return tempf


def plot_that2(frame_infos, show=True):
    """Plotting the heartrates, lowpass filter results and maxes in one plot.
        Parameters:
        -----------
        frame_info:  tuple of lopass results and maxes.
        show: decides if plt is shown or not after the function call.


        """
    # infos = (frame, (time, rate))
    for name, infos in frame_infos.items():

        print(name)
        ax = global_frame_dicts[name].plot("global_time", "rate", title=name, marker="x", kind="scatter", label="HR")
        print("Maxes: "+ str(infos[1].shape))
        infos[1].plot("global_time", "rate", color="g", ax=ax, kind="scatter", linewidth=3, label="Maxes")
        # global_frame_dicts[name].rolling(window=40, center=True).mean().plot("global_time", "rate", ax=ax, color="y", label="HR r Mean")
        infos[0].plot("global_time", "rate", color="r", ax=ax, label="HR lowpass")

        plt.title(name)

        if show:
            plt.show()
    # plt.title(title)


global_frame_dicts = None


def prep_frame(name, shift=True, window=[30,-20]):
    """ Prepares a Dataframe with Heartrates for further use.
    The frames stored in the global_frame_dicts will be accessed to get the
    current frame. Since we'r only interested in the first half of the exercises we need to detect where the first part of the trial ends.
    In order to do that we need to make the assumption, that there exists a gap somewhere between the timestamps. This Assumption is valid and works because the probands will leave the room and the reciever which protocolls the heartrates will loose the connection and does not record anything. To prevent detecting a gap that is too small we search for the biggest gap inside the frame.
    In the next step the maxima detection is performed.
    We always assume that there are 5 maxima we can detect (because of the 5 exercises). The lowpass filter will be applied adaptively until only 5 maxima are detected.


    Parameters:
    -----------
    name: String- key to access the fitting dataframe inside the global_frame_dicts.

    Return: name,(frame_tmpl,maxes_frame)
    -------
    name: see parameter

    frame_tmpl: DataFrame with the smoothed heatrates produced by the
                lowpass_filter

    maxes_frame: DataFrame with the local maxima.
    """

    # 2016-05-03-15-27-20.tide.csv
    # 2016-04-20-14-08-04.tide.csv
    # 2016-03-09-14-07-55.tide.csv
    # 2016-04-08-14-40-47.tide.csv
    # 2015-10-01-18-08-27.tide-heartrate.csv
    # 2016-05-03-14-57-08.tide.csv
    # 2015-10-02-10-13-34.tide-heartrate.csv
    # 2016-05-04-11-13-21.tide.csv
    # 2016-03-29-08-18-08.tide.csv
    # 2015-10-15-13-36-46.tide-heartrate.csv
    # 2016-04-08-17-16-03.tide.csv

    frame = global_frame_dicts[name]
    # Start the gap detection
    gap_diff = [0,0]
    current_row = 0
    current_value = 0
    all_gaps = []
    # print(frame.empty)

    for index, row in islice(frame.iterrows(), 30,frame.shape[0]-60):
        if (current_value == 0):
            current_value = row.global_time
            continue
        diff = row.global_time - current_value
        if gap_diff[1] < diff:
            gap_diff[0] = index
            gap_diff[1] = diff
            all_gaps.append(index)

        current_value = row.global_time

    # Extract the first part of the trial and remove noise inside the heartrates
    if "2016-03-29-08-18-08.tide.csv" in name:
        gap_diff[0] = all_gaps[4]
    # it seems like there is only the first part recordet
    elif name in ["2016-05-03-14-57-08.tide.csv",
        "2016-05-03-15-27-20.tide.csv",
        "2015-10-15-13-36-46.tide-heartrate.csv"]:
        gap_diff[0] = frame.shape[0]
    # else if
    frame_tmpl = frame.iloc[:gap_diff[0]]
    # frame_tmpl = frame_tmpl[frame_tmpl.rate > 40]
    # frame_tmpl = frame_tmpl[frame_tmpl.rate < 190]
    baseline = frame_tmpl.rate.iloc[:30].mean()

    # frame_tmpl.rate = frame_tmpl.rate.rolling(window=19, center=True).mean()

    # /homes/galberding/Dokumente/SS18/Ba/Database/heartrate_with_timestamps/2015-11-24-16-10-53.tide-heartrate.csv
    # print("Shift: {}, Window: {} Name: {}".format(shift, window ,name))
    tmp = lowpass_filter(frame_tmpl.rate.values[window[0]:window[1]])
    # print(tmp.shape)
    # print(frame_tmpl.shape)
    maxes = argrelmax(tmp)
    # print("first max: {} {}".format(maxes, len(maxes[0])))
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
    maxes = maxes[0] + window[0]
    # print(len(maxes))
    # print("maxes: {}".format((maxes)))
    maxes_frame = frame.rolling(window=21, center=True).mean().iloc[maxes]
    # print(maxes_frame)
    # print(maxes_frame)
    fst_max = frame.rolling(window=21, center=True).mean().iloc[maxes[0]]
    last_max = frame.rolling(window=21, center=True).mean().iloc[maxes[-1]]

    diff_start = fst_max.rate - baseline
    diff_end = last_max.rate - baseline
    # print("Name:, Diff_start: {} Diff_end: {}".format(diff_start, diff_end))
    # print(maxes[0])

    # first max is not below baseline but to early detected
    # skip baselinechek and shift right away
    if name in ["2015-10-16-15-20-53.tide-heartrate.csv",
        "2016-05-03-14-57-08.tide.csv",
        "2015-11-16-14-16-18.tide-heartrate.csv",
        "2015-10-01-18-08-27.tide-heartrate.csv",
        "2015-12-02-17-07-43.tide-heartrate.csv",
        # much noise hard to tell
        "2015-09-28-12-15-52.tide-heartrate.csv",
        "2016-03-16-09-05-31.tide.csv",
        "2016-03-29-08-18-08.tide.csv",
        # much noise and homogen
        "2015-10-05-13-12-17.tide-heartrate.csv",
        "2015-10-02-10-13-34.tide-heartrate.csv",
        "2016-03-16-16-18-25.tide.csv",
        "2015-10-07-12-07-44.tide-heartrate.csv",
        "2016-04-08-17-16-03.tide.csv",
        "2015-09-28-13-06-41.tide-heartrate.csv",
        "2015-10-01-18-08-27.tide-heartrate.csv",
        ] and shift:
        if name in ["2015-10-01-18-08-27.tide-heartrate.csv", "2016-03-16-16-18-25.tide.csv"]:
            return prep_frame(name, shift=False, window=[200, -20])

        if name in ["2015-10-07-12-07-44.tide-heartrate.csv", "2015-10-01-18-08-27.tide-heartrate.csv"]:
            return prep_frame(name, shift=False, window=[400, -20])

        if name is "2016-04-08-17-16-03.tide.csv":
            return prep_frame(data, shift=False, window=[1200, -20])


        return prep_frame(name, shift=False, window=[maxes[0], -20])

    tmp_window = [30,-20]
    if diff_start > 0:
        shift = False
    else:
        if diff_start < 0:
            tmp_window[0] = maxes[0]
        if name in "2016-04-20-14-08-04.tide.csv":
            if diff_end < 0:
                tmp_window[1] = maxes[-1]
    # print(tmp_window)
    if shift:
        return prep_frame(name, shift=False, window=tmp_window)
    # Contains multiple records and produces no good maxes
    # dropping one max results in ignoring the detection and skip it


    if name in ["2015-11-17-09-16-47.tide-heartrate.csv", "2015-10-13-15-09-06.tide-heartrate.csv", "2015-09-22-10-43-24.tide-heartrate.csv",
    # just two exercises shown
    "2016-04-14-11-03-42.tide.csv",
    # no clear detection\
    "2015-10-02-09-02-16.tide-heartrate.csv",
    "2015-11-26-13-07-16.tide-heartrate.csv"]:
        maxes_frame = maxes_frame.drop(maxes_frame.index[0])



    return name,(frame_tmpl,maxes_frame)


def load_heart_rates(load_first=True):
    """Loads all heartrates to dataframes.
    Loading will now include automatic smoothing via lowpass filter as well as maxima detection and cutting at specific gaps.

    Parameters:
    -----------

    load_first: only loads the first frame it is given


    """
    source_dir = paths.HEART_RATE_SRC_DIR
    frames = {}
    cut_indice = {}
    count = 0
    lfw_iter = os.scandir(path=source_dir)
    for i in lfw_iter:
        # load the heartrate fand diff_end > 0rom csv
        if(i.name[-4:] == ".csv"):
            frame = pd.read_csv(i.path, header=None, names=["global_time", "port", "local_time", "hrv1", "hrv2", "rate"], verbose=True)
            frames[i.name] = frame
            if load_first:
                break

    global global_frame_dicts
    global_frame_dicts = frames
    pool = Pool()
    results = pool.map(prep_frame, frames.keys())
    pool.close()
    pool.join()
    # print(results)
    # name: frame, (time, rate)
    results = dict(results)
    maxes = {}
    frames = {}
    for name,  info in results.items():
        frames[name] = info[0]
        # maxes[name] = list(zip(info[1].global_time, info[1].rate))
        if info[1].shape[0] != 5:
            maxes[name] = []
        else:
            # frames[name] = info[0]
            maxes[name] = list(zip(info[1].global_time, info[1].rate))

    return results, frames, maxes


def print_table(header, values):
    """Print table in markdown format.
    Parameters:
    -----------
    header: 1d list/array
    values: 2d list/array
            every sublist/array represents a row"""
    separator_str = "| ------- "
    header_str = "|"
    rows = []
    separator_str *= len(header)
    separator_str += "|"
    for i in header:
        header_str = header_str + str(i) + "|"

    for i in values:
        row = "| "
        for j in i:
            row += str(j) + "|"
        rows.append(row)

    print(header_str)
    print(separator_str)
    for i in rows:
        print(i)


def load_record_times(frame_dics):
    """Synchronizes the videos with the heartrates.
    Loads videonames and their timestamps and compares those timestamps with all heartrate timestamps.
    The heartrate file with closest timestamp will associated with the video.

    Parameter:
    ----------
    frame_dics: dataframes with heartrates.

    Return:
    --------
    dict with associated files.
    """

    record_time_path = paths.RECORD_TIMES_ISOMETRIC

    times = pd.read_csv(record_time_path, header=None, names=["file", "global_time", "date"])
    print(times.shape)
    ref_files = {}
    count = 0
    for index, row in times.iterrows():
        # nearest = [diff, name, stamp]
        nearest = [sys.maxsize, "name", 0]
        current_stamp = 0
        print(row.file)
        # if time.localtime(row.global_time).tm_year == 2016:
        #     continue

        for name, rate_frame in frame_dics.items():
            current_stamp = rate_frame.global_time.iloc[0]
            diff = abs(rate_frame.global_time.iloc[0] - row.global_time)

            if diff < nearest[0]:
                nearest[0] = diff
                nearest[1] = name
                nearest[2] = current_stamp

        print("{:02}\tresult for\t{} at\t{} to\t{} with ratefile: {}".format(count,row.file, row.global_time, nearest[2], name))
        count += 1
        ref_files[nearest[1]] = (row.file, row.global_time)
    # ref_files = (filename, timestamp where vid should start)
    return ref_files


global_maxes = None
global_record_times = None


def find_filetype_in_path(file_type, search_dir):
    """ Searches for the given filteype.

    Parameters:
    ----------
    file_type: String, name of filetype.
    search_dir: String, parentdirectory where to search in.

    Return:
    -------
    list of strings with paths to files with the given filetype ending.
    """
    file_path = []
    for path, subdir, files in os.walk(search_dir):
        # print("{} {} {}".format(path, subdir, files))
        for file in files:
            # print(file)
            if glob.fnmatch.fnmatch(file[-len(file_type):],file_type):
                file_path.append(os.path.join(path, file))
    return (file_path)


def exec_cut_vid(name):
    """Use Videohandler to cut a video and safe it.
    Parameters:
    -----------
    name: String, path to video


    Return:
    ------
    None or path to new video
    """
    vid = global_record_times[name]
    vid_name = vid[0] + ".mp4"
    # print(vid_name)
    vh = VideoHandler(vid_name, paths.VIDEO_SEARCH_DIR, paths.VIDEO_SNIP_SAFE_DIR)
    return vh.cut_vid(vid[1], global_maxes[name])

def exec_vid_max_extraction_threading():
    """ Cut all videos in parrallel.

    Return:
    -------
    list of pahts to videos
    """

    pool = Pool()
    results = pool.map(exec_cut_vid, global_record_times)
    pool.close()
    pool.join()
    return (results)

def exec_open_face(input_path, output_path, of_path):
    """Extract AUs for a given video.
    Parameters:
    -----------
    input_path: path to the video

    output_path: path to the directory where to save the AUs.

    of_path: path to OpenFace (binaries)
    """
    command = {"-f": input_path,
                "-out_dir": output_path,
                # "-out_dir": paths.OPEN_FACE_OUTPUT_DIR,
                "-aus": "-nobadaligned"}
    opf = OpenFace(of_path)
    opf.FeatureExtraction(command)


def exec_open_face_threading(path_to_video_snips, output_path, of_path):
    """ Extract Action Units from Videos.
    Parameters:
    -----------
    path_to_video_snips:

    output_path: path to the directory where to save the AUs.

    of_path: path to OpenFace (binaries)
    """

    snip_paths = find_filetype_in_path(".avi", path_to_video_snips)
    pool = Pool(5)
    # exe_of=partial(exec_open_face, output_path=paths.OPEN_FACE_OUTPUT_DIR)
    exe_of=partial(exec_open_face, output_path=output_path, of_path=of_path)
    results = pool.map(exe_of,snip_paths)
    pool.close()
    pool.join()
    print(results)

def setup():
    frame_infos, frames, maxes = load_heart_rates(load_first=False)
    record_times = ((load_record_times(global_frame_dicts)))
    global global_record_times
    global_record_times = record_times
    global global_maxes
    global_maxes = maxes
    # print(frame_dics)
    return frame_infos, maxes, record_times




if __name__ == '__main__':

    frame_infos, maxes, record_times = setup()

    # exec_open_face_threading(paths.VIDEO_SNIP_SAFE_DIR, paths.OPEN_FACE_OUTPUT_DIR, paths.OPEN_FACE_PATH)
    # print(record_times)
    # print(len(record_times))
    plot_that2(frame_infos, show=True, )
    # setup()
    # exec_vid_and_open_face_extraction()

    #
    # exec_vid_max_extraction_threading()
    # # print(init_sampel_paths)
    # exec_open_face_threading(paths.VIDEO_SNIP_SAFE_DIR)
    # exec_open_face_threading(init_sampel_paths)
    # exec_vid_max_extraction_threading()
    # exec_open_face_threading()
