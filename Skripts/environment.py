""" This module contains all paths which will be used by the scripts.
It will take the environment variable BA_PATH which contains the path to this Repo and uses it to adjust the relative paths
listed below.
"""

import os

ba_path = os.environ["BA_PATH"]


HEART_RATE_SRC_DIR = os.path.join(ba_path,"Database/heartrate_with_timestamps/")
# HEART_RATE_BEST_SAMPLE = ""
# HEART_RATE_BEST_SAMPLES =""
# HEART_RATE_CONFUSING_SAMPLE = ""
# RECORD_TIMES_ISOMETRIC = "/vol/prt/analysis/heartrate_au/recordTimes_isometric.csv"
RECORD_TIMES_ISOMETRIC = os.path.join(ba_path, "Database/recordTimes_isometric.csv")


# VIDEO_SEARCH_DIR = "/homes/galberding/Dokumente/SS18/Ba/exclude/search"
VIDEO_SEARCH_DIR = "/vol/prt/analysis/heartrate_au/video/front/"
VIDEO_SNIPPETS = ""
VIDEO_SNIP_SAFE_DIR =  "/homes/galberding/Dokumente/SS18/Ba/exclude/video_out/"
# VIDEO_SNIP_SAFE_DIR =  "/vol/prt/analysis/vid_out/"
# VIDEO_TEST_SNIP_SAFE_DIR =  "/homes/galberding/Dokumente/SS18/Ba/exclude/video_out_test/"
# VIDEO_BROKEN_SAMPLES = "../Database/broken_samples.csv"
# VIDEO_PATH_COLLECTION = "../Database/video_path_collection.csv"
VIDEO_INIT_SAMPLES_DIR = os.path.join(ba_path, "exclude/Start_samples/")
VIDEO_INIT_SAMPLES_bf_tf_DIR = os.path.join(ba_path, "exclude/Start_samples_2016/")
VIDEO_PROGRESS_SAMPLES_bf_tf_DIR = os.path.join(ba_path, "exclude/Sample_ex4/")

# Important this variable needs to be set to get vid_cut_fast.py to work
OPEN_FACE_PATH = os.path.join(ba_path, "exclude/links/OpenFace")


# OPEN_FACE_INPUT_VID = "/vol/prt/analysis/vid_out/s10/s10_snip_15_1444213400_122.35.avi"
OPEN_FACE_OUTPUT_DIR = os.path.join(ba_path, "Database/open_face_out/")
OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES = os.path.join(ba_path, "Database/open_face_out_trial2/init_samples/")
OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES_bf_tf = os.path.join(ba_path, "Database/open_face_out_trial2/init_tf_bf/")
OPEN_FACE_OUTPUT_DIR_PROGRESS_SAMPLES = os.path.join(ba_path, "Database/open_face_out_trial2/progress_samples/")
OPEN_FACE_OUTPUT_DIR_PROGRESS_SAMPLES_bf_tf = os.path.join(ba_path, "Database/open_face_out_trial2/ex4/")
OPEN_FACE_OUTPUT_ALL_VIDS = "/vol/prt/analysis/open_face_out_all/"
OPEN_FACE_PRT_PROGRESS_LOWPASS_V1 = "/vol/prt/analysis/AUs/open_face_out_lowpass_v1"




STATS_PLOT_DIR = os.path.join(ba_path, "Database/stats_dir/plots/")
STATS_DIR = os.path.join(ba_path, "Database/stats_dir/")
STATS_TEST_SAMPLE = "/homes/galberding/Dokumente/SS18/Ba/exclude/s_20_snip_08_1449233649_131.9.csv"

# Contains of csvs for each AU in the 6 classes.
STATS_AUs_BY_CLASSES = os.path.join(ba_path,"Database/AUs_by_classes/")
STATS_AUs_BY_CLASSES_BORG = os.path.join(ba_path,"Database/AUs_by_classes_borg/")
STATS_AUs_BY_CLASSES_BORG_15 = os.path.join(ba_path,"Database/AUs_by_classes_borg_15/")
STATS_AUs_BY_CLASSES_BORG_16 = os.path.join(ba_path,"Database/AUs_by_classes_borg_16/")
STATS_AUs_BY_CLASSES_BORG_17 = os.path.join(ba_path,"Database/AUs_by_classes_borg_17/")
STATS_AUs_BY_CLASSES_BORG_18 = os.path.join(ba_path,"Database/AUs_by_classes_borg_18/")
STATS_BORG_USER_R1 = os.path.join(ba_path, "Database/borg_user_rating.csv")
STATS_BORG_USER_R2 = os.path.join(ba_path, "Database/borg_user_rating_2.csv")


BA_PICTURES = os.path.join(ba_path, "Database/Pictures/")
BA_PICTURES_SVM = os.path.join(ba_path, "Database/Pictures/SVM/")
