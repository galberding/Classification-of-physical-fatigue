""" Module for analysing the Action Units in csvs, produced by Open Face"""
import pandas as pd
import numpy as np
import environment as paths
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
from time import sleep
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from statsmodels.multivariate import manova
from sklearn.model_selection import StratifiedShuffleSplit
import re
from sklearn.metrics import average_precision_score, precision_recall_curve


labels_c = ['AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']

labels = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']

# name: [(frame, hist)]


def load_frame(name):
    """ Load AU csv file to DataFrame.
    The csv file will be loaded and after that the Frames will be prepared after
    the following criterias:
    1) A face needs to be detected (success = 1)
    2) The face needs to be visible for at least a second to avoid noise
    After that each Dataframe will be summed up along the 0 axis and divided
    by the number of frame while the face was visible. This is neccesary to normalize
    all AUs. The result gets saved in hist

    Parameters:
    -----------
    name: String - path to csv file with AUs.

    Return:
    --------
    name: look parameters.
    frame: DataFrame - filtered by the criterias.
    hist: Series - normalized AUs

    Note:
    ´Before using: sed -i -e "s/ //g" *.csv´
    """
    frame = pd.read_csv(name)
    try:
        frame = frame[frame.success == 1]
    except AttributeError as e:

        raise AttributeError("Problem with the output of OpenFace.\nPlease use `sed -i -e \"s/ //g\" *.csv` for removing the whitespaces inside the csv files.") from e
    # frame = frame[frame.confidence > 0.4]
    # print(frame[labels].sum())
    hist = frame[labels].sum() / frame.shape[0]
    if frame.shape[0] < 25:
        hist = pd.Series(index=labels)

    return name,(frame, hist)


def find_filetype_in_path(file_type, search_dir, substr=None):
    """Search for files with a given suffix.
    The function is called find filetype because its handy to give it a file ending
    like .csv or .mp4 and it will search in the given directory for all files with this suffix.
    Its possible to filter the results to match specified substrings.

    Parameters:
    -----------
    file_type: String - Filetype or suffix of the files to search for.

    search_dir: String - Search directory.

    substr: list of Strings - The filenames need to match at leat one of the
            substrings to get accepted.

    Return:
    -------
    file_paths: list of filepaths

    """
    file_paths = []
    for path, subdir, files in os.walk(search_dir):
        for file in files:
            if glob.fnmatch.fnmatch(file[-len(file_type):],file_type):
                if substr:
                    for sub in substr:
                        if sub in file:
                            file_paths.append(os.path.join(path, file))
                else:
                    file_paths.append(os.path.join(path, file))

    return (file_paths)


def load_csv_to_frame(search_paths, substr=None):
    """ Load AUs in csv format to DataFrames.
    The AUs gets filtered s.t. only one normalized histogram is produced for each exercise (and initial state). You can read more about the filtering of the AUs in load_frame().
    The method is able to load the data in two different kinds:
    1) Loads all AUs from the given paths and put them together in a dict s.t. each face gets 6 AU-hists assigned with the label of their exercice.
    2) Loads all AUs in the given path regardless which class and combines them to a single DataFrame of hists.

    Parameters:
    -----------
    search_paths: list of path where to search for AUs in csv format.

    substr: Default None, List of strings. The filename of the AUs needs to match at least one of those Strings to be loaded. Availiable are:
        snip_00 - snip04, describing the exercise class,
        snip_init - describing the initial class (AUs before the exercises)

    Return:
    -------
    face_coll: dict of tuples.
                Key - name assigned to a specific face
                Value[0] - class in which the Value[1] belogs to [-1,4] where -1 is the
                initial and [0,4] the exercise class.
                Value[1] - Series, hist of AUs.

     collect_frame: DataFrame of hists.


     Note:
     -----
     The hists in the face_coll can be a Series of NaN or 0. This happens when no AUs where detected for one exercise.
     Those values are already dropped in the collect_frame.
    """

    file_names = []
    for path in search_paths:
        file_names += find_filetype_in_path(".csv", path, substr=substr)

    # names = []
    # for path_with_name in file_names:
    #     for path in search_paths:
    #         if path in path_with_name:
    #             pattern = "{}(.*)_snip".format(path)
    #             res = re.search(pattern, path_with_name)
    #             # names.append(name)
    #             names.append(res.group(1))
    #             break
    pool = Pool(4)
    # print(file_names)
    results = pool.map(load_frame,file_names)
    pool.close()
    pool.join()
    collect_frame = pd.DataFrame(columns=labels)

    face_coll = {}
    counter = 0
    new_name = None
    for name, value in results:
        if value[0].empty:
            continue
        sample_class = 0
        for path in search_paths:
            if path in name:
                if "snip_init" in name:
                    sample_class = -1
                else:
                    pos = name.find("_snip_")
                    sample_class = (int(name[pos+6:pos+8]))
                pattern = "{}(.*)_snip".format(path)
                res = re.search(pattern, name)
                tmp = res.group(1)
                # Check if it is b/s/t or tf/bf:
                new_name = re.sub("_", "", tmp)
                new_name = new_name[:3] if sum(c.isalpha() for c in new_name) == 1 else new_name[:4]
                if  new_name not in face_coll.keys():
                    face_coll[new_name] = []
                break
        hist = value[1].values
        face_coll[new_name].append((sample_class, hist))
        collect_frame.loc[counter] = hist
        counter +=1

    return face_coll, collect_frame.dropna()

# TODO: Either throw it away or overdo it to create multi class clustering but it is nearly useless because clustering is only on pca data
# TODO: Print pca data from all classes to present data
# TODO: try svm on pca data
def plot_pca_hist_pointdistr_clusters():

    # All Vids
    df_all_points_1 = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR])[1]
    df_all_points_2 = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES, paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES_bf_tf])[1]


    N_progress = df_all_points_1.shape[0]
    N_init = df_all_points_2.shape[0]
    print("Progress: {} Init: {}".format(N_progress, N_init))
    # append_dataframe(df_all_points_1, df_all_points_2)
    for index, row in df_all_points_2.iterrows():
        df_all_points_1.loc[N_progress+index] = row

    pca = PCA(n_components=18)
    pca_res = pca.fit_transform(StandardScaler().fit_transform(df_all_points_1.iloc[:,17:]))
    # pca_res = pca.fit_transform((df_all_points_1.iloc[:,16:]))

    kmeans = KMeans(n_clusters=2, random_state=0).fit((pca_res[:N_progress]))
    cluster_labels = kmeans.predict(pca_res)
    # print(cluster_labels)

    df_cluster0 = pca_res[cluster_labels == 0]
    # print(df_cluster0)
    df_cluster1 = pca_res[cluster_labels == 1]

    plt.hist((pca_res[:N_progress,0], pca_res[N_progress:,0]), bins=40)
    plt.show()
    plt.hist((pca_res[:N_progress,1], pca_res[N_progress:,1]), bins=40)
    plt.show()
    pca = PCA(n_components=18)
    pca_res = pca.fit_transform(StandardScaler().fit_transform(df_all_points_1.iloc[:,17:]))
    # Plot of distribution of the points from the different classes
    plt.scatter(pca_res[:N_progress,0], pca_res[:N_progress,1], label="Ex samples")
    plt.scatter(pca_res[N_progress:,0], pca_res[N_progress:,1], label="Init Faces")
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], label="Cluster Centers", color="m")
    plt.title("AU_c only, all Samples, no StandardScaler")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.show()
    plt.gcf().clear()

    # Cluster Plot
    plt.scatter(df_cluster0[:,0], df_cluster0[:,1], label="Cluster 0")
    plt.scatter(df_cluster1[:,0], df_cluster1[:,1], label="Cluster 1")
    plt.scatter(kmeans.cluster_centers_[0,0], kmeans.cluster_centers_[0,1], label="Cluster Center 0")
    plt.scatter(kmeans.cluster_centers_[1,0], kmeans.cluster_centers_[1,1], label="Cluster Center 1")
    plt.title("AU_c only, all Samples clustering, Kmeans, no StandardScaler")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.show()



def prep_data_inter_faces(choose=False, c_class=0):
    """ Load AUs and create a dataset with 6 classes.
    By default all classes will be loaded and concatenated after each other.
    The labels will be generated automatically.
    The init class will always be loaded with label 0,
    the first exercise class with 1 and so on.

    NOTE:
    Don't get confused by choosing the c_class where you choose the classes in [0-4]

    Parameters:
    ---------------
    choose:   Default False. If True only one class will be loaded.
                The class can be chosen by the c_class parameter.

    c_class:    Default 0. When choose is True the c_class parameter will
    dertermine which class will be loaded.
    The choosable classes are in range(5).
    This is, the different classes are labeled by their name
    (snip_00, snip_01, ...) and by choosing the c_class all samples
    with the suffix c_class will be loaded.

    Return:
    ---------------
    X: Datapoints of all classes.
    y: labels
    """
    # load AUs from init faces.
    names, init_class = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES, paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES_bf_tf])
    # print(names)
    # load AUs from exercises:
    # only load selected class
    if choose:
        names, ex_c = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR], substr=["snip_0"+str(c_class)])
        ex_classes = [ex_c]
    # load all classes
    else:
        ex_classes = []
        for i in range(5):
            names, ex_c = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR], substr=["snip_0{}".format(i)])
            ex_classes.append(ex_c)


    all_classes = [init_class] + ex_classes

    # Create dataset
    y = np.concatenate([np.full(all_classes[i].shape[0], i) for i in range(len(all_classes))])
    X = np.concatenate(all_classes, axis=0)

    return X, y


def prep_data_intra_face():
    """ Create Array with differences for each class.
    It basically substracts the action unit from thefirst class to all other classes.
    The diffrence is taken from each proband induividually.
    
    Return:
    ------
    classes with differrences.
    """
    classes = {0:[], 1:[], 2:[], 3:[], 4:[]}
    faces = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES, paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES_bf_tf, paths.OPEN_FACE_OUTPUT_DIR])[0]
    # print(len(faces.items()))
    for name, val in faces.items():

        face_states = dict(val)

        if len(face_states.keys()) == 1:
            continue
        elif -1 not in face_states.keys():
            continue
        elif np.isnan(np.sum(face_states[-1])):
            continue
        # print("{}, {}".format(name, face_states.keys()))


        for i in range(len(face_states.keys())-1):
            if i not in face_states.keys():
                continue
            elif (np.isnan(np.sum(face_states[i]))):
                continue

            diff = (face_states[i] - face_states[-1])
            classes[i].append(diff)
    return classes

def prep_data_inter_faces_borg(threshold=15):
    """ Same as prep_data_inter_faces but excludes samples which are below the threshhold."""
    
    
    borg1 = pd.read_csv(paths.STATS_BORG_USER_R1)
    borg2 = pd.read_csv(paths.STATS_BORG_USER_R2)

    frame_size = borg1.shape[0]
    for index, row in borg2.iterrows():
        borg1.loc[frame_size+index] = row
    classes = {0:[],1:[],2:[], 3:[], 4:[], 5:[]}
    borg_labels = ["BORGI","BORGII","BORGIII","BORGIV","BORGV"]
    # print(borg1.VP)
    faces = load_csv_to_frame([paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES, paths.OPEN_FACE_OUTPUT_DIR_INIT_SAMPLES_bf_tf, paths.OPEN_FACE_OUTPUT_DIR])[0]
    # print(len(faces.items()))
    for name, val in faces.items():
        face_states = dict(val)
        # print(face_states)
        borg_info = (borg1[borg1["VP"] == name])
        # print(borg_info[borg_labels[0]].values)
        if -1 in face_states.keys():
            if not np.isnan(np.sum(face_states[-1])):
                classes[0].append(face_states[-1])

        for i in range(5):
            if (i) in face_states.keys():
                if borg_info.empty:
                    continue
                # print(borg_info[borg_labels[i]].values[0] > (np.concatenate(Y)))
                if (borg_info[borg_labels[i]].values[0] > threshold):
                    # print(face_states[i])
                    if np.isnan(np.sum(face_states[i])):
                        continue
                    classes[i+1].append(face_states[i])
    X = []
    Y = []
    for key, val in classes.items():
        if val:
            x = (np.array(val))
            X.append(x)
            y = np.full(x.shape[0], key)
            Y.append(y)
    X = (np.concatenate(X))
    Y = (np.concatenate(Y))
    # np.set_printoptions(threshold=np.nan)
    # # print(X[~np.isnan(X).any(axis=1)])
    # print(X.shape)
    # print(Y.shape)
    # X = np.concatenate((X,Y.reshape(X.shape[0],1)), axis=1)
    # print(X)
    # X = X[:,:-1]
    # Y = X[:,-1]
    # print(X)
    # print(Y)

    return X, Y


#TODO: deprecated and out of use
def cv_svm_all_classes(X, y, title="AU_c only"):
    """ Train SVM on given AUs.
    Currently only working with two classes to calculate the precision.
    Parameters:
    -----------
    X: array
    y: labels
    """
    from sklearn.metrics import recall_score
    from sklearn import preprocessing
    from sklearn.metrics import classification_report, roc_curve, auc
    import scipy
    from roc_test import cross_validate_roc
    # Create train and test set
    X = StandardScaler().fit_transform(X)
    # X = preprocessing.minmax_scale(X,axis=0)
    # print(y)

    # Balance clases:
    N_0 = y[y == 0].shape[0]
    n = N_0 // 5 +1
    seed = 1
    shift = N_0

    samples = []
    for i in range(1,6):
        np.random.seed(seed)
        samples.append(np.random.choice(X[y == i].shape[0], n) + shift)
        shift += X[y==i].shape[0]
    samples = np.concatenate(samples)
    X_c = np.concatenate((X[y==0], X[samples]), axis=0)
    y_c = np.concatenate([np.zeros(y[y==0].shape), np.ones(samples.shape)])

    # print(samples.shape)
    # print(y_c)
    # np.random.seed(seed)
    # samples = np.random.choice(X[y == 1].shape[0], y[y==0].shape[0]) + y[y==0].shape[0]
    # X_c = np.concatenate((X[y==0], X[samples]), axis=0)
    # y_c = np.concatenate([np.zeros(y[y==0].shape), np.ones(y[y==0].shape)])

    X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.33, random_state=seed)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

    parameters = {"kernel": ("linear",),
                'class_weight':['balanced', None],
                "C": np.linspace(1,10, 60),
                # "gamma":  2**np.linspace(-9,2, 60),
                }
                # "degree": [1,2,3,4,5,6,7,8,9,10],
                # "coef0": np.linspace(0.0, 0.5, 5)}
    # print(list(scipy.stats.expon(scale=100)))
    # parameters = {'C': scipy.stats.expon(scale=100),
    #     'gamma': scipy.stats.expon(scale=.1),
    #     'kernel': ['rbf'],
    #     'class_weight':['balanced', None]}
    # parameters = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    # svc = svm.SVC(probability=True, class_weight="balanced")
    svc = svm.SVC(probability=True)
    # cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    # clf = GridSearchCV(svc, parameters, cv=cv, n_jobs=-1, scoring='accuracy', verbose=1)
    clf = GridSearchCV(svc, parameters,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")

    clf.fit(X_train, y_train)
    # print(clf.cv_results_)
    print(clf.best_params_)

    # clf = clf.best_estimator_
    # clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(precision_recall_curve(y_test, clf.predict(X_test)))
    print("Train Score: {}\tTest Score: {} Precision: {}, Recall: {}\n".format(
        clf.score(X_train, y_train),
        clf.score(X_test, y_test),
        average_precision_score(y_test, y_pred), recall_score(y_test, y_pred, average='weighted')))
    print(classification_report(y_test, y_pred))
    # y_pred_proba = np.max(clf.predict_proba(X_test), axis=1)
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    # print(y_test)
    # print(y_pred)
    # print((y_pred_proba))
    print("Class 0: {}, Class 1: {}".format(y_test[y_test==0].shape[0],y_test[y_test==1].shape[0]))
    # clf = svm.SVC(gamma=clf.best_params_["gamma"], C=clf.best_params_["C"], class_weight=clf.best_params_["class_weight"], kernel='rbf', probability=True)

    # np.random.seed(1)
    # samples = np.random.choice(X[y == 1].shape[0], y[y==0].shape[0]) + y[y==0].shape[0]
    # X_c1 = np.concatenate((X[y==0], X[samples]), axis=0)
    # y_c1 = np.concatenate([np.zeros(y[y==0].shape), np.ones(y[y==0].shape)])
    # np.set_printoptions(threshold=np.nan)
    # print(X_c == X_c1)
    # cross_validate_roc(X,y, clf)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, )
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    # print(fpr)
    # print(tpr)
    plt.plot(fpr, tpr, color='b',
             label=r'AUC = {}'.format(roc_auc),lw=2, alpha=.8)
    # plt.plot(fpr, tpr, color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC-Curve '+ title)
    plt.legend()
    # plt.show()


#TODO: deprecated and out of use
def svm_for_all(ax, title, parameters, rounds, sel_features=None, use_all_AUs=True, borg_thresh=0, intensitys=False):
    """ SVM applied to all classes vs init.
    That is, the SVM will be trained on the init and one of the
    other ex classes. """
    from svm_train import cross_validate_roc

    print(title)
    X, y = prep_data_inter_faces_borg(threshold=borg_thresh)
    if sel_features is not None:
        X = X[:,sel_features]
        print(X.shape)
    else:
        if use_all_AUs:
            X = X[:,:]
        else:
            if not intensitys:
                X = X[:,17:]
            else:
                X = X[:,:17]


    # y[y > 0] = 1
    # Balance the classes by choosing random samples

    return cross_validate_roc(ax, X, y, parameters, title, rounds=rounds)
    # plt.show()
    # cv_svm_all_classes(X, y, title=title)



def make_box_plot(y_data, xtick_labels, diff_class, ):
    base_color = '#539caf'
    median_color = '#297083'
    x_label = 'AUs'
    y_label = 'Actual diff'
    title = diff_class
    _, ax = plt.subplots()
    ax.boxplot(y_data
               # patch_artist must be True to control box fill
               , patch_artist = True
               # Properties of median line
               , medianprops = {'color': median_color}
               # Properties of box
               , boxprops = {'color': base_color, 'facecolor': base_color}
               # Properties of whiskers
               , whiskerprops = {'color': base_color}
               # Properties of whisker caps
               , capprops = {'color': base_color}, showmeans=True)

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.axhline()
    # ax.set_ylabel(y_label)
    # ax.set_xlabel(x_label)
    ax.set_title(title)
    # plt.legend()
    return ax

def box_plot_AUs_diff():
    """ Plots the Diff for each AU in the different classes.
    The plots will contain 5 Boxes representing the difference of
    one AU in the init class to all other classes"""
    diff_classes = prep_data_intra_face()
    diff0 = np.array(diff_classes[0])
    diff1 = np.array(diff_classes[1])
    diff2 = np.array(diff_classes[2])
    diff3 = np.array(diff_classes[3])
    diff4 = np.array(diff_classes[4])

    for i in range(0,diff0.shape[1]):
        y = [diff0.T[i],diff1.T[i],diff2.T[i],diff3.T[i],diff4.T[i],]
        # print(y)
        make_box_plot(y, [1,2,3,4,5],labels[i] )
        plt.show()


def load_AUs_by_class(path_to_au_by_classes):
    """ Prepare data for ANOVA.
    
    Parameters:
    -----------
    path_to_au_by_classes: String, path to AUs that are sortet by their classes and eventually by the Borg-Value.
    
    Return:
    -------
    dataframe with actionunits and their lables
    """
    
    AU_data = find_filetype_in_path(".csv", path_to_au_by_classes, substr=None)
    # AU_data = find_filetype_in_path(".csv", paths.STATS_AUs_BY_CLASSES_BORG_15, substr=None)
    AU_dict = {}
    for path in AU_data:
        # AU_dict[path[]]
        AU_dict[path[-10:-4]] = pd.read_csv(path)
    return (AU_dict)


def anova_ttest_results(path_to_au_by_classes, return_AU_info=False):
    """ Applies Anova to the given data.
    After selecting a significant AU a t-test decides if the significance appears between  class 0 and any other class.

    Parameters:
    -----------
    path_to_au_by_classes: path to CSV where the AUs are saved by their classes.

    Return:
    -------
    relevant_labels: Names of significant AUs

    count: Counter how often a significance was detected inside one class compared to the 0 class.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from scipy import stats

    count = {"1":0, "2":0, "3":0, "4":0, "5":0}
    AU_info = {}
    # load AU for
    AU_dict = load_AUs_by_class(path_to_au_by_classes)
    relevant_labels = []

    for name, frame in AU_dict.items():
        # print(frame.ex[frame.ex != 0].shape)
        # print(frame.AU)
        # frame.AU = StandardScaler().fit_transform(frame.AU.values)
        for i in range(6):
            frame.ex[frame.ex == i] = str(i)

        mod = ols("AU ~ ex", data=frame).fit()
        aov_table = sm.stats.anova_lm(mod, type=2)
        if ((aov_table.iloc[0].values[-1])) <= 0.05:
            # print("=================================="+name+"==================================")
            # print(aov_table)

            sig = False
            # print(pairwise_tukeyhsd(frame.AU, frame.ex))
            # print(pairwise_tukeyhsd(frame.AU, frame.ex))
            print(aov_table)
            for i in range(1,6):
                AUs = np.concatenate((frame.AU[frame.ex == "0"], frame.AU[frame.ex == str(i)]))
                # print(AUs)
                y = np.concatenate((np.zeros(frame.ex[frame.ex == "0"].shape), np.full(frame.ex[frame.ex == str(i)].shape, str(i))))
                # print(y[y=="0.0"].shape)
                # print(y[y!="0.0"].shape)
                # print(pairwise_tukeyhsd(AUs, y))
                # create classes
                res = pairwise_tukeyhsd(AUs, y)
                # print(res)
                # print((aov_table))
                # print((aov_table["PR(>F)"].iloc[0]))
                # print((aov_table.F.iloc[0]))
                # print("{} {:.4f} {:.4f}".format("tee",res.meandiffs[0], aov_table.F.iloc[0]))


                # Print results in latex table format
                # if i == 1:
                #     print("${}$&{:.5f}&{:.5f}& {:.1f}&{:.1f}&{:.4f}&{:.4f}&{:.4f}&{} \\\\".format(name, aov_table.F.iloc[0], aov_table["PR(>F)"].iloc[0], 0.0, float(i),res.meandiffs[0], res.confint[0][0], res.confint[0][1], res.reject[0]))
                # elif i == 5:
                #     print("&&& {:.2f}&{:.2f}&{:.4f}&{:.4f}&{:.4f}&{} \\\\ \\hline".format(0.0, float(i),res.meandiffs[0], res.confint[0][0], res.confint[0][1], res.reject[0]))
                #
                # else:
                #     print("&&& {:.2f}&{:.2f}&{:.4f}&{:.4f}&{:.4f}&{} \\\\".format(0.0, float(i),res.meandiffs[0], res.confint[0][0], res.confint[0][1], res.reject[0]))

                # print("meandiff", "lower", "upper", "reject")

                # print(stats.shapiro(AUs))
                # print(stats.shapiro([1,1,1,1,1,1,1,1,1,1,0]))


                if name not in AU_info.keys():
                    AU_info[name] = [(i,res.meandiffs[0], res.confint[0][0], res.confint[0][1], res.reject[0])]
                else:
                    AU_info[name].append((i,res.meandiffs[0], res.confint[0][0], res.confint[0][1], res.reject[0]))
                if (True in res.reject):

                    sig = True
                    count[str(i)] += 1


            if sig:
                relevant_labels.append(name)
            else:
                del AU_info[name]

            # print(AU_info)
            # esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])


            # print(aov_table)
            # print("Eta squared: "+ str(esq_sm))

            # Apply t-Test for all 6 classes

    # for i in range(1, 6):
    #     print(str(i)+": ", count[str(i)], count[str(i)]/len(relevant_labels), frame.ex[frame.ex == str(i)].shape)

    return relevant_labels, count

def save_AUs_to_csv():
    """ Safe AUs by their classes to csv file"""
    X, y = prep_data_inter_faces_borg(threshold=16)
    # X, y = prep_data_inter_faces(choose=False, c_class=0)
    classes = []
    for j in range(len(labels)):
        c = []
        for i in range(6):
            c.append(np.concatenate((X[y==i,j][np.newaxis], y[y == i][np.newaxis]), axis=0).T)
        print(labels[j])
        data = (np.concatenate(c, axis=0))
        np.savetxt(paths.STATS_AUs_BY_CLASSES_BORG_16+labels[j]+".csv",data, header='AU,ex', delimiter=',')


def plot_svm(condition, rounds):
    """ Plot svm results for the different Borg levels (0, 15, 16, 17).

        Parameters:
        -----------
        codition:   \"anova\" - plots results for all anova subsets
                    \"AU_all\"- plots with all features
                    \"AU_c_only"- plots with only AU_c features
                    \"AU_r_only\"
        rounds: vary the rounds of random testing
    """
    para_comb = {"kernel": ("rbf",),
             'class_weight':['balanced', None],
             "C": np.linspace(1,10, 60),
             # "gamma":  2**np.linspace(-9,2, 60),
             }
    # rounds = rounds
    fig, ax = plt.subplots()
    anova_data = [paths.STATS_AUs_BY_CLASSES, paths.STATS_AUs_BY_CLASSES_BORG_15, paths.STATS_AUs_BY_CLASSES_BORG_16, paths.STATS_AUs_BY_CLASSES_BORG_17
    ]

    threshs = [0, 15, 16, 17]
    borgs = {}
    for i, thresh in enumerate(threshs):

        lab, count = anova_ttest_results(anova_data[i])
        # print(lab)
        # thresh = 0
        borgs[thresh] = (lab, count)
        idxs = [i for i, e in enumerate(labels) if e in lab]

        if condition is "anova":
            ax = svm_for_all(ax, "Borg: "+ str(thresh), para_comb, rounds, sel_features=idxs, borg_thresh=thresh)
        elif condition is "AU_all":
            ax = svm_for_all(ax, "Borg: "+ str(thresh), para_comb, rounds, borg_thresh=thresh, use_all_AUs=True)
        elif condition is "AU_c_only":
            ax = svm_for_all(ax, "Borg: "+ str(thresh), para_comb, rounds, borg_thresh=thresh, use_all_AUs=False)
        elif condition is "AU_r_only":
            ax = svm_for_all(ax, "Borg: "+ str(thresh), para_comb, rounds, borg_thresh=thresh, use_all_AUs=False, intensitys=True)
        else:
            print("No known condition!")


    ax.plot([0, 1], [0, 1], linestyle='--', lw=2,
             label='Luck', alpha=.8)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('SVM ROC-Curve '+ condition)
    ax.legend()

    # fig.savefig(paths.BA_PICTURES_SVM+condition+".pdf")
    plt.show()
    return borgs


if __name__ == '__main__':
    # labels = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']

    # save_AUs_to_csv()
    # lab Borg significant:
    # lab = ['AU04_r', 'AU10_c', 'AU01_c', 'AU02_c', 'AU14_c', 'AU04_c', 'AU20_r', 'AU23_c']
    # lab = ['AU23_c', 'AU14_c', 'AU02_c', 'AU05_c', 'AU04_r', 'AU15_r', 'AU01_c', 'AU10_c', 'AU20_r', 'AU04_c' ]
    # print(idxs)
    # print("All C Features")




    # print(anova_ttest_results(paths.STATS_AUs_BY_CLASSES))

    rounds = 100
    plot_svm("anova", rounds)
    plot_svm("AU_all", rounds)
    plot_svm("AU_c_only", rounds)
    borgs = plot_svm("AU_r_only", rounds)
    for k, v in borgs.items():
        print(k, v[0], v[1].values())


    # svm_for_all("Anova Subset", para_linear, 3, sel_features=idxs)

    # print("Selected Features")
    # svm_for_all("AU subset", sel_features=idxs, )
    # plt.show()
    # box_plot_AUs_diff()
    # svm_for_all()
    # X, y = prep_data_inter_faces_borg()
    # print(X.shape)
    # print(y)
    # X, y = prep_data_inter_faces(choose=False, c_class=0, remove_below_brog=0)
    # from sklearn.manifold import TSNE
    # X, y = prep_data_inter_faces()
    # X = StandardScaler().fit_transform(X)
    # X = X[:,17:]
    # y[y > 0] = 1
    # X_embedded = TSNE(n_components=2).fit_transform(X)
    # print(X_embedded.shape)
    # plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1])
    # plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1])
    # plt.show()
    # print(y[y == 0].shape)
    # print(y[y == 1].shape)
    # samples = np.random.choice(X[y == 1].shape[0], y[y==0].shape[0])
    # print(samples)
    # samples = np.random.choice(X[y == 1].shape[0], y[y==0].shape[0]) + y[y==0].shape[0]
    # X_c = np.concatenate((X[y==0], X[samples]), axis=0)
    # y_c = np.concatenate([np.zeros(y[y==0].shape), np.ones(y[y==0].shape)])
    # X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.33, random_state=2)
    # from sklearn.neighbors import KNeighborsClassifier
    # neigh = KNeighborsClassifier(n_neighbors=5)
    # neigh.fit(X_train, y_train)
    # print(neigh.score(X_test, y_test))
