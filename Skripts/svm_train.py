""" 
Helper Module to train the SVM, balance the Dataset and creaating the resulting Plots.
"""


import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from stats import prep_data_inter_faces, labels
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import environment as paths

def balance_classes(X,y):
    """ Balance the Init vs exe classes.
    The exe class contains way more examples than the init class.
    This function will split the exe classs into one class which ist approximately
    the same size as the init class.

    Parameters:
    -----------
    X: Array with datapoints.

    y: 1D array. The labels 1-5 for the exe classes are expected!

    Return:
    -------
    Balanced X,y.
    """

    # Balance clases:
    N_0 = y[y == 0].shape[0]
    N_1 = y[y > 0].shape[0]

    diff = N_1 - N_0
    print(diff)


    # Handel cases where init class dominates:
    if abs(diff) <= 10:
        return X, y
    elif diff > 0:
        n = (N_0 // 5) + 1
        # seed = 1
        shift = N_0
        samples = []
        for i in range(1,6):
            # np.random.seed(seed)
            samples.append(np.random.choice(X[y == i].shape[0], n) + shift)
            shift += X[y==i].shape[0]
        samples = np.concatenate(samples)
            # Balanced X and y
        X_c = np.concatenate((X[y==0], X[samples]), axis=0)
        y_c = np.concatenate([np.zeros(y[y==0].shape), np.ones(samples.shape)])
        return X_c, y_c
    else:
        # balance init against ex:
        n = N_1
        # seed = 1
        # shift = N_0
        samples = []
        # for i in range(1,6):
            # np.random.seed(seed)
        samples = np.random.choice(N_0, n)

        # Balanced X and y
        X_c = np.concatenate((X[samples], X[y > 0]), axis=0)
        y_c = np.concatenate([np.zeros(y[samples].shape), np.ones(N_1)])
        return X_c, y_c


def cross_validate_roc(ax, X, y, params, title, rounds=3):
    """Training and testing of the classifyer.
    The SVC will be trained with a crossvalidation and a random selected subset for each iteration.
    The mean and standard deviation of all results will be plotted in a ROC curve.
    
    Parameters:
    -----------
    ax: axis for plotting
    
    X: array with the samplepoints
    
    y: array, labels for X, both needs to have the same size
    
    params: dict, Parameters for the Gridsearch which will tell what parameters to train
    
    title: String, title of the plot
    
    round: Default 3, tells how many times the SVC should be trained. Each training will select a new random subset out of X
            according to how the classes are ballanced.
            
    Return:
    -------
    ax: Plot with mean and std.
    
    from sklearn.metrics import recall_score
    from sklearn import preprocessing
    from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score, make_scorer, accuracy_score
    import scipy
    # Normalize for svm
    X = StandardScaler().fit_transform(X)


    rd = 0

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    while rd < rounds:
        X_c, y_c = balance_classes(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.33)

        # Init svm and cv
        svc = svm.SVC(probability=True)
        # scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        scoring = 'roc_auc'
        clf = GridSearchCV(svc, params, cv=5, n_jobs=-1, verbose=1, scoring=scoring)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        # print(precision_recall_curve(y_test, clf.predict(X_test)))
        print("Train Score: {}\tTest Score: {} Precision: {}, Recall: {}\n".format(
            clf.score(X_train, y_train),
            clf.score(X_test, y_test),
            average_precision_score(y_test, y_pred), recall_score(y_test, y_pred, average='weighted')))
        print(classification_report(y_test, y_pred))
        print(clf.best_params_)
        # print(clf.gamma)
        probas_ = clf.predict_proba(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3)
        rd += 1


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr,
             label=r'%s, AUC = %0.2f $\pm$ %0.2f' % (title, mean_auc, std_auc),
             lw=2, alpha=.8)
    # ax.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    return ax
