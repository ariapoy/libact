#!/usr/bin/env python3
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import LogisticRegression
from libact.query_strategies import RandomSampling, UncertaintySampling
from libact.labelers import IdealLabeler

import sys
sys.path.append("./")
from libact_dev.query_strategies import RandomSampling as RS_dev
from libact_dev.query_strategies import UncertaintySampling as US_dev

import pdb
def run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size=5):
    E_in, E_out = [], []
    
    # for _ in range(quota):
    remain_quota = quota
    # number of query labels at each iter
    num_labels_iter = [] # trn_ds.len_labeled()

    # if there is remain quota, we repeated conduce active learning procedure
    while remain_quota > 0:
        # Standard usage of libact objects
        # batch query
        ask_ids = qs.make_query(n=batch_size)
        if len(ask_ids) != batch_size:
            print("Warning! Inconsistent of batch size configuration and real query.")
        # label and update labeled pool
        # TODO: update labeled pool parallelly
        for b in range(len(ask_ids)):
            ask_id = ask_ids[b]
            lb = lbr.label(trn_ds.data[ask_id][0])
            trn_ds.update(ask_id, lb)
        # update quota
        remain_quota = remain_quota - batch_size
        # update/retrain model
        model.train(trn_ds)
        # record results
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
        num_labels_iter.append(len(ask_ids))

    # time query = len(E_in)
    num_labels_iter = np.cumsum(num_labels_iter)
    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=1126)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 10      # number of samples that are initially labeled

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query
    batch_size = 5

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    # qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression(), n=batch_size)
    qs = US_dev(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size)

    # qs2 = RandomSampling(trn_ds2, n=batch_size)
    qs2 = RS_dev(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota, batch_size)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    assert len(E_in_1) == len(E_in_2)
    query_num = np.arange(1, len(E_in_1) + 1)
    plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
