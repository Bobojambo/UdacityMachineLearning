#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""
import sys
from sklearn import svm
from sklearn.metrics import accuracy_score
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

C_list = [10000]

for c_value in C_list:
    clf = svm.SVC(kernel='rbf', C=c_value)

    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t0 = time()
    labels_pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"

    accuracy = accuracy_score(labels_test, labels_pred)
    print c_value, accuracy
    print "end"

Chris_preds = 0
for prediction in labels_pred:
    if prediction == 1:
        Chris_preds += 1

print "Chris predictions: ", Chris_preds
