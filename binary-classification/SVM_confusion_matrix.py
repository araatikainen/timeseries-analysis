# -*- coding: utf-8 -*-
#https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py

import matplotlib.pyplot as plt
import seaborn
from sklearn import datasets, svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split

# import some data to play with Iris plants dataset
iris = datasets.load_iris()

# Take the first two features (sepal length and sepal width in cm).
# We could avoid this by using a two-dim dataset
X = iris.data[:, :2]

#target is a class information
#class 1: 'setosa'
#class 2: 'versicolor'
#class 3: 'virginica'

y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (
    svm.SVC(kernel="linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X_train, y_train) for clf in models)

# title for the plots
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(models, titles, sub.flatten()):
    y_pred = clf.predict(X_test)  # Predict on the test set

    # Compute confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Plot confusion matrix
    seaborn.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()