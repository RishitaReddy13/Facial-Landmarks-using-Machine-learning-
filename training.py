import os  # Importing the os module for operating system functionality
import numpy as np  # Importing NumPy for numerical computations
from sklearn.model_selection import KFold  # Importing KFold for cross-validation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score  # Importing metrics for model evaluation
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from scikit-learn ensemble module
from sklearn.svm import SVC  # Importing SVC (Support Vector Classifier) from scikit-learn SVM module
from sklearn.tree import DecisionTreeClassifier  # Importing DecisionTreeClassifier from scikit-learn tree module
from math import acos, pi  # Importing acos and pi from math module for trigonometric calculations
from numpy import cos, sin, dot, transpose  # Importing trigonometric functions and array operations from NumPy
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from mpl_toolkits.mplot3d import Axes3D  # Importing Axes3D for 3D plotting
from tqdm import tqdm  # Importing tqdm for progress visualization
import pandas as pd  # Importing pandas for data manipulation

class ExperimentRunner:
    def plot_and_save_3d_scatter(self, data, title, filename, save_path):
        """
        Plots a 3D scatter plot of data and saves it to a file.

        Parameters:
        - data (np.array): Array containing 3D coordinates.
        - title (str): Title of the plot.
        - filename (str): Name of the file to save the plot.
        - save_path (str): Directory path to save the plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2], c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(title)
        plt.savefig(os.path.join(save_path, filename))
        plt.close()
        
    def run_experiment(self, classifier, data, labels):
        """
        Runs an experiment using k-fold cross-validation.

        Parameters:
        - classifier: Machine learning classifier object.
        - data (np.array): Input data.
        - labels (np.array): Labels corresponding to the input data.

        Returns:
        - tuple: Tuple containing average confusion matrix, accuracy, precision, recall, actual labels, and predicted labels.
        """
        kf = KFold(n_splits=10, shuffle=True)
        confusion_matrices = []
        accuracies = []
        precisions = []
        recalls = []
        all_actual_labels = []
        all_predicted_labels = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            train_data_flat = train_data.reshape(train_data.shape[0], -1)
            classifier.fit(train_data_flat, train_labels)

            test_data_flat = test_data.reshape(test_data.shape[0], -1)
            predictions = classifier.predict(test_data_flat)
            all_actual_labels.extend(test_labels)
            all_predicted_labels.extend(predictions)

            confusion_matrices.append(confusion_matrix(test_labels, predictions))
            accuracies.append(accuracy_score(test_labels, predictions))
            precisions.append(precision_score(test_labels, predictions, average='weighted', zero_division=1))
            recalls.append(recall_score(test_labels, predictions, average='weighted', zero_division=1))

        max_shape = max(cm.shape for cm in confusion_matrices)
        padded_matrices = [np.pad(cm, ((0, max_shape[0] - cm.shape[0]), (0, max_shape[1] - cm.shape[1])), mode='constant') for cm in confusion_matrices]
        avg_confusion_matrix = np.mean(padded_matrices, axis=0)
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        return avg_confusion_matrix, avg_accuracy, avg_precision, avg_recall, all_actual_labels, all_predicted_labels
