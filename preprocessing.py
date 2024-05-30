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


class LandmarkProcessor:
    def read_bnd_file(self, file_path):
        """
        Reads landmark data from a .bnd file.

        Parameters:
        - file_path (str): Path to the .bnd file.

        Returns:
        - np.array: Array containing landmark coordinates.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
            landmarks = []
            for line in lines:
                coords = line.split()[1:]
                landmarks.append([float(coord) for coord in coords])
            return np.array(landmarks)

    def load_data(self, data_directory):
        """
        Loads landmark data from .bnd files in the specified directory.

        Parameters:
        - data_directory (str): Directory containing .bnd files.

        Returns:
        - tuple: Tuple containing data (landmark coordinates) and labels (expressions).
        """
        data = []
        labels = []
        for root, dirs, files in os.walk(data_directory):
            if os.path.basename(root) != ".ipynb_checkpoints":
                for file in files:
                    if file.endswith('.bnd'):
                        file_path = os.path.join(root, file)
                        expression = os.path.basename(root)
                        subject_id = os.path.basename(os.path.dirname(root))
                        label = expression
                        landmarks = self.read_bnd_file(file_path)
                        data.append(landmarks)
                        labels.append(label)
        assert len(data) == len(labels), "Data and labels lengths do not match"
        return np.array(data), np.array(labels)

    def calculate_average_landmarks(self, data):
        """
        Calculates the average landmarks from a set of landmark data.

        Parameters:
        - data (np.array): Array containing landmark coordinates.

        Returns:
        - np.array: Array containing the average landmark coordinates.
        """
        return np.mean(data, axis=0)

    def translate_to_origin(self, data):
        """
        Translates landmark data to the origin.

        Parameters:
        - data (np.array): Array containing landmark coordinates.

        Returns:
        - np.array: Array containing translated landmark coordinates.
        """
        average_landmarks = self.calculate_average_landmarks(data)
        return data - average_landmarks

    def rotate_landmarks(self, data, axis):
        """
        Rotates landmark data around the specified axis.

        Parameters:
        - data (np.array): Array containing landmark coordinates.
        - axis (str): Axis to rotate around ('x', 'y', or 'z').

        Returns:
        - np.array: Array containing rotated landmark coordinates.
        """
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cos(pi), sin(pi)],
                [0, -sin(pi), cos(pi)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [cos(pi), 0, -sin(pi)],
                [0, 1, 0],
                [sin(pi), 0, cos(pi)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [cos(pi), sin(pi), 0],
                [-sin(pi), cos(pi), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'")
        return np.dot(data, rotation_matrix)
