import os
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from math import acos, pi
from numpy import cos, sin, dot, transpose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from preprocessing import LandmarkProcessor  # Importing LandmarkProcessor from preprocessing module
from training import ExperimentRunner  # Importing ExperimentRunner from training module
from tqdm import tqdm  # Importing tqdm for progress visualization
import pandas as pd  # Importing pandas for data manipulation

class Analysis:
    def __init__(self):
        self.processor = LandmarkProcessor()  # Initializing LandmarkProcessor
        self.runner = ExperimentRunner()  # Initializing ExperimentRunner

    def analyze_with_progress(self, classifiers, data_types, data, labels,save_path):
        results = {}  # Dictionary to store analysis results
         
        # save_path = r"C:\Users\ADMIN\Facial_landmarks_project\data\final-results-with_graphs_and_csvs_final"  # Directory path to save results
        os.makedirs(save_path, exist_ok=True)  # Creating directory if not exists

        for classifier_name, classifier in tqdm(classifiers.items(), desc='Classifiers'):  # Iterating over classifiers with progress visualization
            for data_type_code, data_type in tqdm(data_types.items(), desc='Data Types', leave=False):  # Iterating over data types with progress visualization
                processed_data = None  # Initializing processed data variable
                if data_type_code == 'o':
                    processed_data = data  # Using original data
                elif data_type_code == 't':
                    processed_data = self.processor.translate_to_origin(data)  # Translating data to origin
                elif data_type_code in ['x', 'y', 'z']:
                    processed_data = self.processor.rotate_landmarks(data, data_type_code)  # Rotating landmarks based on data type
                self.runner.plot_and_save_3d_scatter(processed_data[0], f"{classifier_name} - {data_type}", f"{classifier_name}_{data_type_code}_scatter.png", save_path)  # Plotting and saving 3D scatter plot
                avg_confusion_matrix, avg_accuracy, avg_precision, avg_recall, all_actual_labels, all_predicted_labels = self.runner.run_experiment(classifier, processed_data, labels)  # Running experiment and obtaining performance metrics

                output_file = f"results_{classifier_name}_{data_type_code}.txt"  # Output file name
                output_path = os.path.join(save_path, output_file)  # Output file path
                with open(output_path, 'w') as f:  # Writing results to a text file
                    f.write("Average Confusion Matrix:\n")
                    np.savetxt(f, avg_confusion_matrix, fmt='%d')  # Writing confusion matrix to file
                    f.write("\nAverage Accuracy: {:.2f}\n".format(avg_accuracy))  # Writing average accuracy to file
                    f.write("Average Precision: {:.2f}\n".format(avg_precision))  # Writing average precision to file
                    f.write("Average Recall: {:.2f}\n".format(avg_recall))  # Writing average recall to file
                
                df = pd.DataFrame({'Actual Label': all_actual_labels, 'Predicted Label': all_predicted_labels})  # Creating DataFrame from actual and predicted labels
                csv_output_path = os.path.join(save_path, f'actual_predicted_{classifier_name}_{data_type_code}.csv')  # CSV output file path
                df.to_csv(csv_output_path, index=False)  # Writing DataFrame to CSV file

                results[(classifier_name, data_type)] = {  # Storing results in the dictionary
                    'avg_confusion_matrix': avg_confusion_matrix,
                    'avg_accuracy': avg_accuracy,
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall
                }

        return results  # Returning analysis results
    
def main():
    if len(sys.argv) != 4:
        print("Usage: python Project1.py <classifier to run> <data type to use> <data directory>")
        return

    classifier_input = sys.argv[1].upper()
    data_type_input = sys.argv[2].lower()
    data_directory = sys.argv[3]
    save_path = "./results"  # Default save path, change as needed

    classifiers = {  # Dictionary containing classifiers
        'RF': RandomForestClassifier(),
        'SVM': SVC(),
        'TREE': DecisionTreeClassifier()
    }
    data_types = {  # Dictionary containing data types
        'o': 'Original',
        't': 'Translated',
        'x': 'Rotated X',
        'y': 'Rotated Y',
        'z': 'Rotated Z'
    }

    # Handling relative data directory path
    data_directory = os.path.abspath(data_directory)
    processor = LandmarkProcessor()  # Initializing LandmarkProcessor
    data, labels = processor.load_data(data_directory)  # Loading data and labels

    analysis = Analysis()  # Initializing Analysis

    if classifier_input in classifiers and data_type_input in data_types:
        selected_classifier = {classifier_input: classifiers[classifier_input]}
        selected_data_type = {data_type_input: data_types[data_type_input]}
        results = analysis.analyze_with_progress(selected_classifier, selected_data_type, data, labels, save_path)  # Performing analysis
    else:
        print("Invalid input for classifier or data type.")
        results = None

    return results  # Returning analysis results

if __name__ == "__main__":
    main()
