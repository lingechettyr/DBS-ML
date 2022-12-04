import json
import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sortedcollections import SortedDict
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import pyplot as plt


## Comment the below out to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Z-score normalization function
def scale_data(array,means,stds):
    return (array-means)/stds

class Tree_train:
    ## Specify the feature names
    df_column_names = ["Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10", "Activation"]

    ssd_node_names = ["ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10"]
    ec_node_names = ["ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10"]
    fsd_node_names = ["fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10"]

    def __init__(self, regression, num_ecs, num_fsds, num_ssds, eta, max_depth, tree_method, objective, eval_metric):
        self.regression = regression
        self.num_ecs = num_ecs
        self.num_fsds = num_fsds
        self.num_ssds = num_ssds
        self.eta = eta
        self.max_depth = max_depth
        self.tree_method = tree_method
        self.objective = objective
        self.eval_metric = eval_metric

        for ind in range(int((11-num_ecs)/2)):
            self.ec_node_names.pop(0)
            self.ec_node_names.pop()

        for ind in range(int((11-num_ssds)/2)):
            self.ssd_node_names.pop(0)
            self.ssd_node_names.pop()

        for ind in range(int((11-num_fsds)/2)):
            self.fsd_node_names.pop(0)
            self.fsd_node_names.pop()

        if num_ecs == 0:
            self.ec_node_names.pop()
        if num_ssds == 0:
            self.ssd_node_names.pop()
        if num_fsds == 0:
            self.fsd_node_names.pop()

    def build(self):
        ## Create the self.model
        if self.regression == 1:
            self.objective = "reg:squarederror"
            self.eval_metric = "mape"
            self.model = XGBRegressor(eta=self.eta, max_depth=self.max_depth, tree_method=self.tree_method, objective=self.objective, eval_metric=self.eval_metric)
        else:
            self.model = XGBClassifier(eta=self.eta, max_depth=self.max_depth, tree_method=self.tree_method, objective=self.objective, eval_metric=self.eval_metric)
    
    def dataNorm(self, data, norm):
        dataset = pd.read_csv(data, names=self.df_column_names)

        dataset_features = dataset[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        dataset_labels = dataset_features.pop("Activation")

        ## Calculate the normalization parameters based on the inputted training dataset
        if norm == 1:
            self.norm_ec_mean = None
            self.norm_ec_var = None
            self.norm_fsd_mean = None
            self.norm_fsd_var = None
            self.norm_ssd_mean = None
            self.norm_ssd_var = None

            if len(self.ec_node_names) != 0:
                self.norm_ec_mean = dataset_features[self.ec_node_names].to_numpy().mean()
                self.norm_ec_var = dataset_features[self.ec_node_names].to_numpy().var()

            if len(self.fsd_node_names) != 0:
                self.norm_fsd_mean = dataset_features[self.fsd_node_names].to_numpy().mean()
                self.norm_fsd_var = dataset_features[self.fsd_node_names].to_numpy().var()

            if len(self.ssd_node_names) != 0:
                self.norm_ssd_mean = dataset_features[self.ssd_node_names].to_numpy().mean()
                self.norm_ssd_var = dataset_features[self.ssd_node_names].to_numpy().var()

            self.norm_pw_mean = dataset_features[["Pulse Width"]].to_numpy().mean()
            self.norm_pw_var = dataset_features[["Pulse Width"]].to_numpy().var()

            self.norm_mean = np.array([self.norm_pw_mean])
            self.norm_vars = np.array([self.norm_pw_var])

            for ind in range(len(self.ec_node_names)):
                self.norm_mean = np.append(self.norm_mean, self.norm_ec_mean)
                self.norm_vars = np.append(self.norm_vars, self.norm_ec_var)

            for ind in range(len(self.fsd_node_names)):
                self.norm_mean = np.append(self.norm_mean, self.norm_fsd_mean)
                self.norm_vars = np.append(self.norm_vars, self.norm_fsd_var)

            for ind in range(len(self.ssd_node_names)):
                self.norm_mean = np.append(self.norm_mean, self.norm_ssd_mean)
                self.norm_vars = np.append(self.norm_vars, self.norm_ssd_var)

            # Normalize the data
            dataset_features_std = scale_data(dataset_features, self.norm_mean, self.norm_vars **0.5)
            return dataset_features_std, dataset_labels
        return dataset_features, dataset_labels
    
    def splitData(self, data):
        XFull, YFull = self.dataNorm(data, 0)
        XTrain, XTest, YTrain, YTest = train_test_split(XFull, YFull, test_size=0.3)
        XTest_og = XTest
        XTest_arr = XTest.to_numpy()
        pulse_widths = list(set(XTest_arr[:, 0]))
        return XTrain, XTest_og, YTrain, YTest, pulse_widths

    def trainEval(self, data):
        XTrain, XTest, YTrain, YTest, pulse_widths = self.splitData(data)

        self.model.fit(XTrain, YTrain)
        # Training accuracy
        train_acc = None
        YTrainPred = self.model.predict(XTrain)
        if self.regression == 1:
            train_acc = mean_absolute_percentage_error(YTrain, YTrainPred)
        else:
            train_acc = accuracy_score(YTrain, YTrainPred)
        
        # Validation accuracy
        val_acc = None
        YTestPred = self.model.predict(XTest)
        if self.regression == 1:
            val_acc = mean_absolute_percentage_error(YTest, YTestPred)
        else:
            val_acc = accuracy_score(YTest, YTestPred)

        return train_acc, val_acc

    def featureImportance(self):
        # feature importance of xgboost model plot
        plot_importance(self.model)
        plt.show()

    def graphTree(self):
        # plot single tree
        plot_tree(self.model, num_trees=0, rankdir='LR')
        plt.show()

    def crossValBoxPlot(self, data):
        # box plot of each unique pulse width and its accuracy for 10-fold cross validation 
        k_folds = 10
        acc = SortedDict()
        for k in range(k_folds):
            XTrain, XTest, YTrain, YTest, pulse_widths = self.splitData(data)
            self.model.fit(XTrain, YTrain)
            YTestPred = self.model.predict(XTest)
            XTest, YTest = XTest.to_numpy(), YTest.to_numpy()

            for width in pulse_widths:
                if width not in acc:
                    acc[width] = []
                index = np.where(XTest[:, 0] == width)
                if self.regression == 1:
                    acc[width].append(mean_absolute_percentage_error(YTest[index], YTestPred[index]))
                else:
                    acc[width].append(accuracy_score(YTest[index], YTestPred[index]))
        
        labels, data = acc.keys(), acc.values()
        plt.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.xlabel("Pulse Width")
        plt.ylabel("Accuracy Skew")
        plt.show()

    def SaveModel(self, output_filename):
        ## Save xgboost model
        self.model.save_model(output_filename + ".json")