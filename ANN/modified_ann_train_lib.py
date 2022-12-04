'''
    This script provides a class to build, train, and evaluate ANN models.
    Several of the functions are not used anymore
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import os
import math

## Comment the below out to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Z-score normalization function
def scale_data(array,means,stds):
    return (array-means)/stds

class ANN_train:

    ## Specify the feature names
    df_column_names = ["Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10", "Activation"]

    ssd_node_names = ["ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10"]
    ec_node_names = ["ec_2"]
    fsd_node_names = ["fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10"]

    def __init__(self, regression, num_ecs, num_fsds, num_ssds, num_layers, neurons, dropout, act_func, l_rate, epochs, batch_size):

        self.regression = regression
        self.num_ecs = num_ecs
        self.num_fsds = num_fsds
        self.num_ssds = num_ssds
        self.num_layers = num_layers
        self.neurons = neurons
        self.dropout = dropout
        self.act_func = act_func
        self.l_rate = l_rate
        self.epochs = epochs
        self.batch_size = batch_size

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

    def Build(self):

        input_size = len(self.ec_node_names) + len(self.fsd_node_names) + len(self.ssd_node_names) + 1 # plus 1 for pulse width

        ## Build the network using passed in hparameters
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))
        for ind in range(self.num_layers):
            self.model.add(tf.keras.layers.Dense(self.neurons, activation=self.act_func))
            self.model.add(tf.keras.layers.Dropout(self.dropout))

        if self.regression == 1:
            self.model.add(tf.keras.layers.Dense(1, activation='relu'))
            met = tf.keras.metrics.MeanAbsolutePercentageError()
            loss = tf.keras.losses.MeanAbsolutePercentageError()
        else:
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            met = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            loss = tf.keras.losses.BinaryCrossentropy()
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.l_rate)

        ## Prepare the model for training by compiling
        self.model.compile(optimizer=opt,
                        loss=loss,
                        metrics=met)    

    def TrainEarlyStoppage(self, train_ds, val_ds):
        # Load the training dataset csv
        dataset = pd.read_csv(
            train_ds,
            names=self.df_column_names)

        self.train_ds = train_ds

        # Separate the dataset into features and labels
        dataset_features = dataset[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        dataset_labels = dataset_features.pop("Activation")

        ## Calculate the normalization parameters based on the inputted training dataset
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

        # Normalize the training data
        dataset_features_std = scale_data(dataset_features, self.norm_mean, self.norm_vars **0.5)

        # Load the validation dataset csv
        dataset_val = pd.read_csv(
            val_ds,
            names=self.df_column_names)

        self.val_ds = val_ds

        # Separate the dataset into features and labels
        dataset_val_features = dataset_val[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        dataset_val_labels = dataset_val_features.pop("Activation")
        dataset_val_features_std = scale_data(dataset_val_features, self.norm_mean, self.norm_vars **0.5)

        ## Train the model    
        early_stop_callback = EarlyStopping(patience=10, restore_best_weights=True)    
        self.model.fit(dataset_features_std, dataset_labels, batch_size=self.batch_size, epochs=self.epochs, validation_data=(dataset_val_features_std, dataset_val_labels), callbacks=[early_stop_callback], verbose=2)

        # Evaluate the model on the entire training dataset
        _, dataset_acc = self.model.evaluate(dataset_features_std, dataset_labels, verbose=2)

        # Evaluate the model on the validation dataset
        _, dataset_val_acc = self.model.evaluate(dataset_val_features_std, dataset_val_labels, verbose=2)

        return dataset_acc, dataset_val_acc
 
    def SaveModel(self, output_filename):
        ## Save TF model
        self.model.save(output_filename)

        ## Save normalization parameters
        norm_dict = {}
        norm_dict["norm_ssd_mean"] = self.norm_ssd_mean
        norm_dict["norm_ssd_var"] = self.norm_ssd_var
        norm_dict["norm_ec_mean"] = self.norm_ec_mean
        norm_dict["norm_ec_var"] = self.norm_ec_var
        norm_dict["norm_pw_mean"] = self.norm_pw_mean
        norm_dict["norm_pw_var"] = self.norm_pw_var
        norm_dict["norm_fsd_mean"] = self.norm_fsd_mean
        norm_dict["norm_fsd_var"] = self.norm_fsd_var

        with open(os.path.join(output_filename,'norm.json'), 'w') as outfile:
            json.dump(norm_dict, outfile, indent=4) 

        ## Save hyperparameters
        hparam_dict = {}
        hparam_dict["regression"] = self.regression
        hparam_dict["num_ecs"] = self.num_ecs
        hparam_dict["num_fsds"] = self.num_fsds
        hparam_dict["num_ssds"] = self.num_ssds
        hparam_dict["num_layers"] = self.num_layers
        hparam_dict["neurons"] = self.neurons
        hparam_dict["dropout"] = self.dropout
        hparam_dict["act_func"] = self.act_func
        hparam_dict["l_rate"] = self.l_rate
        hparam_dict["epochs"] = self.epochs
        hparam_dict["batch_size"] = self.batch_size
        hparam_dict["train_ds"] = self.train_ds
        hparam_dict["val_ds"] = self.val_ds

        with open(os.path.join(output_filename, 'hparams.json'), 'w') as outfile:
            json.dump(hparam_dict, outfile, indent=4) 

    ## The below functions aren't used anymore. TrainEarlyStoppage was used to train/validate
    # the ANNs that are presented in the journal article. This function uses a more standard
    # approach to ANN validation (i.e., early stopping) than the below functions. I am however 
    # leaving these below functions here in case they can be of use in the future.

    def Train(self, train_ds):
        # Load the training dataset csv
        dataset = pd.read_csv(
            train_ds,
            names=self.df_column_names)

        self.train_ds = train_ds

        # Separate the dataset into features and labels
        dataset_features = dataset[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        dataset_labels = dataset_features.pop("Activation")

        ## Calculate the normalization parameters based on the inputted training dataset
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

        # Normalize the training data
        dataset_features_std = scale_data(dataset_features, self.norm_mean, self.norm_vars **0.5)

        ## Train the model        
        self.model.fit(dataset_features_std, dataset_labels, batch_size=self.batch_size, epochs=self.epochs, verbose=2)

        # Evaluate the model on the entire training dataset
        _, dataset_acc = self.model.evaluate(dataset_features_std, dataset_labels, verbose=2)

        return dataset_acc

    def Evaluate(self, val_ds):
        # Load the validation dataset csv
        dataset = pd.read_csv(
            val_ds,
            names=self.df_column_names)

        self.val_ds = val_ds

        # Separate the dataset into features and labels
        dataset_features = dataset[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        dataset_labels = dataset_features.pop("Activation")

        ## Evaluate model on validation data
        if self.regression == 1:
            # Normalize the validation data
            dataset_features_std = scale_data(dataset_features, self.norm_mean, self.norm_vars **0.5)
            _, dataset_acc = self.model.evaluate(dataset_features_std, dataset_labels, verbose=2)
        else:
            # Use a binary search to get the thresholds as predicted by the model,
            # which can then be used to compute the MAPE of the thresholds. The data
            # will be normalized in this function so don't normalize here.
            dataset_acc = self.EvaluateClassification(dataset_features, dataset_labels)

        return dataset_acc     

    def EvaluateClassification(self, input_tensor, true_labels):
        
        input_tensor = np.asarray(input_tensor)
        true_labels = np.asarray(true_labels)

        upper_bound = 50
        precision = 0.001 # specific the desired threshold precision
        lower_bounds = [0 for k in range(len(input_tensor))]
        upper_bounds = [upper_bound for k in range(len(input_tensor))]
        multipliers = [(upper_bound + 0 / 2) for k in range(len(input_tensor))]

        # Do a binary search
        iterations = math.ceil(math.log((upper_bound-0)/precision,2)) #calculate the number of binary search iterations to achieve a certain precision
        for i in range(iterations):
            input_tensor_temp = []
            for j in range(len(input_tensor)):
                multipliers[j] = ((upper_bounds[j] + lower_bounds[j]) / 2)
                input_tensor_temp.append(np.insert(np.multiply(input_tensor[j][1:], multipliers[j]), 0, input_tensor[j][0]))

            prediction = self.batch_predict(input_tensor_temp)

            for j in range(len(prediction)):            
                if prediction[j] == 1:
                    upper_bounds[j] = multipliers[j]
                else:
                    lower_bounds[j] = multipliers[j]

        thresholds = []
        for i in range(len(input_tensor)):
            roundTo = 2 + int(abs(math.log(precision, 10)))
            thresholds.append(round(multipliers[i], roundTo))

        MAPE = np.multiply(np.mean(np.divide(np.abs(np.subtract(np.asarray(thresholds), true_labels)),true_labels)), 100)
                
        return MAPE

    def batch_predict(self, input_tensor):

        input_norm = scale_data(input_tensor, self.norm_mean, self.norm_vars**0.5)
        prediction = self.model.predict(input_norm)

        predictions_rnd = []
        for i in prediction:
            if i > 0.5:
                predictions_rnd.append(1)
            else:
                predictions_rnd.append(0)

        return predictions_rnd
