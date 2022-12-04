'''
    This script provides a class to build, train, and evaluate ANN models.
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

USE_TRACT_INDEXED_DATASETS = 1

## Z-score normalization function
def scale_data(array,means,stds):
    return (array-means)/stds

class ANN_train:

    ## Specify the feature names
    if USE_TRACT_INDEXED_DATASETS == 1:
        df_column_names = ["Fiber Index", "Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10", "Activation"]
    else:
        df_column_names = ["Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10", "Activation"]

    ssd_node_names = ["ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10"]
    ec_node_names = ["ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10"]
    fsd_node_names = ["fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10"]

    def __init__(self, regression, num_ecs, num_fsds, num_ssds, num_layers, neurons, dropout, act_func, l_rate, epochs, batch_size, train_ds, val_ds):

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
        self.train_ds = train_ds
        self.val_ds = val_ds

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

        dataset = pd.read_csv(
            train_ds,
            names=self.df_column_names)

        if USE_TRACT_INDEXED_DATASETS == 1:
            self.dataset_features = dataset[["Fiber Index"] + ["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        else:
            self.dataset_features = dataset[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]

        self.dataset_labels = self.dataset_features.pop("Activation")

        ## Split the initial dataset into training and test cases (features = inputs, labels = outputs)
        features_train, features_test, self.labels_train, self.labels_test = train_test_split(self.dataset_features,self.dataset_labels,test_size=0.33,random_state=42)

        ## Normalize the training data manually
        self.norm_ec_mean = None
        self.norm_ec_var = None
        self.norm_fsd_mean = None
        self.norm_fsd_var = None
        self.norm_ssd_mean = None
        self.norm_ssd_var = None

        if len(self.ec_node_names) != 0:
            self.norm_ec_mean = self.dataset_features[self.ec_node_names].to_numpy().mean()
            self.norm_ec_var = self.dataset_features[self.ec_node_names].to_numpy().var()

        if len(self.fsd_node_names) != 0:
            self.norm_fsd_mean = self.dataset_features[self.fsd_node_names].to_numpy().mean()
            self.norm_fsd_var = self.dataset_features[self.fsd_node_names].to_numpy().var()

        if len(self.ssd_node_names) != 0:
            self.norm_ssd_mean = self.dataset_features[self.ssd_node_names].to_numpy().mean()
            self.norm_ssd_var = self.dataset_features[self.ssd_node_names].to_numpy().var()

        self.norm_pw_mean = self.dataset_features[["Pulse Width"]].to_numpy().mean()
        self.norm_pw_var = self.dataset_features[["Pulse Width"]].to_numpy().var()

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

        # self.features_train_std = scale_data(features_train, self.norm_mean, self.norm_vars **0.5)
        # self.features_test_std = scale_data(features_test, self.norm_mean, self.norm_vars **0.5)

        # if val_ds != None:
        #     dataset_val_1 = pd.read_csv(
        #         val_ds,
        #         names=["Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10", "Activation"])                       

        #     self.dataset_val_1_features = dataset_val_1[["Pulse Width"] + self.ec_node_names + self.fsd_node_names + self.ssd_node_names + ["Activation"]]
        #     self.dataset_val_1_labels = self.dataset_val_1_features.pop("Activation")    
        #     self.dataset_val_1_features_std = scale_data(self.dataset_val_1_features, self.norm_mean, self.norm_vars **0.5)

    def BuildandTrain(self):

        input_size = len(self.ec_node_names) + len(self.fsd_node_names) + len(self.ssd_node_names) + 1 # plus 1 for pulse width
        early_stop_callback = EarlyStopping(patience=1, restore_best_weights=True)

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
            # met = tf.keras.metrics.MeanAbsoluteError()
            # loss = tf.keras.losses.MeanAbsoluteError()
        else:
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            met = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            loss = tf.keras.losses.BinaryCrossentropy()
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.l_rate)

        ## Prepare the model for training by compiling
        self.model.compile(optimizer=opt,
                        loss=loss,
                        metrics=met)

        ## Train the model        
        self.model.fit(self.features_train_std, self.labels_train, batch_size=self.batch_size, epochs=self.epochs)
        # self.model.fit(features_train_std, labels_train, batch_size=batch_size, epochs=epochs, validation_data=(dataset_val_2_features_std, dataset_val_2_labels), callbacks=[early_stop_callback])

        ## Evaluate the model on the original dataset evaluation data
        _, test_acc = self.model.evaluate(self.features_test_std, self.labels_test, verbose=2)

        return test_acc      

    def BuildModel(self):
        input_size = len(self.ec_node_names) + len(self.fsd_node_names) + len(self.ssd_node_names) + 1 # plus 1 for pulse width
        early_stop_callback = EarlyStopping(patience=1, restore_best_weights=True)

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
            # met = tf.keras.metrics.MeanAbsoluteError()
            # loss = tf.keras.losses.MeanAbsoluteError()
        else:
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            met = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            loss = tf.keras.losses.BinaryCrossentropy()
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.l_rate)

        ## Prepare the model for training by compiling
        self.model.compile(optimizer=opt,
                        loss=loss,
                        metrics=met)

    def KFoldCrossValidate(self):

        features_cv = self.dataset_features #scale_data(self.dataset_features, self.norm_mean, self.norm_vars **0.5)
        labels_cv = self.dataset_labels

        print(type(features_cv))
        print(type(labels_cv))

        # Get number of distinct fiber trajectories
        dti_fiber_inds = []

        for row_ind in range(features_cv.shape[0]):
            if features_cv.iloc[row_ind][0] not in dti_fiber_inds:
                dti_fiber_inds.append(features_cv.iloc[row_ind][0])
            
        print("Number of distinct fiber tracts = " + str(len(dti_fiber_inds)))
            
        k = 3
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        val_acc_array = []

        for train_ind, val_ind in kf.split(dti_fiber_inds):
            features_train = features_cv[features_cv["Fiber Index"].isin(np.asarray(dti_fiber_inds)[train_ind])]
            features_val = features_cv[features_cv["Fiber Index"].isin(np.asarray(dti_fiber_inds)[val_ind])]
            labels_train = labels_cv[features_cv["Fiber Index"].isin(np.asarray(dti_fiber_inds)[train_ind])]
            labels_val = labels_cv[features_cv["Fiber Index"].isin(np.asarray(dti_fiber_inds)[val_ind])]

            # print(features_val)
            # input()

            features_train.pop("Fiber Index")
            features_val.pop("Fiber Index")

            print(features_train.shape)
            print(features_val.shape)
            print(labels_train.shape)
            print(labels_val.shape)

            #tf.keras.backend.clear_session()
            self.BuildModel()

            ## Train the model  
            # if self.regression == 1:
            #     features_train = scale_data(features_train, self.norm_mean, self.norm_vars **0.5)
            #     self.model.fit(features_train, labels_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, shuffle=True)
            # else:
            #     self.FitClassification(features_train, labels_train, self.batch_size, self.epochs, 50, 0.02)    

            features_train = scale_data(features_train, self.norm_mean, self.norm_vars **0.5)
            self.model.fit(features_train, labels_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, shuffle=True)
            # self.model.fit(features_train, labels_train, batch_size=batch_size, epochs=epochs, validation_data=(dataset_val_2_features_std, dataset_val_2_labels), callbacks=[early_stop_callback])

            features_val = scale_data(features_val, self.norm_mean, self.norm_vars **0.5)

            if self.regression == 1:
                _, val_acc = self.model.evaluate(features_val, labels_val, verbose=2)
            else:
                val_acc = self.EvaluateClassification(features_val, labels_val)

            print("Val Accuracy = " + str(val_acc))
            val_acc_array.append(val_acc)

        val_acc_mean = np.mean(np.asarray(val_acc_array))
        val_acc_std_dev = np.std(np.asarray(val_acc_array))
        
        return val_acc_mean, val_acc_std_dev

    def FullTrain(self):
        self.BuildModel()
        if USE_TRACT_INDEXED_DATASETS == 1:
            self.dataset_features.pop("Fiber Index")

        labels_cv = self.dataset_labels.iloc[:]

        ## Train the model  
        if self.regression == 1:
            features_cv = scale_data(self.dataset_features, self.norm_mean, self.norm_vars **0.5).iloc[:]
            self.model.fit(features_cv, self.dataset_labels, batch_size=self.batch_size, epochs=self.epochs, verbose=2, shuffle=True)
        else:
            self.FitClassification(self.dataset_features.iloc[:], labels_cv, self.batch_size, self.epochs, 50, 0.02)    


        ## Train the model        
        #self.model.fit(features_cv, labels_cv, batch_size=self.batch_size, epochs=self.epochs)
        # self.model.fit(features_train_std, labels_train, batch_size=batch_size, epochs=epochs, validation_data=(dataset_val_2_features_std, dataset_val_2_labels), callbacks=[early_stop_callback])

    def FitClassification(self, features_train, labels_train, batch_size, epochs, amps_per_vth, amp_std):
        data_features = []
        data_labels = []

        for ind in range(len(features_train)):
            features = features_train.iloc[ind]
            threshold = labels_train.iloc[ind]

            amps = np.random.normal(threshold, threshold * amp_std, amps_per_vth)
            for amp in amps:
                row = [features[0]] + [k*amp for k in features[1::]]
                data_features.append(row)

                if amp >= threshold:
                    data_labels.append(1)
                else:
                    data_labels.append(0)

        data_features_std = scale_data(data_features, self.norm_mean, self.norm_vars **0.5)

        self.model.fit(data_features_std, np.asarray(data_labels), batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)

    def Evaluate(self):
        if self.val_ds != None:
            ## Evaluate model on val dataset 
            if self.regression == 1:
                _, dataset_val_1_acc_t = self.model.evaluate(self.dataset_val_1_features_std, self.dataset_val_1_labels, verbose=2)
            else:
                dataset_val_1_acc_t = self.EvaluateClassification(self.dataset_val_1_features, self.dataset_val_1_labels)
        else:
            dataset_val_1_acc_t = 0

        return dataset_val_1_acc_t     

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
