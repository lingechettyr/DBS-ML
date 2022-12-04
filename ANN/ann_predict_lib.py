'''
    This file provides a class for loading and making predictions
    with existing ANN models made in TensorFlow.
'''

import numpy as np
import json
import tensorflow as tf
import math

# Comment the below out to use GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ANN:

    def __init__(self, model_path):
        with open(model_path + '/norm.json') as f:
            norm_dict = json.load(f)

        norm_pw_mean = norm_dict["norm_pw_mean"]
        norm_pw_var = norm_dict["norm_pw_var"]
        norm_ec_mean = norm_dict["norm_ec_mean"]
        norm_ec_var = norm_dict["norm_ec_var"]
        norm_fsd_mean = norm_dict["norm_fsd_mean"]
        norm_fsd_var = norm_dict["norm_fsd_var"]
        norm_ssd_mean = norm_dict["norm_ssd_mean"]
        norm_ssd_var = norm_dict["norm_ssd_var"]

        with open(model_path + '/hparams.json') as f:
            hparam_dict = json.load(f)

        self.hparam_dict = hparam_dict
        self.num_ecs = hparam_dict["num_ecs"]
        self.num_fsds = hparam_dict["num_fsds"]
        self.num_ssds = hparam_dict["num_ssds"]

        self.norm_means = np.array([norm_pw_mean])
        self.norm_vars = np.array([norm_pw_var])

        for ind in range(self.num_ecs):
            self.norm_means = np.append(self.norm_means, norm_ec_mean)
            self.norm_vars = np.append(self.norm_vars, norm_ec_var)

        for ind in range(self.num_fsds):
            self.norm_means = np.append(self.norm_means, norm_fsd_mean)
            self.norm_vars = np.append(self.norm_vars, norm_fsd_var)

        for ind in range(self.num_ssds):
            self.norm_means = np.append(self.norm_means, norm_ssd_mean)
            self.norm_vars = np.append(self.norm_vars, norm_ssd_var)

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, input_tensor):

        input_norm = self.scale_data(input_tensor, self.norm_means, self.norm_vars**0.5)
        tensor = np.asarray(input_norm).reshape(-1,len(input_tensor))

        prediction = self.model.predict(tensor)
        if prediction > 0.5:
            return 1
        else:
            return 0

    def predict_threshold(self, input_tensor, upper_bound):
        precision = 0.001 # specific the desired threshold precision
        lower_bound = 0

        # Do a binary search
        iterations = math.ceil(math.log((upper_bound-lower_bound)/precision,2)) #calculate the number of binary search iterations to achieve a certain precision
        for i in range(iterations):
            multiplier = ((upper_bound + lower_bound) / 2)
            input_scaled = np.insert(np.multiply(input_tensor[1:], multiplier), 0, input_tensor[0])

            prediction = self.predict(input_scaled)
            #print("Binary search: iteration = " + str(i) + ", multiplier = " + str(multiplier) + ", spikes = " + str(prediction))
            if prediction == 1:
                upper_bound = multiplier
            else:
                lower_bound = multiplier
        
        roundTo = 2 + int(abs(math.log(precision, 10)))
        threshold = round(multiplier, roundTo)

        return threshold

    def batch_predict(self, input_tensor):

        input_norm = self.scale_data(input_tensor, self.norm_means, self.norm_vars**0.5)
        prediction = self.model.predict(input_norm)

        predictions_rnd = []
        for i in prediction:
            if i > 0.5:
                predictions_rnd.append(1)
            else:
                predictions_rnd.append(0)

        return predictions_rnd

    def batch_predict_threshold(self, input_tensor, upper_bound):
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

        return thresholds

    ## All predict functions above this are for classification ANNs. The below
    # function is instead for regression ANNs, where the output of the ANN is 
    # directly used. This is in contrast to classification ANNs, where the ANN 
    # output is compared to 0.5 and rounded down to 0 or up to 1. 
    def batch_predict_threshold_reg(self, input_tensor):
        input_norm = self.scale_data(input_tensor, self.norm_means, self.norm_vars**0.5)
        thresholds = self.model.predict(input_norm)

        return thresholds

    def scale_data(self,array,means,stds):
        return (array-means)/stds

    def get_input_sizes(self):
        return self.num_ecs, self.num_fsds, self.num_ssds

    def get_hparam_dict(self):
        return self.hparam_dict