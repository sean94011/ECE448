# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    W = np.zeros(train_set.shape[1])
    b = 0

    for i in range(max_iter):
        for index in range(train_labels.shape[0]):
            y_head = np.sign(np.sum(train_set[index][:] * W) + b)
            y_head = np.where(y_head >= 0, y_head, 0)
            W += (learning_rate*(train_labels[index]-y_head)*train_set[index][:])
            b += learning_rate*(train_labels[index]-y_head)


    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    predicted_label = np.zeros(dev_set.shape[0])
    for i in range(dev_set.shape[0]):
        predicted_label[i] = np.sign(np.sum(dev_set[i][:] * W)+b)

    predicted_label = list(np.where(predicted_label >= 0, predicted_label, 0))

    return predicted_label

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    import heapq as hq
    predicted_label = []
    
    for cur_index in range(dev_set.shape[0]):
        q = []
        hq.heapify(q)
        for train_index in range(train_labels.shape[0]):
            cur_dist = sum((train_set[train_index][:] - dev_set[cur_index][:])**2)
            hq.heappush(q,(cur_dist,train_labels[train_index]))
        cur_labels = hq.nsmallest(k,q)
        label_sum = 0
        for i in cur_labels:
            label_sum += i[1]
        if label_sum > (k/2):
            cur_label = 1
        else:
            cur_label = 0
        predicted_label.append(cur_label)

    return predicted_label
