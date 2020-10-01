# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
# from nltk.corpus import stopwords
# nltk.download(stopwords)
# stopwords_list = stopwords.words('english')

stopwords_list = ['and,', 'or,']

def unigram_freq_words(train_set,train_labels):
    freq_poswords = {}
    freq_negwords = {}

    for i in range(len(train_set)):
        if train_labels[i]:
            for w in train_set[i]:
                if w in stopwords_list:
                    continue
                if w not in freq_poswords:
                    freq_poswords[w] = 0
                freq_poswords[w] += 1
        else:
            for w in train_set[i]:
                if w in stopwords_list:
                    continue
                if w not in freq_negwords:
                    freq_negwords[w] = 0
                freq_negwords[w] += 1
    
    return freq_poswords, freq_negwords

def unigram_prob(freq_words, dev_set_text, smoothing_parameter, smo_denominator, log_prior):
    prob = log_prior

    for w in dev_set_text:
        if w not in freq_words:
            prob += np.log(smoothing_parameter/smo_denominator)
        else:
            prob += np.log((smoothing_parameter+freq_words[w])/smo_denominator)

    return prob

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    freq_poswords, freq_negwords = unigram_freq_words(train_set,train_labels)

    log_neg_prior = np.log(1-pos_prior)
    log_pos_prior = np.log(pos_prior)

    neg_smo_denominator = sum(freq_negwords.values()) + smoothing_parameter*(len(freq_negwords) + 1)
    pos_smo_denominator = sum(freq_poswords.values()) + smoothing_parameter*(len(freq_poswords) + 1)

    dev_labels = []
    
    for i in range(len(dev_set)):
        prob_neg = unigram_prob(freq_negwords, dev_set[i], smoothing_parameter, neg_smo_denominator, log_neg_prior)
        prob_pos = unigram_prob(freq_poswords, dev_set[i], smoothing_parameter, pos_smo_denominator, log_pos_prior)

        if prob_pos > prob_neg:
            dev_labels.append(True)
        else:
            dev_labels.append(False)

    return dev_labels

def bigram_freq_word_pairs(train_set,train_labels):
    freq_posword_pairs = {}
    freq_negword_pairs = {}

    for i in range(len(train_set)):
        if train_labels[i]:
            for idx in range(len(train_set[i])-1):
                if (train_set[i][idx] in stopwords_list) or (train_set[i][idx+1] in stopwords_list):
                    continue
                if (train_set[i][idx],train_set[i][idx+1]) not in freq_posword_pairs:
                    freq_posword_pairs[(train_set[i][idx],train_set[i][idx+1])] = 0
                freq_posword_pairs[(train_set[i][idx],train_set[i][idx+1])] += 1
        else:
            for idx in range(len(train_set[i])-1):
                if (train_set[i][idx] in stopwords_list) or (train_set[i][idx+1] in stopwords_list):
                    continue
                if (train_set[i][idx],train_set[i][idx+1]) not in freq_negword_pairs:
                    freq_negword_pairs[(train_set[i][idx],train_set[i][idx+1])] = 0
                freq_negword_pairs[(train_set[i][idx],train_set[i][idx+1])] += 1
    
    return freq_posword_pairs, freq_negword_pairs

def bigram_prob(freq_word_pairs, dev_set_text, smoothing_parameter, smo_denominator, log_prior):
    prob = log_prior

    for wp in range(len(dev_set_text)-1):
        if (dev_set_text[wp],dev_set_text[wp+1]) not in freq_word_pairs:
            prob += np.log(smoothing_parameter/smo_denominator)
        else:
            prob += np.log((smoothing_parameter+freq_word_pairs[(dev_set_text[wp],dev_set_text[wp+1])])/smo_denominator)

    return prob

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.5, bigram_smoothing_parameter=0.005, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model

    freq_poswords, freq_negwords = unigram_freq_words(train_set,train_labels)
    freq_posword_pairs, freq_negword_pairs = bigram_freq_word_pairs(train_set,train_labels)

    log_neg_prior = np.log(1-pos_prior)
    log_pos_prior = np.log(pos_prior)

    neg_smo_denominator = sum(freq_negwords.values()) + unigram_smoothing_parameter*(len(freq_negwords) + 1)
    pos_smo_denominator = sum(freq_poswords.values()) + unigram_smoothing_parameter*(len(freq_poswords) + 1)

    neg_smo_wp_denominator = sum(freq_negword_pairs.values()) + bigram_smoothing_parameter*(len(freq_negword_pairs) + 1)
    pos_smo_wp_denominator = sum(freq_posword_pairs.values()) + bigram_smoothing_parameter*(len(freq_posword_pairs) + 1)

    dev_labels = []
    
    for i in range(len(dev_set)):
        prob_neg = unigram_prob(freq_negwords, dev_set[i], unigram_smoothing_parameter, neg_smo_denominator, log_neg_prior)
        prob_pos = unigram_prob(freq_poswords, dev_set[i], unigram_smoothing_parameter, pos_smo_denominator, log_pos_prior)

        prob_neg_wp = bigram_prob(freq_negword_pairs, dev_set[i], bigram_smoothing_parameter, neg_smo_wp_denominator, log_neg_prior)
        prob_pos_wp = bigram_prob(freq_posword_pairs, dev_set[i], bigram_smoothing_parameter, pos_smo_wp_denominator, log_pos_prior)

        prob_neg_combined = (1-bigram_lambda)*prob_neg + (bigram_lambda*prob_neg_wp)
        prob_pos_combined = (1-bigram_lambda)*prob_pos + (bigram_lambda*prob_pos_wp)


        if prob_pos_combined > prob_neg_combined:
            dev_labels.append(True)
        else:
            dev_labels.append(False)

    return dev_labels