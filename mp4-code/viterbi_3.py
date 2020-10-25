"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import numpy as np
import math

def make_freq_tags(train):
    occurence = {}
    tag_word_table = {}
    for sentence in train:
        for word in sentence:
            if word[1] not in occurence:
                occurence[word[1]] = 0
                tag_word_table[word[1]] = {}
            occurence[word[1]] +=1
            if word[0] not in tag_word_table[word[1]]:
                tag_word_table[word[1]][word[0]] = 0
            tag_word_table[word[1]][word[0]] += 1
    unique_tags = np.array(list(occurence.keys()))
    V = unique_tags.shape[0]
    return occurence, unique_tags, V, tag_word_table

def make_freq_transition_pairs(train, unique_tags):
    occurence = {}
    for sentence in train:
        for i in range(len(sentence)-1):
            cur_pair = (sentence[i][1],sentence[i+1][1])
            if cur_pair not in occurence:
                occurence[cur_pair] = 0
            occurence[cur_pair] += 1
    for q in unique_tags:
        for k in unique_tags:
            if (q,k) not in occurence:
                occurence[(q,k)] = 0
    return occurence

def make_freq_word_tag_pairs(train):
    occurence = {}
    for sentence in train:
        for word in sentence:
            if word not in occurence:
                occurence[word] = 0
            occurence[word] += 1
    return occurence

def make_initial_prob(freq_tags, unique_tags,initial_sp, train,V):
    smoothed_prob = {}
    for tag in unique_tags:
        if tag == 'START':
            smoothed_prob[tag] = math.log((freq_tags[tag]+initial_sp)/(len(train)+initial_sp*(V)))
        else:
            smoothed_prob[tag] = math.log((initial_sp)/(len(train)+initial_sp*(V)))
    return smoothed_prob

def make_transition_prob(freq_tag_pairs,freq_tags,V,transition_sp):
    smoothed_prob = {}
    for tag_pair in freq_tag_pairs:
        smoothed_prob[tag_pair] = math.log((transition_sp + freq_tag_pairs[tag_pair]) / (freq_tags[tag_pair[0]] + transition_sp*(V)))
    return smoothed_prob


def make_emission_prob(freq_word_tag_pairs,V,freq_tags,emission_sp,hapax_tag_prob,tag_word_table):
    smoothed_prob = {}
    for w_t_pair in freq_word_tag_pairs:
        smoothed_prob[w_t_pair] = math.log((emission_sp*hapax_tag_prob[w_t_pair[1]] + freq_word_tag_pairs[w_t_pair]) / (freq_tags[w_t_pair[1]] + emission_sp*hapax_tag_prob[w_t_pair[1]]*(len(tag_word_table[w_t_pair[1]]))))
    return smoothed_prob




def decode(test,initial_prob,transition_prob,emission_prob,emission_sp,unique_tags,V,num_additional_class,n,hapax_tag_prob,all_suffix, tag_suffix_table,freq_none_class):
    tagged_sentences = []
    for sentence in test:
        cur_sentence = []
        cur_trellis = np.zeros((len(sentence),V,2))
        for word_index in range(len(sentence)):
            for cur_tag_index in range(V):
                cur_tag = unique_tags[cur_tag_index]
                cur_word = sentence[word_index]
 
                if (cur_word,cur_tag) in emission_prob:
                    p_e = emission_prob[(cur_word,cur_tag)]
                else:
                    suffix_prob = float('-inf')
                    for suffix in all_suffix:
                        if len(suffix) != 2:
                            continue
                        if cur_word[-1*all_suffix[suffix]:] == suffix:
                            suffix_prob = max(tag_suffix_table['pattern_'+suffix][cur_tag],suffix_prob)
                    
                    if suffix_prob == float('-inf'):
                        hapax_prob = freq_none_class[cur_tag]
                    else:
                        hapax_prob = suffix_prob
                
                    p_e = math.log((emission_sp*hapax_prob) / (n + emission_sp*hapax_prob*(V+num_additional_class)))

                if sentence[word_index] == 'START':
                    cur_trellis[word_index][cur_tag_index][0] = initial_prob[cur_tag] + p_e
                    cur_trellis[word_index][cur_tag_index][1] = None
                else:
                    previous_row = cur_trellis[word_index-1,:,0]

                    each_v = np.zeros(previous_row.shape)
                    for prev_tag_index in range(V):
                        each_v[prev_tag_index] = previous_row[prev_tag_index] + transition_prob[(unique_tags[prev_tag_index],cur_tag)] + p_e

                    cur_trellis[word_index][cur_tag_index][0] = np.amax(each_v)
                    cur_trellis[word_index][cur_tag_index][1] = int(np.argmax(each_v))

        cur_index = np.argmax(cur_trellis[-1,:,0])
        for i in range(cur_trellis.shape[0]-1,-1,-1):
            cur_index = int(cur_index)
            cur_sentence.append((sentence[i],unique_tags[cur_index]))
            cur_index = cur_trellis[i][cur_index][1]
            if cur_index == None:
                break
        cur_sentence.reverse()

        tagged_sentences.append(cur_sentence)
    return tagged_sentences

def find_suffix_hapax(hapax_word):
    cut_off = 100
    freq_suffix = {}
    for word in hapax_word:
        if word[-2:] not in freq_suffix:
            freq_suffix[word[-2:]] = 0
        freq_suffix[word[-2:]] += 1
            
    suffix_class = []
    all_suffix = {}
    for suffix in freq_suffix:
        if freq_suffix[suffix] >= cut_off:
            suffix_class.append('pattern_' + suffix)
    suffix_class.append('pattern_'+'s')
    all_suffix = make_all_suffix(suffix_class)
    return suffix_class, all_suffix

def make_hapax_prob(train,freq_word_tag_pairs,unique_tags,hapax_sp,V,tag_word_table):
    hapax_word = {}
    check_freq = {}
    for tag in tag_word_table:
        for word in tag_word_table[tag]:
            if tag_word_table[tag][word] != 1:
                check_freq[word] = tag
                hapax_word.pop(word,None)
                continue
            if word not in check_freq:
                check_freq[word] = tag
                hapax_word[word] = tag
            else:
                hapax_word.pop(word,None)
    none_class = hapax_word.copy()
    suffix_class, all_suffix = find_suffix_hapax(hapax_word)
    ori_freq_hapax_tag = {}
    total_hapax_tag = 0
    for word in hapax_word:
        if hapax_word[word] not in ori_freq_hapax_tag:
            ori_freq_hapax_tag[hapax_word[word]] = 0
        ori_freq_hapax_tag[hapax_word[word]] += 1
        total_hapax_tag += 1

    for tag in unique_tags:
        if tag not in ori_freq_hapax_tag:
            ori_freq_hapax_tag[tag] = 0

    for tag in ori_freq_hapax_tag:
        ori_freq_hapax_tag[tag] = (hapax_sp + ori_freq_hapax_tag[tag]) / (total_hapax_tag + hapax_sp*(V))

    tag_suffix_table = check_suffix_class(suffix_class,hapax_word,all_suffix,none_class)
    for suffix in suffix_class:
        for tag in unique_tags:
            if tag not in tag_suffix_table[suffix]:
                tag_suffix_table[suffix][tag] = 0

    for suffix in tag_suffix_table:
        suffix_n = sum(tag_suffix_table[suffix].values())
        suffix_V = len(tag_suffix_table[suffix])
        for tag in tag_suffix_table[suffix]:
            tag_suffix_table[suffix][tag] = (hapax_sp + (tag_suffix_table[suffix][tag])) / (suffix_n + hapax_sp*(suffix_V))
    
    freq_none_class = {}
    total_none_class = 0
    for word in none_class:
        if none_class[word] not in freq_none_class:
            freq_none_class[none_class[word]] = 0
        freq_none_class[none_class[word]] += 1
        total_hapax_tag += 1

    for tag in unique_tags:
        if tag not in freq_none_class:
            freq_none_class[tag] = 0

    for tag in freq_none_class:
        freq_none_class[tag] = (hapax_sp + freq_none_class[tag]) / (total_hapax_tag + hapax_sp*(V))

    
    return freq_none_class, tag_suffix_table, suffix_class, all_suffix, freq_none_class
    
def make_all_suffix(suffix_class):
    all_suffix = {}
    for pattern_suffix in suffix_class:
        suffix = pattern_suffix[8:]
        all_suffix[suffix] = len(suffix)
    return all_suffix

def check_suffix_class(suffix_class,hapax_word,all_suffix,none_class):
    tag_suffix_table = {}
    for pattern_suffix in suffix_class:
        tag_suffix_table[pattern_suffix] = {}
        for word in hapax_word:
            if hapax_word[word] not in tag_suffix_table[pattern_suffix]:
                tag_suffix_table[pattern_suffix][hapax_word[word]] = 0
            if word[-1*all_suffix[pattern_suffix[8:]]:] == pattern_suffix[8:]:
                tag_suffix_table[pattern_suffix][hapax_word[word]] += 1
                none_class.pop(word,None)
    #print(tag_suffix_table)
    return tag_suffix_table


def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    freq_tags, unique_tags, V, tag_word_table = make_freq_tags(train)
    freq_tag_pairs = make_freq_transition_pairs(train,unique_tags)
    freq_word_tag_pairs = make_freq_word_tag_pairs(train)

    n = sum(freq_tags.values())
    initial_sp = 0.00001
    transition_sp = 0.00001
    emission_sp = 0.00001
    hapax_sp = 0.00001

    hapax_tag_prob, tag_suffix_table, suffix_class, all_suffix, freq_none_class = make_hapax_prob(train,freq_word_tag_pairs,unique_tags,hapax_sp,V,tag_word_table)   
    num_additional_class = 1  

    initial_prob = make_initial_prob(freq_tags,unique_tags,initial_sp,train,V)
    transition_prob = make_transition_prob(freq_tag_pairs,freq_tags,V,transition_sp)
    emission_prob = make_emission_prob(freq_word_tag_pairs,V,freq_tags,emission_sp,hapax_tag_prob,tag_word_table)
    
    return decode(test,initial_prob,transition_prob,emission_prob,emission_sp,unique_tags,V,num_additional_class,n,hapax_tag_prob,all_suffix, tag_suffix_table, freq_none_class)