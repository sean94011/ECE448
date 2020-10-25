"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import numpy as np

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
            smoothed_prob[tag] = np.log((freq_tags[tag]+initial_sp)/(len(train)+initial_sp*(V)))
        else:
            smoothed_prob[tag] = np.log((initial_sp)/(len(train)+initial_sp*(V)))
    return smoothed_prob

def make_transition_prob(freq_tag_pairs,freq_tags,V,transition_sp):
    smoothed_prob = {}
    for tag_pair in freq_tag_pairs:
        smoothed_prob[tag_pair] = np.log((transition_sp + freq_tag_pairs[tag_pair]) / (freq_tags[tag_pair[0]] + transition_sp*(V)))
    return smoothed_prob


def make_emission_prob(freq_word_tag_pairs,V,freq_tags,emission_sp,tag_word_table):
    smoothed_prob = {}
    for w_t_pair in freq_word_tag_pairs:
        smoothed_prob[w_t_pair] = np.log((emission_sp + freq_word_tag_pairs[w_t_pair]) / (freq_tags[w_t_pair[1]] + emission_sp*(len(tag_word_table[w_t_pair[1]]))))
    return smoothed_prob




def decode(test,initial_prob,transition_prob,emission_prob,emission_sp,unique_tags,V,n):
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
                    p_e = np.log((emission_sp) / (n + emission_sp*(V+1)))

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




def viterbi_1(train, test):
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

    initial_prob = make_initial_prob(freq_tags,unique_tags,initial_sp,train,V)
    transition_prob = make_transition_prob(freq_tag_pairs,freq_tags,V,transition_sp)
    emission_prob = make_emission_prob(freq_word_tag_pairs,V,freq_tags,emission_sp,tag_word_table)
    
    return decode(test,initial_prob,transition_prob,emission_prob,emission_sp,unique_tags,V,n)