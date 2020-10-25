"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    ''' 
    freq_table = {}
    freq_tags = {}
    for sentence in train:
        for word in sentence:
            if word[0] not in freq_table:
                    freq_table[word[0]] = {}
            if word[1] not in freq_table[word[0]]:
                    freq_table[word[0]][word[1]] = 0
            freq_table[word[0]][word[1]] += 1
            if word[1] not in freq_tags:
                freq_tags[word[1]] = 0
            freq_tags[word[1]] += 1

    max_tag = max(freq_tags, key=freq_tags.get)

    tagged_sentences = []
    for sentence in test:
        cur_sentence = []
        for word in sentence:
            if word not in freq_table:
                cur_sentence.append((word,max_tag))
                continue
            cur_tag = max(freq_table[word], key=freq_table[word].get)
            cur_sentence.append((word,cur_tag))
        tagged_sentences.append(cur_sentence)
    return tagged_sentences