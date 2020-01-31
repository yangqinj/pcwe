# -*- coding: UTF-8 -*-
'''
Test the embeddings on word analogy task
Dataset:
'''
from word_sim import build_dictionary, read_vectors
import numpy as np
import random
import pdb
import sys,getopt

def read_word_analogy(ana_file):
        f1 = open(ana_file,'r')
        capital = []
        state = []
        family = []
        cnt = 0
        for line in f1:
                pair = line.split()
                if pair[0] == ':':
                        cnt = cnt + 1
                        continue
                if cnt == 1:
                        capital.append(pair)
                elif cnt == 2:
                        state.append(pair)
                else:
                        family.append(pair)
        f1.close()
        return capital,state,family


def predict_word(w1, w2, w3,w4, embeddings, dict_word, measure_func=0):
        #return the index of predicted word
        id1 = dict_word[w1]
        id2 = dict_word[w2]
        id3 = dict_word[w3]
        reverse_dict = dict(zip(dict_word.values(), dict_word.keys()))

        sim = ''
        if measure_func == 0:
                pattern = embeddings[id2] - embeddings[id1] + embeddings[id3]
                pattern = pattern / np.linalg.norm(pattern)
                # print (embeddings.shape, pattern.shape)
                sim = embeddings.dot(pattern.T)
        elif measure_func == 1:
                cos1 = embeddings.dot(embeddings[id2].T)
                cos1 = (cos1 + 1) / 2
                cos2 = embeddings.dot(embeddings[id3].T)
                cos2 = (cos2 + 1) / 2
                cos3 = embeddings.dot(embeddings[id1].T)
                cos3 = (cos3 + 1) / 2 + 0.001
                sim = cos1 * cos2 / cos3

        sim[id1] = sim[id2] = sim[id3] = -1   #remove the input words
        predict_index = np.argmax(sim)
        id4 = dict_word[w4]
        if predict_index == id4:
                return 1
        else:
                '''
                print(w1)
                print(w2)
                print(w3)
                print(w4)
                print(reverse_dict[predict_index])
                pdb.set_trace()
                '''
                return 0


def analogy(pairs, embeddings,dict_word, measure_func=0):
        total = len(pairs)
        reverse_dict = dict(zip(dict_word.values(), dict_word.keys()))
        in_dict_cnt = 0
        predict_cnt = 0
        print('dictionary_lengh ', len(dict_word))
        for pair in pairs:
                in_dict = True
                for i in range(len(pair)):
                        #pair[i] = pair[i].decode('utf-8')
                        in_dict = in_dict and (pair[i] in dict_word)
                if(in_dict):
                        in_dict_cnt = in_dict_cnt + 1
                        predict_cnt = predict_cnt + predict_word(pair[0], pair[1], pair[2],pair[3], embeddings, dict_word)
        return total,in_dict_cnt,predict_cnt


if  __name__ == '__main__':
        ana_f = ''
        embed_file = ''
        measure_func = 0

        try:
                opts, args = getopt.getopt(sys.argv[1:],"ha:e:f:",["analogy_file=","embed_file=", "measure_func="])
        except getopt.GetoptError:
                print ('word_analogy.py -a <analogy_file> -e <embed_file> -f <bool>')
                sys.exit(2)
        for opt, arg in opts:
                if opt == '-h':
                        print ('word_analogy.py -a <analogy_file> -e <embed_file> -f <bool>')
                        sys.exit()
                elif opt in ("-a", "--analogy_file"):
                        ana_f = arg
                elif opt in ("-e", "--embed_file"):
                        embed_file = arg
                elif opt in ("-f", "--measure_func"):
                        measure_func = int(arg)

        capital,state,family = read_word_analogy(ana_f)
        # print(len(capital), len(state), len(family))
        #pdb.set_trace()
        word_size, embed_dim, dict_word, embeddings = read_vectors(embed_file)
        # print(embeddings.shape)
        capital_total, capital_dict, capital_correct = analogy(capital, embeddings, dict_word, measure_func)
        state_total, state_dict, state_correct = analogy(state, embeddings, dict_word, measure_func)
        family_total, family_dict, family_correct = analogy(family, embeddings, dict_word, measure_func)
        total = capital_total + state_total + family_total
        indict = capital_dict + state_dict + family_dict
        correct = capital_correct + state_correct + family_correct
        print('capital total ', capital_total, ' in dict ', capital_dict, ' correct ', capital_correct, 'acc = ', capital_correct/capital_dict)
        print('state total ',state_total, ' in dict ', state_dict, ' correct ', state_correct, 'acc = ', state_correct/state_dict)
        print('family total ',family_total, ' in dict ', family_dict, ' correct ',family_correct, 'acc = ', family_correct/family_dict)
        print(' total ', total,' indict ',indict, ' correct ',correct, 'acc = ', correct/indict)
