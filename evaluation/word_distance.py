import numpy as np
import pdb
import sys,getopt
import pandas as pd

nearest_words = 20

def read_vectors(vec_file):
        f = open(vec_file,'r')
        cnt = 0
        word_list = []
        embeddings = []
        word_size = 0
        embed_dim = 0
        for line in f:
            data = line.strip().split()
            if cnt == 0:
                word_size = int(data[0])
                embed_dim = int(data[1])
            else:
                word_list.append(data[0])
                tmpVec = [float(x) for x in data[1:]]
                embeddings.append(tmpVec)
            cnt = cnt + 1

        f.close()
        embeddings = np.array(embeddings)
        for i in range(int(word_size)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
        
        # print('read embeddings')
        # embeddings = pd.read_csv(vec_file, skiprows=1, usecols=list(range(1, 201)), header=list(range(201)), \
        #     encoding='utf8')
        # embeddings = embeddings.as_matrix()
        # file = codecs.open(vec_file, encoding = 'utf-8')
        # embeddings = np.loadtxt(file, dtype='float', skiprows=1, usecols=list(range(1, 201)))
        # for i in range(int(word_size)):
            # embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

        # print('read words')
        # file = codecs.open(file, encoding = 'utf-8')
        # word_list = np.loadtxt(vec_file, dtype='str', skiprows=1, usecols=(0))
        # print(word_list[0], embeddings[0])

        return word_size, embed_dim, word_list, embeddings


if  __name__ == '__main__':
        embed_file = ''
        try:
                opts, args = getopt.getopt(sys.argv[1:],"he:",["embed_file="])
        except getopt.GetoptError:
                print ('word_distance.py -e <embed_file>')
                sys.exit(2)
        for opt, arg in opts:
                if opt == '-h':
                        print ('word_distance.py -e <embed_file>')
                        sys.exit()
                elif opt in ("-e", "--embed_file"):
                        embed_file = arg

        word_size, embed_dim, word_list, embeddings = read_vectors(embed_file)
        print (word_size, embed_dim)
        while True:
            word = input("input a word:")

            # word1 = ''
            # word2 = ''
            # with open(pron_file, 'r') as file:
            #     word1 = file.readline().strip().split()
            #     word2 = file.readline().strip().split()
            # word_vec = []
            # for i in range(len(word1)):
            #     vec = (float(word1[i]) + float(word2[i])) / 2
            #     word_vec.append(vec)

            # word1 = ''
            # with open(pron_file, 'r') as file:
            #     word1 = file.readline().strip().split()
            # word_vec = [float(x) for x in word1]


            if word in word_list:
                    word_vec = embeddings[word_list.index(word)]


                    distance = []
                    for i in range(word_size):
                        distance.append(np.linalg.norm(embeddings[i] - word_vec))
                    print('end calculate distance')
                    nearest_index = list(range(word_size))
                    nearest_index = sorted(nearest_index, key=lambda x: distance[x])
                    nearest_word = [word_list[x] for x in nearest_index[:nearest_words]]
                    print('nearest word:')
                    for w in nearest_word:
                        print(w)
