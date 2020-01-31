C++ implementation of Pronunciation-enhanced Chinese Word Embedding (PCWE).

# Directory Structure
	
	```
	├─PCWE
		├─dataset
		│
		├─evaluation
		│	├─240.txt
		│	├─297.txt
		│	├─analogy.txt
		│	├─word_sim.py
		│	├─word_analogy.py
		│
		├─src
		│	├─pcwe.c
		│	├─makefile
		│	├─run.sh
		│
		├─subcharacter
		│	├─char2comp.txt
		│	├─char2radical.txt
		│	├─comp.txt
		│	├─pron.txt
		│	├─pron_tone.txt
		│	├─radical.txt
		│	├─word2pron.txt
		│
		├─README.md

	```
## "dataset" directory
This directory contains the training corpora and its learning embeddings.

## "src" directory
This directory contains C implementation code of PCWE, makefile and run.sh shell script.

## "subcharacter" directory
This folder contians the subcharacter data we collected.

1. radical.txt, comp.txt, pron_tone.txt are radical list, component list and pinyin list respectively.
2. char2radical.txt, char2comp.txt and word2pron.txt are the mapping between Chinese characters and their radicals or their components respectively.

**NOTE**
1. radical.txt, comp.txt, char2radical.txt and char2comp.txt files are provided by (Yu et al. 2017). If you want to use them in your paper, please cite their paper.
2. pron_tone.txt are crawled from [Online Xinhua Dictionary](http://xh.5156edu.com/pinyi.html). word2pron.txt are obtained by transforming vocabulary in training corpus to pinyin by tool [HanLP](https://github.com/hankcs/HanLP/).

## "evaluation" directory
This directory contains evaluation datasets and codes for word similarity and word analogy reasoning tasks.


# Compiling

On Unix/Linux/Cygwin/MinGW environmens, go to directory of "./src", type:
	$ make clean
	$ make all

# Learn Word Embedding
Go to the directory of "./src", run the shell script "run.sh":
	$ ./run.sh

If an permission error occurs, type:
	$ chmod +x run.sh
and then rerun "run.sh"

"run.sh" contains command line to use pcwe

	$ ./pcwe -train <train_file> -output-word <word_vec_file> -output-char <char_vec_file> -output-comp <comp_vec_file> -output-pron <pron_vec_file> -size <int> -window <int> -sample <float> -negative <int> -iter <int> -threads <int> -min-count <int> -alpha <float> -binary <int> -comp <comp_file> -char2comp <char2comp_file> -pron <pron_file> -word2pron <word2pron_file> -join-type <int> -pos-type <int> -average-sum <int>

	where:
	-train <train_file>:
		The training corpus file.

	-output-word <word_vec_file>:
		The output word embedding file.

	-output-char <char_vec_file>:
		The output character embedding file.

	-output-comp <comp_vec_file>:
		The output componnet embedding file.

	-output-pron <pron_vec_file>:
		The output pronunciation embedding file.

	-size <int>:
		The dimension of embedding. Embeddings of words, characters, components and pronunciations have same dimension.

	-window <int>:
		The size of context window.

	-sample <float>:
		The threshold of high frequency words.

	-negative <int>:
		The size of negative samples. Must greater than 0.

	-iter <int>:
		The iteration times.

	-threads <int>:
		The number of threads.

	-min-count <int>:
		The minimum frequency of words.

	-alpha <float>:
		The subsampling parameter.

	-binary <int>:
		Whether save embeddings as binary format.

	-comp <comp_file>:
		The componnet list file.

	-char2comp <char2comp_file>:
		The file that maps character to components.

	-pron <pron_file>:
		The pronunciation list file.

	-word2pron <word2pron_file>:
		The file that maps word to its pronunciation.

	-join-type <int>:
		Joint type of words, characters, components and pronunciations(default = 1: individual, 2: collective).

	-pos-type <int>:
		The type of pronunciatoin's position (default = 1: use the components of surrounding words, 2: use the components of the target word, 3: use both)

	-average-sum <int>:
		Compose way of context. (default = 1: average, 2: sum).

Example: 
	$ ./pcwe -train ../dataset/zh_wiki_small -output-word ../dataset/word_vec -output-char ../dataset/char_vec -output-comp ../dataset/comp_vec -output-pron ../dataset/pron_vec -size 200 -window 5 -sample 1e-4 -negative 10 -iter 100 -threads 24 -min-count 5 -alpha 0.025 -binary 0 -comp ../subcharacter/comp.txt -char2comp ../subcharacter/char2comp.txt -pron ../subcharacter/pron_tone.txt -word2pron ../subcharacter/word2pron.txt -join-type 1 -pos-type 3 -average-sum 1


# Evaluation

### Word Similarity
word_sim.py is the code for evaluating similarity between word embeddings. 240.txt and 297.txt are two datasets provided by [(Chen et al., 2015)](https://github.com/Leonard-Xu/CWE)

To run word_sim.py, type:

	$ word_sim.py -s <similarity_file> -e <embed_file>

	where:
	-s <similarity_file>:
		The word similarity dataset (240.txt or 297.txt).

	-e <embed_file>
		The word embeddings learned by PCWE.

### Word Analogy Reasoning
word_analogy.py is the code for word analogy reasoning tasks and analogy.txt is evaluation dataset provided by [(Chen et al., 2015)](https://github.com/Leonard-Xu/CWE)

To run word_analogy.py, type:
	
	$ word_analogy.py -a <analogy_file> -e <embed_file> -f <bool>

	where:
	-a <analogy_file>:
		The word analogy dataset (analogy.txt).

	-e <embed_file>:
		The word embeddings learned by PCWE.

	-f <bool>:
		The measure function: default = 0: 3CosAdd, 1: 3CosMul.

### Text classification
	The dataset for text classification task is Fudan corpus. You can obtain training and testing dataset from [here](http://download.csdn.net/download/github_36326955/9747927) and [here](http://download.csdn.net/download/github_36326955/9747929). The classifier is [LIBLINEAR](https://github.com/cjlin1/liblinear).


# References

(Chen et al., 2015) X. Chen, L. Xu, Z. Liu, M. Sun, and H. Luan, “Joint learning of character and word embeddings,” in Proceedings of the Twenty-Fourth International Joint Conference on Artificial Intelligence (IJCAI 2015) Joint, 2015, vol. 2015–Janua, no. Ijcai, pp. 1236–1242.

(Yu et al. 2017) J. Yu, X. Jian, H. Xin, and Y. Song, “Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components,” in EMNLP, 2017, pp. 286–291.
