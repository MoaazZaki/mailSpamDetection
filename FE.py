import numpy as np
import pandas as pd

# Constants
MAX_VOCAB_SIZE = 50000
EMBEDDING_DIM = 5000
DELTA = 1e-12
BETA = 1e-6
GAMMA = 1e-3


class Vocabulary:
    """
    Description
    -----------
    - Keeps the words in dataset with count
    - Gives a token for each word
    """

    def __init__(self, max_vocab_size=-1):
        """  
        Description
        -----------
          Initialize vocabulary 

        Parameters
        ----------
        max_vocab_size : int
          maximum vocabulary size
        """
        # Members
        self.word_to_index = dict()
        self.index_to_word = dict()
        self.word_count = pd.Series(dtype=np.int32)
        self.unique_word_count = pd.Series(dtype=np.int32)
        self.prev_sentence_index = -1
        self.vocab_size = 0

        self.max_vocab_size = max_vocab_size

        self.word_to_index[' '] = self.vocab_size
        self.index_to_word[self.vocab_size] = '<empty>'
        self.vocab_size += 1
        self.word_count.loc['<empty>'] = GAMMA

        self.word_to_index['<unkown>'] = self.vocab_size
        self.index_to_word[self.vocab_size] = '<unkown>'
        self.vocab_size += 1
        self.word_count.loc['<unkown>'] = GAMMA

        self.tf_dict = {}
        self.tf_dict['<unkown>'] = {}
        self.tf_dict['<empty>'] = {}

        self.bow = []
        self.unique_words = {}  # for bow method

    def __len__(self):
        """
        Description
        -----------
          Get the size of vocabulary
        """
        return self.vocab_size

    def __getitem__(self, key):
        """
        Description
        -----------
          Get the size of vocabulary

        Parameters
        ----------
        key : int/str
          Index/word to get its corresponding word/index

        Returns
        -------
        int/str
          Query
        """

        # If key is string
        if type(key) == type('ss'):
            query = 1
            try:
                query = self.word_to_index[key]
            except:
                pass
            return query
        # If key is integer
        elif type(key) == type(50):
            query = 0
            try:
                query = self.index_to_word[key]
            except:
                raise KeyError('Index out of range')
            return query
        # If key is an unknown type
        else:
            raise KeyError("Invalid key type, key must be string or integer")

    def add_word(self, word, sentence_index=0, sentence_len=1, calculate_tf=False):
        """
        Description
        -----------
          Add word to the vocabulary

        Parameters
        ----------
        word : str
          Word to be added

        Returns
        -------
        bool
          The state of adding the word (success/fail)
        """

        try:
            self.word_count.loc[word] += 1
            if self.prev_sentence_index != sentence_index:
                self.unique_word_count[word] += 1
                self.prev_sentence_index = sentence_index
            if calculate_tf:
                try:
                    self.tf_dict[word][sentence_index] += 1 / \
                        (sentence_len+GAMMA)
                except:
                    self.tf_dict[word][sentence_index] = 1/(sentence_len+GAMMA)
        except:
            # If the vocab reached max size
            if self.vocab_size == self.max_vocab_size:
                return False
            # Adding new word
            self.word_count.loc[word] = 1
            self.unique_word_count.loc[word] = 1
            self.prev_sentence_index = sentence_index
            self.word_to_index[word] = self.vocab_size
            self.index_to_word[self.vocab_size] = word
            self.vocab_size += 1
            if calculate_tf:
                self.tf_dict[word] = {sentence_index: 1/(sentence_len+GAMMA)}

        return True

    def get_vocab_words(self):
        """
        Description
        -----------
          Get the word in vocab

        Returns
        -------
        np.aaray
          words
        """
        return self.index_to_word.values()

    def create_tfidf_matrix(self, df, X='Body'):
        """
        Description
        -----------
          Get Tf-idf matrix of given dataset

        Parameters
        ----------
        df : pd.DataFrame
          Dataset to be processed

        X : str
          Name of text column in df

        Returns
        -------
        None
        """
        self.tfidf_matrix = np.empty(
            [len(df), self.vocab_size], dtype=np.float64)

        for word, dic in self.tf_dict.items():
            for index, word_tf in dic.items():
                idf = np.log(len(df) / self.unique_word_count.loc[word])
                self.tfidf_matrix[index,
                                  self.word_to_index[word]] = word_tf * idf

    def add_vectorizer(self, vectorizer):
        """
        Description
        -----------
          Add vectorized object

        Parameters
        ----------
        word : str
          Word to be added

        Returns
        -------
        bool
          The state of adding the word (success/fail)
        """
        self.vectorizer = vectorizer

    def word_to_vector(self, word):
        vector = np.zeros(EMBEDDING_DIM)
        try:
            vector = self.vectorizer[word]
        except:
            pass
        return vector

    def create_embedding_matrix(self):
        """
        Description
        -----------
          Get embedding matrix of given dataset for stored vocabulary

        Returns
        -------
        None
        """
        self.embedding_matrix = np.empty(
            [self.vocab_size, EMBEDDING_DIM], dtype=np.float64)
        for index, word in self.index_to_word.items():
            self.embedding_matrix[index] = self.word_to_vector(word)

    def create_vocab(self, df, X='Body', calculate_tf=False, calculate_tfidf=False):
        """
        Description
        -----------
          Create a vocab of given dataset

        Parameters
        ----------
        df : pd.DataFrame
          Dataset to be processed

        X : str
          Name of text column in df

        calculate_tf : boolean (default = False)
          flag for storing term frequency while creating vocab

        calculate_idf : boolean (default = False)
          flag for storing inverse document frequency while creating vocab
        Returns
        -------
        None
        """
        for index, sentence in enumerate(df[X]):
            for word in sentence:
                self.add_word(word, index, len(sentence), calculate_tf)
        if calculate_tfidf:
            self.create_tfidf_matrix(df)

    def create_bag_of_words_matrix(self, df, X='Body'):
        """
        Description
        -----------
          Create a bag of words matrix givan dataframe

        Parameters
        ----------
        df : pd.DataFrame
          Dataset to be processed

        X : str
          Name of text column in df

        Returns
        -------
        None
        """
        for sentence in df['Body']:
            sentence_dict = {}
            for word in sentence:
                if word not in sentence_dict:
                    sentence_dict[word] = 1
                    if word not in self.unique_words:
                        self.unique_words[word] = 1
                else:
                    sentence_dict[word] += 1
            self.bow.append(sentence_dict)
            self.vocab_size = len(self.unique_words)

    def get_row(self, index):
        """
        Description
        -----------
          Get a vector represents row using BOW matrix

        Parameters
        ----------
        index : int
          row index


        Returns
        -------
        list
          vector of words in this row
        list
          vector of values of returned words
        """
        return list(self.bow[index].keys()), list(self.bow[index].values())
