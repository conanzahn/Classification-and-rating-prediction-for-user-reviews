#!/usr/bin/env python3
# COMP9444 hw2
# Group ID:g023641
# Group Members: Haonan Zhang(z5151812), Hanrui Tao(z5237012)
"""
Question:
Briefly describe how your program works, and explain any design and training
decisions you made along the way.

Our Answer:
In this program, we pre-processed the text, built thenetwork layer, and then
converted the output of the network layer.

Before any processing of the text, we use Regular expressions to remove any
non alphanumeric characters. re.sub will substitute all non alphanumeric
characters with empty string. We also remain split() method in tokenise function
to break up text into smaller components of text. For the pre-processing, we
use stopwords removing words from a string that don’t provide any information
about the tone of a statement. We find that post-processed does not required.
After many tries, we decide word vector dimension is 100 for better performance.

For network structure, we use bidirectional LSTM and one Linear layer, we choose
RuLE activation function. For loss function, after many tries, we dicide to use
cross entropy loss.

We also made some modifications of training hyper-parameters. We change the
trainValSplit to 0.9, because more training data will improve the performance.
We change the optimiser from SGD to Adam, this improves accuracy significantly.
"""

# ------------------------------------------------------------------------------
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import sklearn
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    # use Regular expressions to remove any non alphanumeric characters.
    # re.sub will substitute all non alphanumeric characters with empty string.
    sample = re.sub('[^a-zA-Z0-9 ]+', '', sample)
    # breaking up text into smaller components of text
    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove illegal characters: punctuation, special characters and numbers
    # noise removal: remove punctuation
    # sample = re.sub(r'[\.\?\!\,\:\;\"]', '',sample)
    # use Regular expressions to remove any non alphanumeric characters.
    # re.sub will substitute all non alphanumeric characters with empty string.
    # sample = re.sub('[^a-zA-Z0-9 ]+', '', sample)   # [^A-Za-z0-9]

    # use lemmatization bringing words down to their root forms
    # lemmatizer = WordNetLemmatizer()
    # lemmatized = [lemmatizer.lemmatize(token) for token in sample]

    # use stemming bluntly removing word affixes
    # stemmer = PorterStemmer()
    # stemmed = [stemmer.stem(token) for token in lemmatized]

    # removing words from a string that don’t provide any information about the tone of a statement
    # stop_words = set(stopwords.words('english'))
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
                  "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
                  "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
                  "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                  "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
                  "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
                  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
                  "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                  "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
                  "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    statement_no_stop = [word for word in sample if word not in stop_words]
    sample = statement_no_stop
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    return batch

stopWords = {}
# set word dimension to 300
wordVectorDimension = 300
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # label ratingOutput
    # return the nearest integer
    # print("convert")
    # ratingOutput = ratingOutput.round()
    # # convert int to numpy
    # ratingOutput = np.array(ratingOutput)
    # # convert numpy to tensor
    # ratingOutput = torch.from_numpy(ratingOutput)
    # convert tensor to LongTensor
    # ratingOutput = ratingOutput.long()
    # set the value to 0 and 1
    # ratingOutput[ratingOutput > 1] = 1
    # ratingOutput[ratingOutput < 0] = 0
    ratingOutput = torch.argmax(ratingOutput,dim = 1)

    # label categoryOutput
    # return the nearest integer
    # categoryOutput = categoryOutput.round()
    # # convert int to numpy
    # categoryOutput = np.array(categoryOutput)
    # # convert numpy to tensor
    # categoryOutput = torch.from_numpy(categoryOutput)
    # convert tensor to LongTensor
    # categoryOutput = categoryOutput.long()
    # set the value to 0, 1, 2, 3, 4
    # categoryOutput[categoryOutput > 4] = 4
    # categoryOutput[categoryOutput < 0] = 0
    categoryOutput = torch.argmax(categoryOutput, dim=1)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """
    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(wordVectorDimension, hidden_size=50, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.out1 = tnn.Sequential(
            tnn.Linear(200, 64),
            tnn.ReLU(),
            tnn.Linear(64, 2)
        )
        self.out2 = tnn.Sequential(
            tnn.Linear(200, 64),
            tnn.ReLU(),
            tnn.Linear(64, 5)
        )
        # dropout = 0.5
        # vocab_size =100
        # # self.embedding = tnn.Embedding(vocab_size, wordVectorDimension)
        # self.lstm = tnn.LSTM(wordVectorDimension, hidden_size=50, num_layers=2, bidirectional=True, dropout=0.5)
        # # self.linear = tnn.Linear(2 * 50, 64)
        # self.relu = tnn.ReLU()
        # self.out1 = tnn.Linear(100, 2)
        # self.out2 = tnn.Linear(100, 5)
        # self.dropout = tnn.Dropout(dropout)


    def forward(self, input, length):
        # # text = [seq len, batch size]
        # # lengths = [batch size]
        # # input = input.long()
        # # embedded = self.dropout(self.embedding(input))
        # embedded = self.dropout(input)
        # # embedded = [seq len, batch size, emb dim]
        # packed_embedded = tnn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)
        # packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # output, _ = tnn.utils.rnn.pad_packed_sequence(packed_output)
        # # outputs = [seq_len, batch size, n directions * hid dim]
        # # hidden = [n layers * n directions, batch size, hid dim]
        # hidden_fwd = hidden[-2]
        # hidden_bck = hidden[-1]
        # # hidden_fwd/bck = [batch size, hid dim]
        # hidden = torch.cat((hidden_fwd, hidden_bck), dim=1)
        # # hidden = [batch size, hid dim * 2]
        # prediction = self.dropout(hidden)
        # prediction = self.relu(prediction)
        # prediction1 = self.out1(prediction)
        # prediction2 = self.out2(prediction)
        # # prediction = [batch size, output dim]
        # return prediction1, prediction2

        out, (hide, cell) = self.lstm(input)
        # concatenate the output of normal order and reversed order
        x = torch.cat((out[:, -1, :], out[:, 0, :]), dim=1)
        out1 = self.out1(x)
        out2 = self.out2(x)
        return out1, out2

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        # self.BCEWithLogitsLoss = tnn.BCEWithLogitsLoss()
        self.CrossEntropyLoss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingLoss = self.CrossEntropyLoss(ratingOutput, ratingTarget)
        categoryLoss = self.CrossEntropyLoss(categoryOutput, categoryTarget)
        final_loss = ratingLoss + categoryLoss
        # result = torch.mean(final_loss)
        return final_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.95
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.001)
