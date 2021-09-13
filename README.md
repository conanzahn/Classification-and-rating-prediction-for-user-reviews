# Classification and rating prediction for user reviews (Deep Learning)

UNSW COMP9444 Neural Networks and Deep Learning



## Group

Group ID: g023641

Group Members: Haonan Zhang(z5151812), Hanrui Tao(z5237012)



## Description

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



## Thanks

If you like this website don't forget give it a ⭐ and also feel free to share feedback with me [here](mailto:conanzahn@gmail.com).

If you would like to communicate web development skills with me, feel free to contact me in Wechat: Zahnisme