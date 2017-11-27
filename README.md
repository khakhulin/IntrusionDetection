# IntrusionDetetction

Intrusion Detetction algorithms for classification networks connections on 4 classes.


## Task and data

The data was chosen as KDD99. It contains data of approximately 4,000,000 connections, each of which has 41 attributes.
It's worth saying that the data set itself was very bad. First, it is very outdated. Secondly, the data is not balanced and contains a large number of duplicates.
At the moment, companies that develop data sets are not available for public access.

____

A connection is a sequence of TCP packets starting and ending at some well defined times, between which data flows to and from a source IP address to a target IP address under some well defined protocol. Each connection is labeled as either normal, or as an attack, with exactly one specific attack type. Each connection record consists of about 100 bytes.

Attacks fall into four main categories:

- DOS: denial-of-service, e.g. syn flood;
- R2L: unauthorized access from a remote machine, e.g. guessing password;
- U2R: unauthorized access to local superuser (root) privileges, e.g., various ''buffer overflow'' attacks;
- probing: surveillance and other probing, e.g., port scanning.

In addition, there are also normal connections.

The complete task description could be found [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html).

## Results 

As a metrics of classification score was selected F1-score. In order to take account influence of False positive examples.

We repeated the results of the studies presented in previous years and add some new approaches.



| Algorithm       | F1-macro |
|----------------|----------|
| Naive Bayes    | 0.3781   |
| SVM            | 0.6754   |
| Neural Network | 0.6835   |
| KNN            | 0.7047   |
| Random Forest  | 0.7832   |
| Xgboost        | 0.7904   |
| Catboost       | 0.8938   |


 This task in inspired by KDD CUP 1999 competition.