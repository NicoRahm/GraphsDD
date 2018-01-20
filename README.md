# GraphsDD
This project is a Python implementation of different methods for generating directed simple graphs from a prescribed in-out degree sequence. We first explored a Havel-Hakimi type algorithm for finding a realization of the bi-sequence. Then, sampling can be done in a MCMC-style using edge swaps. Another track is also explored with a rejection scheme using the configuration model. Finally we implemented an independant sampler on the whole graph class which was motivated by a finer analysis of the Havel-Hakimi scheme. 

We also tried to quantify the differences between the generated graphs using 5 different similarity measures. 

More details about the methods, the experiments and some references are presented in our technical report (please note that if the NIPS format was used, this report is not a NIPS publication). 

This project was part of the course Graphs in Machine Learning of the master MVA (http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/) and we thank Philippe Preux for having tutoring this project.

### Setup
You should install the igraph library and the cairocffi for graph visualization 
``` bash 
pip install python-igraph
pip install cairocffi
``` 

### Authors
**Nicolas Rahmouni** 

**Mhamed Hjaiej** 



