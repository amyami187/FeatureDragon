# FeatureDragon
... and how to train it ;)


In this repo, we propose an ansatz to implement any gate-based feature map in the quantum distance-based classifier (https://arxiv.org/abs/1703.10793). Due to the increasing depth of the circuit as more data points are added, we built instead an ensemble of quantum classifiers using batches of smaller subsets of training data. The ensemble is then optimised using PyTorch optimisers. 
