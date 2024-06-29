## Enhanced Phishing Detection Model Based on ResNETS
Comparisons are made with different ResNET types, CNNs and other ML Models
The most well-known way of detecting a website for phishing is Blacklists, 
and although they work pretty well if given a good enough database to reference from, there are two issues with this model, 
one that they can get outdated very quickly and small domain changes or changes in parts of the URL can make the detection or parser invalid.

Methodology 
Usually, Neural Networks generally get better as the number of layers in their architecture is increased, but after a certain threshold, 
there is a common problem that encountered an increased error rate than its counterparts with lower number of layers, 
the exact reason for this has been attributed to the vanishing or exploding gradient descent problem also overfitting, 
this is well explained in [5]  where it’s called degradation. 
ResNets or Residual Networks is an idea proposed to solve this particular problem of degradation and the training of extremely deep networks, 
originally proposed in 2015 by researchers at Microsoft Research, although this is primarily used for Image classification, this repository is to show
it’s viability in detecting phishing URLs by comparing it with other Machine Learning techniques and classic convolutional neural networks.
