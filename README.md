# Bi-Directional-Channel-Reciprocity
Python code and related materials for DualNet and U2D network.
## Introduction 
This repository contains the code and related materials for DualNet and its extension U2D network. DualNet is described in 
Zhenyu Liu, Lin Zhang, and Zhi Ding, “Exploiting Bi-Directional Channel Reciprocity in Deep Learning for Low Rate Massive MIMO CSI Feedback,” IEEE Wireless Communications Letters, 2019. [Online]. Available: https://ieeexplore.ieee.org/document/8638509/. U2D network has been submitted.
## Requirements
- Python 3.5 (or 3.6)
- Keras (>=2.1.1)
- Tensorflow (>=1.4)
- Numpy
## Data Set
The CSI data is generated using COST 2100 channel model. We will upload the data set later. You can refer the paper below and the corresponding implementations: L. Liu, J. Poutanen, F. Quitin, K. Haneda, F. Tufvesson, P. De Doncker, P. Vainikainen and C. Oestges, “The COST 2100 MIMO channel model,” IEEE Wireless Commun., vol 19, issue 6, pp 92-99, Dec. 2012. [Online]. Available: https://ieeexplore.ieee.org/document/6393523/

We have uploaded the indoor data set for DualNet. Normalization is required using the file "training_testing_data_generation.m" to generate the training set and testing set.

https://www.dropbox.com/s/wmi2wuq4betzryu/mat_indoor5351_bw20MHz_up.mat?dl=0

https://www.dropbox.com/s/av0u0m9kfr95vtf/mat_indoor5351_bw20MHz_down.mat?dl=0
## CsiNet
The implementation of CsiNet can be found in https://github.com/sydney222/Python_CsiNet. Thank authors for sharing their code.
