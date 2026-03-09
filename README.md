# Diffusion-models-for-channel-estimation

## Introduction
This repository contains an implementation of using generative diffusion models for high-dimensional channel estimation.

### Paper: [Generative Diffusion Models for High Dimensional Channel Estimation](https://ieeexplore.ieee.org/abstract/document/10930691), in IEEE TWC 2025.

## Dataset
The QuaDRiGa channel dataset used in this repository can be downloaded from [download url](https://drive.google.com/drive/folders/1imPAGQuJw9P4b1OyhY0Ac2rCdzzsEmEM?usp=drive_link)

## Steps to start
### 1. Clone the repository to your local machine and navigate into the project directory
### 2. Environment setup
It is highly recommended to use a virtual environment (e.g., Anaconda or venv) to manage your dependencies.
(Note: If a requirements.txt is provided in future updates, you can simply run pip install -r requirements.txt)
### 3. Dataset preparation
Download the dataset from theprovided Google Drive link.
Extract the downloaded files and place them into the data/ directory so that loaders.py can load them correctly.
### 4. Training
Run train_diffusion_cnn.py, and the model will be saved in the model/ directory.
### 5. Evaluation and testing
To evaluate the channel estimation performance using a trained model, run the testing script (test_diffusion_cnn.py). It will load the test dataset and the pre-trained weights.

## References
[1] X. Zhou, L. Liang, J. Zhang, P. Jiang, Y. Li, and S. Jin, “Generative diffusion models for high dimensional channel estimation,” IEEE Trans. Wireless Commun., vol. 24, no. 7, pp. 5840–5854, Jul. 2025.

[2] B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick, “Diffusion-based generative prior for low-complexity MIMO channel estimation,” Mar. 2024. [Online] Available: http://arxiv.org/abs/2403.03545.  (Repository: [url](https://github.com/benediktfesl/Diffusion_channel_est))

[3] M. Arvinte and J. I. Tamir, “MIMO channel estimation using score-based generative models,” IEEE Trans. Wireless Commun., vol. 22, no. 6, pp. 3698–3713, Jun. 2023.  (Repository: [url](https://github.com/utcsilab/score-based-channels))

## Contact
If you have any questions or comments about this work, please feel free to contact xy_zhou@seu.edu.cn.
