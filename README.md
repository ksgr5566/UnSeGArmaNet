# UnSeGArmaNet
Official code for the paper UnSeGArmaNet: Unsupervised Image Segmentation using Graph Neural Networks with Convolutional ARMA Filters

# Abstract
The data-hungry approach of supervised classification drives the interest of the researchers toward unsupervised approaches, especially for problems such as medical image segmentation, where labeled data are difficult to get. Motivated by the recent success of Vision transformers (ViT) in various computer vision tasks, we propose an unsupervised segmentation framework with a pre-trained ViT. Moreover, by harnessing the graph structure inherent within the image, the proposed method achieves a notable performance in segmentation, especially in medical images. We further introduce a modularity-based loss function coupled with an Auto-Regressive Moving Average (ARMA) filter to capture the inherent graph topology within the image. Finally, we observe that employing Scaled Exponential Linear Unit (SELU) and SILU (Swish) activation functions within the proposed Graph Neural Network (GNN) architecture enhances the performance of segmentation. The proposed method provides state-of-the-art performance (even comparable to supervised methods) on benchmark image segmentation datasets such as ECSSD, DUTS, and CUB, as well as challenging medical image segmentation datasets such as KVASIR, CVC-ClinicDB, ISIC-2018, and ETIS.

# Steps to set up the repository

1. Clone and cd to this repo.
2. This project uses Python 3.11 version. Run `python3 -m venv venv`.
3. If Windows:
     `.\venv\Scripts\activate`
   <br/>
   If Mac:
     `. venv/bin/activate`
4. Pip install the required torch version from [here](https://pytorch.org/). In this project we are using PyTorch 2.0 version.
5. Make the script.sh executable, run: `./script.sh`
6. Follow steps mentioned [here](/datasets/DATASETS.md) to setup the datasets and follow the below execution steps.

Note: For some reason, ECSSD (via deeplake) is not getting installed on colab. Kindly run it in another environment, if you face any issue.\
Currently in this repo, only CUB, ECSSD and DUTS are supported. Other datasets are being added soon.

# Execution:
You can run `python main.py` with the following arguments.

| Arguments      | Possible Values                                     | Default    |
|----------------|-----------------------------------------------------|------------|
| bs             | boolean                                             | False      |
| epochs         | positive int                                        | 20         |
| resolution     | two positive ints                                               | [224, 224] |
| activation | "deepcut_activation", "relu", "silu", "gelu", "selu" | "selu"     |
| loss_type      | "DMON", "NCUT"                                      | "DMON"     |
| process        | "KMEANS_DINO", "DINO", "MEDSAM_INFERENCE" | "DINO"     |
| dataset        | "CUB", "ECSSD", "DUTS"                                     | "ECSSD"    |
| threshold      | float                                               | 0          |
| conv_type      | "ARMA", "GCN"                                       | "ARMA"     |

Example call:\
`python main.py --dataset "ECSSD" --threshold 0.1 --bs False --epochs 20 --resolution 224 224 --activation "selu" --loss_type "DMON" --process "DINO" --conv_type "ARMA"`

# Acknowledgements
We extend our heartfelt gratitude to the creators and contributors of [Deepcut](https://github.com/SAMPL-Weizmann/DeepCut) which laid the foundation for our code, which is licensed under the [MIT License](https://github.com/SAMPL-Weizmann/DeepCut/blob/main/LICENSE.txt).
