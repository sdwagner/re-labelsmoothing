# Replication Repository for the paper _When Does Label Smoothing Help?_ by Müller et al.

This repository contains the code and report for a replication of the paper

```
Rafael Müller, Simon Kornblith, and Geoffrey E. Hinton. "When does label smoothing
help?." Advances in neural information processing systems 32 (2019).
```

## 1. Installation

This implementation uses [Jupyter notebooks](https://jupyter.org/) to directly visualize the results while keeping the logs of the training process directly accessible.
Furthermore, it uses [PyTorch](https://pytorch.org/) to implement the different Neural Networks.  
Step One: Clone this Repo to the location of your liking.  
Step Two: The required packages can be installed as follows (using Python 3.8-3.11):

```bash
pip install -r requirements.txt
```

After conducting these steps, you should be able to run the Jupyter notebooks, provided you have an editor that can open them.
Possible Editors are [Visual Studio Code](https://code.visualstudio.com/) with the appropriate extensions or [JupyterLab](https://jupyter.org/install).

While most Datasets are downloaded automatically using PyTorch, there are two datasets that have to be downloaded manually and unpacked in the correct folder.

| Dataset       | Download-Link                                                                                      | Data-Folder              |
| ------------- | -------------------------------------------------------------------------------------------------- | ------------------------ |
| CUB-200-2011  | [CUB_200_2011.tgz](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) | `data/CUB_200_2011`      |
| Tiny ImageNet | [tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip)                          | `data/tiny-imagenet-200` |

Note that the files should be placed in a way that for CUB-200-2011, the "Data-Folder" should include the `classes.txt` file, and for Tiny ImageNet the "Data-Folder" should include the `words.txt` file.

If you intend to use the pre-trained models, this [Link](https://drive.google.com/drive/folders/1PCtHG5sWDRy_rK5fsqFdmA46oZwMVxuB?usp=sharing) provides the models trained by us to verify the results of the paper.
You just need to download the `models` folder and place it directly in the project root folder.

## 2. Folder Structure

The project consists of the following folder structure:

- `root`: This folder provides the notebooks to the corresponding experiments detailed in the replication paper.
- `datasets/`: This folder provides the wrapper classes for the used datasets with custom DataLoaders for some of them.
- `architectures/`: This folder provides the custom implementations of the different Neural Networks.
- `util/`: This folder provides Python files for utility modules, as well as training methods for the different architectures.
- `figures/`: This folder provides the figures used in our replication paper and some more.
- `models/`: In this folder, you should put the pre-trained models or your own trained models should appear here.
- `report/`: This folder contains the LaTeX source code of the replication report.

## 3. Using the Code

The available notebooks correspond to the following architectures, datasets, and experiments in the replication paper:
| File Name | Architecture | Dataset | Accuracy (Section 3) | Implicit Model Calibration (Section 5) | Knowledge Distillation (Section 6)|
| - | - | - | - | - | - |
| `FC_MNIST_Accuracy_IMC_Toy.ipynb`| Fully-Connected | MNIST | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[x] (Toy Example)</li></ul> |
| `FC_MNIST_KD.ipynb`| Fully-Connected | MNIST | <ul><li>[ ] </li></ul> | <ul><li>[ ] </li></ul> | <ul><li>[x] </li></ul> |
| `FC_EMNIST_Accuracy_IMC.ipynb`| Fully-Connected | EMNIST | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `FC_FMNIST_Accuracy_IMC.ipynb`| Fully-Connected | FMNIST | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `AlexNet_CIFAR10_Accuracy_IMC.ipynb`| AlexNet | CIFAR-10 | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `ResNet34_CUB-200-2011_Accuracy_IMC.ipynb`| ResNet-34 | CUB-200-2011 | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `ResNet50_TinyImageNet_Accuracy_IMC.ipynb`| ResNet-50 | TinyImageNet | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `ResNet56_CIFAR10_Accuracy_IMC.ipynb`| ResNet-56 | CIFAR-10 | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `ResNet56_CIFAR100_Accuracy_IMC.ipynb`| ResNet-56 | CIFAR-100 | <ul><li>[x] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |
| `ResNet56_AlexNet_CIFAR10_KD.ipynb`| ResNet-56/AlexNet | CIFAR-10 | <ul><li>[ ] </li></ul> | <ul><li>[ ] </li></ul> | <ul><li>[x] </li></ul> |
| `Transformer_Multi30K_IMC.ipynb`| Transformer | Multi30k | <ul><li>[ ] </li></ul> | <ul><li>[x] </li></ul> | <ul><li>[ ] </li></ul> |

Furthermore, the `PenultimateLayerRepresentation.ipynb` notebook combines experiments for the _Penultimate Layer Representation_ (Section 4).  
Inside the notebooks, you can execute the cells in which order you want.
While a full training cycle might take several hours (or days), using the pre-trained models, only the cells corresponding to the evaluation can be executed.
As such, the models can be used to directly validate our findings. Some of the evaluations might take several minutes.

Don't be afraid to try out our code and do your own experiments with it :)
