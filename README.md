# Physics-integrated variational autoencoders for robust and interpretable generative modeling

Implementation of the method presented in the following paper:

Naoya Takeishi and Alexandros Kalousis.
Physics-integrated variational autoencoders for robust and interpretable generative modeling.
In Advances in Neural Information Processing Systems 34 (NeurIPS), 2021.

[https://openreview.net/forum?id=0p0gt1Pn2Gv](https://openreview.net/forum?id=0p0gt1Pn2Gv)

[https://arxiv.org/abs/2102.13156](https://arxiv.org/abs/2102.13156)

NOTE: Some notations are not consistent between the paper and the codes here. If you find something too unclear, feel free to ask me! You can find contact information from my website.

## Prerequisite

- Python 3.8.3
- NumPy 1.19.2
- SciPy 1.5.2
- PyTorch 1.7.0
- pytorchvision 0.8.1 (for galaxy experiment)
- torchdiffeq (for pendulum/advdif/locomotion experiments)

## Usage

First, create data with `makedata.sh` or `makedata.m` in each data directory. You have to first download original data for the galaxy and locomotion datasets; see corresponding readme.txt.

You can train NN+phys(+reg) model using scripts [EXPTNAME]\_train.sh, for example by `. advdif_train.sh physnn`.
Then the .ipynb files perform some experiments shown in the paper.

## Author

**Naoya Takeishi** - [https://ntake.jp/](https://ntake.jp/)
