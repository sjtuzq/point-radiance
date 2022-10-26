# point-radiance

---

This code release accompanies the following paper:

### Differentiable Point-Based Radiance Fields for Efficient View Synthesis
Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiweicz, Felix Heide

*Siggraph Asia*, 2022

 [PDF](https://arxiv.org/pdf/2205.14330.pdf) | [arXiv](https://arxiv.org/abs/2205.14330) 
**Abstract:** 
We propose a differentiable rendering algorithm for efficient novel
view synthesis. By departing from volume-based representations
in favor of a learned point representation, we improve on existing
methods more than an order of magnitude in memory and run-
time, both in training and inference. The method begins with a
uniformly-sampled random point cloud and learns per-point posi-
tion and view-dependent appearance, using a differentiable splat-
based renderer to train the model to reproduce a set of input train-
ing images with the given pose. Our method is up to 300 × faster
than NeRF in both training and inference, with only a marginal
sacrifice in quality, while using less than 10 MB of memory for a
static scene. For dynamic scenes, our method trains two orders of
magnitude faster than STNeRF and renders at a near interactive
rate, while maintaining high image quality and temporal coherence
even without imposing any temporal-coherency regularizers.


## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements:

```bash
# Create and activate new conda env
conda create -n my-conda-env python=3.9
conda activate my-conda-env

# Install pytorch and related libraries
conda install -y pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.0 -c pytorch
conda install numpy matplotlib tqdm imageio
```
Then follow the official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install [pytorch3d](https://pytorch3d.org/).

## Reproduce
You can train the model on NeRF synthetic dataset within 3 minutes. Here datadir is the dataset folder path. Dataname is the scene name. Basedir is the log folder path. Data_r is the ratio between the used point number and the initialized point number. Splatting_r is the radius for the splatting.
```bash
python main.py --datadir xxx --dataname hotdog --basedir xxx --data_r 0.012 --splatting_r 0.015
```
After around three minutes, you can see the following output (the example is tested on one A100 GPU):

```
Training time: 148.59 s
Rendering quality: 34.70 dB
Rendering speed: 120.01 fps
Model size: 7.32 MB
```

## Citation

If you find this work useful for your research, please consider citing:
```
@article{zhang2022differentiable,
  title={Differentiable Point-Based Radiance Fields for Efficient View Synthesis},
  author={Zhang, Qiang and Baek, Seung-Hwan and Rusinkiewicz, Szymon and Heide, Felix},
  journal={arXiv preprint arXiv:2205.14330},
  year={2022}
}
```
