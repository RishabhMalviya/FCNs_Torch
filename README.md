--- I have implemented this successfully in PyTorch, and will soon be putting up that code. This repo is basically deprecated.  ---

# FCNs Torch
An implementation of the paper [Fully Convolutional Neural Networks for Image Segmentation](./papers/FCN.pdf) in Torch.

## models
The FCN models were taken from the `.prototxt` and `.caffemodel` files available in the [original repository](https://github.com/shelhamer/fcn.berkeleyvision.org), and converted using [loadcaffe](https://github.com/szagoruyko/loadcaffe). Any mistakes in the architecture that still persisted were remedied manually.

As of now, only the original (non FCN) architectures of two nets are available: AlexNet (from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)) and VGG_16 (from [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)). AlexNet requires [cudnn](https://github.com/soumith/cudnn.torch) to run, because the grouping operations shown in the original paper (study section 3.5 of `./papers/AlexNet.pdf`) are not available in `nn`, `nngraph`, or `cunn`.

---

### TODO
1. Make sure the cuddn version of AlexNet trains properly on GPU. [This](https://gist.github.com/Kaixhin/68ffc5a2d1a69cc1556f1b2d2f1ae345) example should help. I can't test this on my laptop, because I've only got a Fermi micro-architecture GPU (aka 'compute cpapbility 2.0'), and CuDNN needs at least compute capability 3.0. I'll write out the (hopefully correct code) for the network architecture, and anyone with a compatible GPU is welcome to check if it's fine.
2. Write the lua file to return the net for FCN-8_VGG-16 and FCN-32_AlexNet (may need to go through .prototxt files manually).
3. Download the data (PASCAL VOC 12 dataset).
4. Find out what loss function they use for the FCNs.
5. Train
6. Test, report results (what is mean IU?).
