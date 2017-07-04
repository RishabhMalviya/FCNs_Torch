# FCNs Torch
An implementation of the paper [Fully Convolutional Neural Networks for Image Segmentation](./papers/FCN.pdf) in Torch.

## models
The FCN models were taken from the `.prototxt` and `.caffemodel` files available in the [original repository](https://github.com/shelhamer/fcn.berkeleyvision.org), and converted using [loadcaffe](https://github.com/szagoruyko/loadcaffe). Any mistakes in the architecture that still persisted were remedied manually.

As of now, only the original (non FCN) architectures of two nets are available: AlexNet (from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)) and VGG_16 (from [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)). AlexNet requires [cudnn](https://github.com/soumith/cudnn.torch) to run, because the grouping operations shown in the original paper (study section 3.5 of `./papers/AlexNet.pdf`) are not available in `nn`, `nngraph`, or `cunn`.

---

### TODO
1. Make sure the cuddn version of AlexNet trains properly on GPU.
2. Write the lua file to return the net for FCN-8_VGG-16 and FCN-32_AlexNet (may need to go through .prototxt files manually).
3. Download the data (PASCAL VOC 12 dataset).
4. find out what loss function they use for the FCNs.
5. train
6. test, report results (what is mean IU?).
