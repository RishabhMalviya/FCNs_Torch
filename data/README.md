# Data

## PASCAL VOC12
 To get everything working with the scripts and data in this folder, download the *training/validation data* and *development kit code and documentation* from [this link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).

 Extract both tar files in this folder (they both put everything into the same folder, `VOCdevkit`), i.e., you should have the directory `VOCdevkit` and all of its directly under this directory (`data`).

 ### TODO

 Currently, the `loadDataset.lua` script takes waaaay too long to create data tensors from the files in the folder. Need to write CUDA kernels that will do the same jobs parallely; also, a lua file that will launch those kernels from C wrappers that it will bind to through the `ffi` library.

 ## SBD
 Apparently, this is a major extension to the segmentation data of the PASCAL VOC12 dataset (PASCAL VOC12 contains data annotated for a number of other tasks). It contains about 8000 images annotated for segmentation, whereas PASCAL VOC12 has only about 2000.

 [This mirror](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) should dowload the data.

 ### TODO
 Still need to download data from this dataset and write scripts to create appropriate torch Tensors.
