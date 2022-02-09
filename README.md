# Image inpainting using partial convolutions

Theory: https://arxiv.org/abs/1804.07723.

Image datasets used
===========
Nvidia flickr faces dataset: https://github.com/NVlabs/ffhq-dataset.

Celeb-a-hq faces dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html

All the images are augmented with random rotation, shift and sheer. The pipeline for preprocessing images is in create_dataset.py. Tfrecord files with serialized image tensors are to be used. The final functions takes list of names of the files.

Masks generation
===========
For masks generation set of random ellipses, circles and lines is used. Examples are below:

![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/a.png)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/b.png)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/c.png)
The generator is located in masks_generator.py file. Masks are expected to be pregenerated and stored in tfrecord files.

Obtained results
===========
All the code works on TPU. The model was trained on google TPU-v3 for about 6 hours. Hole loss from the article was replaced by loss between blurred images (using gaussian blur). It was found to be better in some cases.

![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/results.png)

![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/res11.png)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/res12.png)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/res13.png)

![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/res21.png)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/res22.png)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/res23.png)

In addition, the model was found to be successfull on pix2pix task (https://github.com/mojaevr/edges2cats ):

![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/cat0.jpg)
![alt text](https://github.com/paradro1d/ImageInpainting/blob/master/cat1.jpg)
