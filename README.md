# Denoising grayscale images with convolutional autoencoder and study its neural representations
First part of this work implements deep Convolutional Autoencoder in PyTorch for image denoising.
The second part studies the neural representation of the autoencoder encoder
last latent space, and its development along the training.

50X50 grayscale images were used as the dataset. The images describe a rat running in a 
round arena with 6 ports around the walls. Each frame is labeled with multiple features:
 Location of the rat, the direction of its torso, its distance to the center of the arena.

# Tl;dr
1. Convolutional autoencoder successfully denoise grayscale images.
2. The last latent space of the autoencoder encoder encodes the structure of the arena 
and the features of the rat.
3. The encoding development can be studies by the UMAP dimensionality reduction algorithm.
 
   The figure below shows how the 2D UMAP of the latent space representation changes with training
    <p align="center"><img width="350" height="400" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/2d_UMAP_change_with training.gif"></p>
4. I replaced the encoder by a PCA and trained a decoder on the resulting low dimensional features.
The performance was unaffected while the training time reduced by half.
5. Objective of the learning affects neural representation of the latent space, 
even if the architecture of the network is the same.

 
## Table of content:
- Installation instructions and technical information hoe to run the scripts can be found in the file ['Installation_Readme.md'](https://github.com/doronharitan/autoencoder/blob/master/Installation_Readme.md)
- [Convolutional AutoEncoder](#convolutional-autoencoder) - short explanation on what I did and how I used it 
- [Convolutional autoencoder denoises grayscale images.](#convolutional-autoencoder-denoises-grayscale-images) 
- [The last latent space of the autoencoder encoder encodes the structure 
of the arena  and the features of the rat.](#the-last-latent-space-of-the-autoencoder-encoder-encodes-the-structure-of-the-arena--and-the-features-of-the-rat)
- [Replacing the encoder part by non-convolutional dimensionality reduction techniques](#replacing-the-encoder-part-by-non-convolutional-dimensionality-reduction-techniques) of my project
-[How the objective of the learning affects the latent space neural Representation](#how-the-objective-of-the-learning-affects-the-latent-space-neural-representation)

## Convolutional AutoEncoder
  The AE (Autoencoder) model contains two parts: 
1.	An encoder that reduces the dimensionality of the input to a low-dimensional latent space (in my model its 16D).I refer to this latent space later on by the term ‘latent space’. I used convolutional and max-pooling layers, followed by a fully connected layer. 
2.	A decoder that reconstructs the image to its original dimensions. I used fully
 connected layer followed by transposed convolutional layers. 
The objective function is L2 distance between the input and output images.

 The network architecture:
  <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/ae_network.jpg"></p>

###  Convolutional autoencoder denoises grayscale images.
A video showing a stack of sample images from the test dataset before the de-noise and after is seen below: 
<p align="center"><img width="300" height="150" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/input_vs_output_image.gif"></p>
From the comparison above we can learn that the autoencoder was able to denoise the images. 

Note: the dataset noise comes from the imaging system and not syntactically introduced.

### The last latent space of the autoencoder encoder encodes the structure of the arena  and the features of the rat.
I was curious to learn **what is encoded in the AE latent space**? 
   
To address this question, I transformed all the images to a latent space using 
the encoder. 
In order to visualize the resulting latent space I reduced its dimension from 16D to 2D using Uniform Manifold Approximation and Projection
([UMAP](https://umap-learn.readthedocs.io/en/latest/)).
I colored the 2D data-points according to a single feature of the dataset, 
such as the body angle of the rat (Fig A). If the points are cluster by color or arranged in a color gradient it suggests 
that the latent space embeds this feature in its representation.
The features I colored  were: 
 1. The body angle of the rat (Fig A)
 2. The distance of the rat from the arena center (Fig B).
      <p align="center"><img width="355" height="300" vspace="100" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/body_angle_of_the_rat.jpg"> 
    <img width="330" height="300" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/dis_from_arena_center.jpg"></p>

  The 2D point cloud is shaped in a circular structure with 6 elements that comes out form the circle. This spatial
   arrangements correlates with the arena geometry. 
   
   The color gradients and cluster in the plots suggests that the distance of the rat form the center of the arena and 
   its body angle are embedded in the latent space. 

### Replacing the encoder part by non-convolutional dimensionality reduction techniques.
The new model contains two parts:
1.	Dimensionality reduction by UMAP/PCA (Principal component analysis).
2.	Decoder – identical to the AE decoder.

 The network architecture:
   <p align="center"><img width="650" height="240" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/PCA_UMAP_network.jpg"></p>

For each technique I embedded all the images in 16D or 2D.

 <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/pca_umap_results.jpg"></p>
 
  The only technique that reach similar results as the AE encoder was the PCA dimensionality reduction to 16D. 
 The 16 principal components of the PCA explained ~55% of the dataset variance. We can see that the 
 only feature the network didn't learn to reconstruct correctly was the tail of the rat. The 2D UMAP 
 latent space representation resembled the off the AE encoder latent space. The 
 difference in this case is expressed in the 6 port representation around the arena which in this case are less visible.

## How the objective of the learning affects the latent space neural Representation
 A direction tagger network was trained to predict the body direction of the rat. 
Its input is the single image of the rat and its output is the rat direction. 
The whole architecture of the direction tagger is identical to the AE encoder 
(except for the last linear readout). 
Its objective function is a regression to body direction (with the L2 metric)
  
 The network architecture:
  <p align="center"><img width="500" height="190" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/predict_body_angle_network.jpg"></p>

 After the direction tagger was trained, I used UMAP to reduce the dimensionality of the last hidden
  layer which is analogous to the latent space of the AE.
 I colored the data-points according to the body angle of the rat.
  <p align="center"><img width="375" height="400" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/2D_umap_body_angle.png"> </p>

 The above results suggest that the  learned embedding represents the circularity of the rat body 
 angle. The distance of the rat from the center of the arena, on the contrary,
  has no clear representation in this embedding.
[add figure]
The geometry of the 2D AE embedding (fig A) is significantly different than the direction tagger
 embedding (fig B). 
This can be explained by the difference of the objective functions: 
While the direction tagger was trained to predict only the body angle of the rat, 
the AE was trained to reconstruct the whole image, and thus embedded all its major
 features (rat body angle and distance from center, arena ports).
  <p align="center"><img width="700" height="350" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/predict_body_angle_vs_AE.jpg"> </p>
