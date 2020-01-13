# Objective of the learning affects neural representation
Originally this project was suppose to be only about PyTorch implementation of deep Convolutional AutoEncoder for 
de-noise images and data compression.
But, with time and curiosity, the project become more about the questions: 
 1. What is encoded in the AE latent space? 
 2. If we create the latent space that would be used as the input of the decoder by using dimension (dim) reduction techniques 
 (like UMAP or PCA), would the decoder be able to decode de-noise images from it?
   3. How the objective of the learning affects the latent space neural
    representation? in this case I compared the latent space representation 
    of AE vs direction tagger (predict rat body angle)

# Conclusions
(#add)
1. dim of the arena is learned (add video)

## In this git you can find:
- [Convolutional AutoEncoder](#convolutional-autoencoder) - short explanation on what I did and how I used it 
- [visualize change of latent space](#visualize-change-of-latent-space) - short explanation on what I did and how I used it 
- [PCA/UMAP as encoder](#pcaumap-as-encoder) - short explanation on what I did and how I used it 
- [Predict rat body angle](#predict-rat-body-angle) - short explanation on what I did and how I used it 
- [Installation](#installation) instructions
- [Dataset requirement](#dataset-requirement)
- How to run the [Train and test modes](#train-and-test-modes) instructions
- How to run the [visualize change of latent space run mode](#visualize-change-of-latent-space-mode) instructions 
- [Results](#results) of my project

## Convolutional AutoEncoder
 The AE (AutoEncoder) network uses feature embeddings (AKA the latent representation) that were extracted from images by an encoder to reconstruct an image without it's noise.
 In the following network I utilized convolutional  and max-pooling layers, followed by a flattening and fully connected layer to encode the image in a reduced-dimensional space.
 For decoding, I used the inverse operations. I utilized fully connected layer followed by transposed convolutional layers to decode the image to it's original space.    

 The network architecture:
  <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/ae_network.jpg"></p>

The input of the network is 50X50 gray-scale* images (in my case I trained the network on 50X50X1 images of freely behaving rats)

*default settings, can be change by setting params in the AE network.

## visualize change of latent space
In order to answer the first question "What is encoded in the AE latent space?", I built a script 
(called visualize_change_of_latent_space.py) 
which uses UMAP to reduce the dimensionality of the latent space to 2D (from 16D) and enables
 to visualize the latent space embeddings.
 
 The visualization make it easier to investigate the topological properties of the 
 latent space embeddings. Together with coloring the data-points according to specific condition
, for example the body angle of the rat, we can learn if the networks 
learns to code in the latent space specific elements from the environment or the rat composition in the image. 

The above script enables to fit a UMAP to the desired latent space and than transform a number of 
chosen latent spaces to this UMAP model. For example, This is used in order to visualise the change in 
the latent space representation with the training of the AE model. 
In this case we fit the UMAP to the last latent space saved in the training mode and transformed each 
latent space (saved form epoch zero till the end of the training) to this model. The result can be seen here (*add link) 

The options to fit the UMAP model to are:
1. 'last' - last saved latent space 
2. 'first' - first saved latent space (epoch 0)
3. 'alternative latent space' - meaning a latent space that you provide and is not necessarily from this run.
4. 'all epochs' - fit and transform every latent space to it self
4. 'All'

## PCA/UMAP as encoder
The network uses feature embeddings (AKA the latent representation) that were extracted from images by dim
 reduction techniques, Principal component analysis (PCA) or 
 Uniform Manifold Approximation and Projection (UMAP), to reconstruct (decode) an image without it's noise.
 For decoding, I used the same architecture as the AE decoder.

 The network architecture:
  <p align="center"><img width="650" height="230" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/PCA_UMAP_network.jpg"></p>

The input of the network is the same as for the AE network.
This network was used in order to address the 2nd question " If we create the latent space that would be used as the input of the decoder by using dimension (dim) reduction techniques 
 (like UMAP or PCA), would the decoder be able to decode de-noise images from it?". For results see 
 ["pca/umap as encoder results"](#pcaumap-as-encoder-results) 
 
 
## Predict rat body angle
A network which uses feature embeddings (AKA the latent representation) that were extracted from images
 by an encoder to predict what is the body angle of the rat in the image
 For extracting the images embeddings I used An encoder with the same architecture as the AE encoder.
  
 The network architecture:
  <p align="center"><img width="600" height="230" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/predict_body_angle_network.jpg"></p>

The input of the network is the same as for the AE network.
This network was used in order to address the 3rd question "How the objective of the learning affects the latent space neural
    representation?". For results see 
 ["predict rat boy angle results"](#predict-rat-boy-angle-results) 


## Installation
This implementation uses Python 3.7.4 and PyTorch.

All dependencies can be installed into a conda environment with the provided environment.yml file.
``` 
# ==== clone repository =====
git clone https://github.com/doronharitan/autoencoder.git
cd autoencoder

# ==== create conda env and activte it =====
conda env create -f environment.yml
conda activate autoencoder_env
```

##  Dataset requirement
#### Train/Test dataset: 
The dataset needs to be in npz/npy format. The dimensions of the array needs to be: dataset_size X image_width X image_height. 
In case your images are not grey-scale, the dimension of the array need to be: dataset_size X num_channels X image_width X image_height.

#### metadata file:
As you would see in the [results paragraph below](#results), the metadata file would be use: 1. To color the data points according to specific condition. for 
example the angle of the rat body (for more interesting details and results click here(add link)) 2. As the labels in the predicted body angle task 

The format of this file need to be npy. The array dimensions should be:  dataset_size X conditions (could be as many as you want). 
To control which condition you want the data points to be color according to change the 'color_datapoints_according_to_specific_condition_dict'
 dictionary which appear in the beginning of utils_local.py.     

The metadata file needs to be saved in the same directory as the train/test data.


##  Train and test modes
*Default args parameters to train and test modes are detailed below

### Train modes:  
- #### Convolutional AutoEncoder Train mode
```
python Autoencoder/train.py   --train_data_dir       dir_where_the_data_for_the_training_is_saved\
                              --file_name            name_of_train_data_file\
```
- #### PCA/UMAP as encoder Train mode
```
python Autoencoder/train_encoder_pca_umap.py   --train_data_dir       dir_where_trained_data_is_saved\
                                               --file_name            name_of_train_data_file\
                                               --meta_data_file_name  name_of_metadata_file\ 
                                               --dim_reduction_algo   'UMAP' or 'PCA'
```
- #### Predict rat body angle Train mode
```
add all condtion and possabilites***
```

### Test mode:
By default model checkpoints are saved in the 'model check points' directory using the following naming convention:
 model_<num_epoch>_epoch.pth.tar 
 
- #### Convolutional AutoEncoder Test mode Test mode
Testing the ability of the model to de-noise an image on the designated test data.
```
python Autoencoder/test.py    --train_data_dir       dir_where_the_data_for_the_test_is_saved\
                              --file_name            name_of_test_data_file\
                              --meta_data_file_name  name_of_metadata_file\
                              --checkpoint_to_load   name_of_checkpoint_to_load\
                              --checkpoint_path      path_of_the_checkpoint_to_load
```
- #### PCA/UMAP as encoder Test mode 
Testing the ability of the model to de-noise an image on the designated test data.

```
python Autoencoder/test_pca_umap_as_encoder.py  --train_data_dir       dir_where_the_data_for_the_test_is_saved\
                                                --file_name            name_of_test_data_file\
                                                --meta_data_file_name  name_of_metadata_file\
                                                --checkpoint_to_load   name_of_checkpoint_to_load\
                                                --checkpoint_path      path_of_the_checkpoint_to_load\
                                                --dim_reduction_algo   'UMAP' or 'PCA'
```

- #### Predict rat body angle Test mode
```
add all condtion and possabilites***
```

#### Default args parameters to train and test modes
```
--batch_size                        64 
--batch_size_latent_space           128         #batch size for the analysis of the latent space
--seed                              42
--epochs                            150
--split_size                        0.2         #set the size of the split between validation data and train data
--lr                                1e-3
--open_new_folder                   'True'      #open a new folder where all of the run data would be saved at 
--max_pixel_value                   255.0       #raw images max pixel value you want to scale the images according to
--latent_space_dim                  16          #The dim featuers you want to extract from the images
--save_model_checkpoints            True
--checkpoint_interval               5
--load_checkpoint                   False
--checkpoint_path                   ''          #The path of the checkpoint model we want to load
--checkpoint_to_load                ''          #The name of the model checkpoint we want to load
--save_latent_space                 True        #Should we save the latent space during the run? Would be use in the visualization script
--checkpoint_latent_space_interval  3           #Interval between saving latent_space checkpoints
--val_check_interval                5           #Interval between running validation test
```

## visualize change of latent space mode
```
python visualize_change_of_latent_space.py   --train_data_dir       dir_where_trained_data_is_saved\
                                             --file_name            name_of_train_data_file\
                                             --meta_data_file_name  name_of_metadata_file\                                           
                                             --dim_reduction_algo   'UMAP' or 'PCA' or ''
```

#### Default args parameters to visualize change of latent space mode
```
# used in the visualize_change_of_latent_space.py
--extract_latent_space                                True      #In the dim reduction analysis should we extract the latent space? 
--extract_latent_space_fc2                            True      #In the dim reduction analysis should we analize also the FC_2 (begining of the decoder) feature space? 
--analysis_latent_space_stop_index                    'All',    #enable an early stop of latent_space analysis in a case we dont want to plot all of the latent space saved during the training
--save_plots_or_only_create_movie                     False     #in the dim reduction visualization do we want to save each plot or do we want to create only the video?
--umap_dim_reduction_fit_according_to_specific_epoch  'last'    #according to which epoch to fit the umap? Options: every_epoch, first, last, fit to alternative latent space, All
--alternative_latent_space_to_fit_dir                 ''        #What is the dir of the alternative latent space we want to fit the data to
```


## Results 
The train model that was tested below was trained on 32K 50X50X1 gray-scale images of freely 
behaving rats in an arena.The structure of the arena includes 6 ports where the rat can get food from.
 Thus, images could capture a rat in the middle space of the arena (the image would not include
  a ports or walls of the area) and it could capture a rat in a port or in the edges of the arena
   (the image would include a wall of the area).
   
- ##### Convolutional AutoEncoder test results:
    A video showing a stack of sample images from the test dataset before the de-noise and after is seen below: 
  <p align="center"><img width="400" height="200" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/input_vs_output_image.gif"></p>
    From the comparision above we can learn that the Autoencoder was able to clean the noise from the images 

    I was curious to learn **what is encoded in the AE latent space**? 
or in other words, what are the features that the model learn to extract from the image, so only 
based on them the decoder can reconstruct the image but without it's noise. 

    To address this question, I used UMAP to reduce the dimensionality of my latent space (from 16D to 2D)
and than I plotted the results (for a reminder how I did it and which script I used [click here](#visualize-change-of-latent-space)).
To assess what specific elements from the image were learned and represented in the latent space I
 colored the data-points according to specific condition. The conditions I used are: 
 1. The body angle of the rat 2. distance from the arena center.
 The results can be seen here:
    <p align="center"><img width="350" height="300" vspace="100" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/body_angle_of_the_rat.jpg"> 
    <img width="330" height="300" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/dis_from_arena_center.jpg"></p>
 
 The above results were surprising, they show that the latent space that
  was learned with training represented the dimensionality of the arena. 
  In the plot above We can actually see an arena (circle) with 6 ports spaced evenly around it. 
  This result is strengthened when we consider also the coloring of the data-points, which shows that 
  the topological shape that was learned is arrange according to the distance of the rat from the 
  center of the arena and by the body angle of the rat.  
  When you consider the fact that input image could be from any location in the arena, its quiet amazing 
  that the network learned to mapped it to the real arena shape and used this element as an important 
  feature to encode the image data.
  
#### PCA/UMAP as encoder results
In the autoencoder we extract the features embedding using learned convolotinal layers. Meaning
the network learns to extract the features that are important for the task, in this case, the 
feature that would enable the reconstructions of the image without it's noise. 
What would happen if I would extracte the features embedding using dim reduction techniques,
 so the only learned part in our network would be the decoder? would the decoder be able to learn
  how to decode the de-noise images from this features embedding?
  
 To test the above I used PCA and UMAP to reduce the dim of the input images and than passed it
  to a decoder with the same architecture as the AE one.
   For each technique I extracted 16D (The latent space dim extracted by the AE encoder)
    and 2D features embedding.
 
 <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/pca_umap_results.jpg"></p>

 The only technique that reach similar results as the AE encoder was the PCA dim reduction to 16D. 
 The PCA feature embedding in this case explained ~55% of the dataset variance. We can see that the 
 only feature the network didn't learn to reconstruct correctly was the tail of the rat. The 2D UMAP 
 latent space representation also looked similar to the one we get from the ae encoder latent space. The 
 difference in this case  is express in the 6 port representation around the arena which in this case are less clear and 
 less structured.
 

#### predict rat boy angle results
Another interesting question that rose during this study was "How the objective of the learning
 affects the latent space neural representation?". To question this I change the objective of the
  AE encoder. So, The aim of the network would be to predict the body angle of the rat 
  (for more details [click here](#predict-rat-body-angle)).
  
  After the network was train, I used UMAP to reduce the dimensionality of the latent space 
  (from 16D to 2D) and than I plotted the results (for a reminder how I did it and which script I used [click here](#visualize-change-of-latent-space)).
 I colored the data-points according to the body angle of the rat.
  <p align="center"><img width="400" height="350" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/2D_umap_body_angle.png"> </p>

 The above results show that the latent space that
  was learned with training represent the circularity of tha angle, Which when we think about it is not suppressing.
  This results, indicates that the objective of the learning
 affects the latent space neural representation even if the
  architecture of the network is the same. further research is needed in order to conform it. For more details see [future work](#future-work)
  
 

#future work
See of rhe FC@ develops with 2,
see what are the features eacg convtranspose is responsibale for
transfer learning 