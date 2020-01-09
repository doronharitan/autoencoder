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
 For extracting the images embeddings I used:
  1. An encoder with the same architecture as the AE encoder.
  2. pre-trained AE encoder (trained on the de-noise task) with different depth of unfreezing layers 
  
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
 
- #### Convolutional AutoEncoder Test mode and PCA/UMAP as encoder Test mode
Testing the ability of the model to de-noise an image on the designated test data.
```
python Autoencoder/test.py    --train_data_dir       dir_where_the_data_for_the_test_is_saved\
                              --file_name            name_of_test_data_file\
                              --meta_data_file_name  name_of_metadata_file\
                              --checkpoint_to_load   name_of_checkpoint_to_load\
                              --checkpoint_path      path_of_the_checkpoint_to_load
                               #if we want to test the UMAP/PCA as encoder please add the following arg:
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
The train model that was tested below was trained on a randomly chosen train/test set (from the 3 possible provided by UCF-101 dataset).
The model was trained and thus, tested on 55 classes from the 101 possible classes in UCF-101 dataset.
- _**Basic test mode**_:  The model reached a classification accuracy of **90.5%**.

##### PCA/UMAP as encoder results

##### predict rat boy angle results