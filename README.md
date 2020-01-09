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

## In this git you can find:
- [Convolutional AutoEncoder](#convolutional-autoencoder) - short explanation on what I did and how I used it 
- [PCA/UMAP as encoder](#pcaumap-as-encoder) - short explanation on what I did and how I used it 
- [Predict rat body angle](#predict-rat-body-angle) - short explanation on what I did and how I used it 
- [Installation](#installation) instructions
- [Dataset requirement](#dataset-requirement)
- How to run the [Train and test modes](#train-and-test-modes) instructions
- [Results](#results) of my project

## Convolutional AutoEncoder
 The AE (AutoEncoder) network uses feature embeddings (AKA the latent representation) that were extracted from images by an encoder to reconstruct an image without it's noise.
 In the following network I utilized convolutional  and max-pooling layers, followed by a flattening and fully connected layer to encode the image in a reduced-dimensional space.
 For decoding, I used the inverse operations. I utilized fully connected layer followed by transposed convolutional layers to decode the image to it's original space.    

 The network architecture:
  <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/ae_network.jpg"></p>

The input of the network is 50X50 gray-scale* images (in my case I trained the network on 50X50X1 images of freely behaving rats)

*default settings, can be change by setting params in the AE network.

## PCA/UMAP as encoder
The network uses feature embeddings (AKA the latent representation) that were extracted from images by dim
 reduction techniques, Principal component analysis (PCA) or 
 Uniform Manifold Approximation and Projection (UMAP), to reconstruct (decode) an image without it's noise.
 For decoding, I used the same architecture as the AE decoder.

 The network architecture:
  <p align="center"><img width="700" height="250" src="https://github.com/doronharitan/autoencoder/blob/master/figuers/PCA_UMAP_network.jpg"></p>

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
  <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/predict_body_angle_network.jpg"></p>

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
As you would see in the results paragraph below(add link), the metadata file would be use to color the data points according to specific condition. for example the angle of the rat body (for more interesting details and results click here(add link)) 

The format of this file need to be npy. The array dimensions should be:  dataset_size X conditions (could be as many as you want). 
To control which condition you want the data points to be color according to change the 'color_datapoints_according_to_specific_condition_dict' dictionary which appear in the beginning of utils_local.py.     

##  Train and test modes
*Default args parameters to train and test modes are detailed below

### Train modes:  
- #### Convolutional AutoEncoder Train mode
```
python train.py   --train_data_dir dir_where_trained_data_is_saved    
```
- #### PCA/UMAP as encoder Train mode



### Test mode:
Testing the ability of the model to de-noise an image on the designated test data.

By default:
 1. Model checkpoints are saved in the 'model check points' directory using the following naming convention:
 model_<num_epoch>_epoch.pth.tar
 2. The model is saved every X (args.val_check_interval)* epochs

*default settings, can be change by setting params: args.val_check_interval

#### Default args parameters to train and test modes






explain how to run it 
What we wanted to achive 
what we chose to look on and why, 
what was the results.
add analysis woth diffrent colors 

what question it rises - 
1. regarding how the code changes with time, how it learns to map the featuers
2. regarding what happens if I will extract the feature using dim reduction teqnicques like umap and pca instead of the encoder will the network learn?
3. what happen if I will change the task will the new network learn the same map? 
and than add body prediction
add transfer and etc
 #######explain about the fc2 option
 ######add what is the requirments of the metadata format. what is the datapoints labels that I have
## Results 
The train model that was tested below was trained on a randomly chosen train/test set (from the 3 possible provided by UCF-101 dataset).
The model was trained and thus, tested on 55 classes from the 101 possible classes in UCF-101 dataset.
- _**Basic test mode**_:  The model reached a classification accuracy of **90.5%**.

##### PCA/UMAP as encoder results

##### predict rat boy angle results