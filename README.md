# Convolutional AutoEncoder for image noise reduction
PyTorch implementation of deep Convolutional AutoEncoder for denoising images.

## Convolutional AutoEncoder
 The AE (AutoEncoder) network uses feature embeddings (AKA the latent representation of the input) that were extracted by the encoder to decode the image without it's noise.
 In the following network I utilized convolutional  and max-pooling layers, followed by a flattening and fully connected layer to encode the image in a reduced-dimensional space.
 For decoding, I used the inverse operations. For this task, I utilized fully connected layer followed by transposed convolutional layers to decode the image to it's original space.    

 The network architecture:
  <p align="center"><img src="https://github.com/doronharitan/autoencoder/blob/master/figuers/ae_model.jpg"></p>

The input of the network is 50X50 gray-scale* images (in my case I trained the network on 50X50X1 images of freely behaving rats) 

*default settings can be change by setting other params in the AE network.

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

##  Train and test modes
*Default args parameters to train and test modes are detailed below

### Train mode
```
python train.py   --train_data_dir dir_where_trained_data_is_saved    
```


 
   
