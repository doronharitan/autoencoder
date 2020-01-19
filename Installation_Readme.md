## Table of content:
- [Installation](#installation)
- [Dataset requirement](#dataset-requirement)
- [How to run the train and test script](#train-and-test-modes)
- [Default args parameters to train and test modes](#default-args-parameters-to-train-and-test-modes)
- [How to run the script that visualize the latent space](#visualize change of latent space)

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
python Predict_body_angle/train predict body angle.py   --train_data_dir       dir_where_trained_data_is_saved\
                                                        --file_name            name_of_train_data_file\
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
python Predict_body_angle/test_predict_body_angle.py   --train_data_dir       dir_where_the_data_for_the_test_is_saved\
                                                       --file_name            name_of_test_data_file\
                                                       --meta_data_file_name  name_of_metadata_file\
                                                       --checkpoint_to_load   name_of_checkpoint_to_load\
                                                       --checkpoint_path      path_of_the_checkpoint_to_load
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

## visualize change of latent space
visualize_change_of_latent_space.py script uses UMAP to reduce the dimensionality
 of the matrix giving to 2D (usually from 16D)
 
The script fit a UMAP to the desired matrix and than transform a number of 
chosen matrix to this UMAP model. 

The options to fit the UMAP model to are:
1. 'last' - last saved latent space 
2. 'first' - first saved latent space (epoch 0)
3. 'alternative latent space' - meaning a latent space that you provide and is not necessarily from this run.
4. 'all epochs' - fit and transform every latent space to it self
4. 'All'
