from utils_local import setting_parameters, loading_plus_preprocessing_data_with_labels, \
    test_model_predicted_position, plot_labels_vs_predictions_on_arena
import os
import torch
from torch import nn
from natsort import natsorted
import numpy as np

# ====== setting the HP param dict =======
folder_dir, hp_parameters_dict = setting_parameters(use_folder_dir=True) #todo mofule the code
# ====== loading the data and pre-processing it =========
dataloader = loading_plus_preprocessing_data_with_labels(hp_parameters_dict, mode='test')
encoder_mode = 'AE extract images'
# ====== load the model =======
last_saved_model = natsorted([model_name for model_name in os.listdir(hp_parameters_dict['dir_model'])])[-1]
model_path = os.path.join(hp_parameters_dict['dir_model'], last_saved_model)
model = torch.load(model_path)
model.eval()
criterion = nn.MSELoss()
# ====== run prediction ======
with torch.no_grad():
    val_loss, original_position, predicted_points = test_model_predicted_position(model, dataloader, criterion, folder_dir, 'test')
    val_loss_avr = val_loss / len(dataloader.dataset)
    print('Validation loss %.5f' % val_loss_avr)
    np.savez_compressed(os.path.join(folder_dir, 'original_position.npz'), original_position)
    np.savez_compressed(os.path.join(folder_dir, 'predicted_points.npz'), predicted_points)
# path = 'D:\Autoencoder\Predict body position\predict body position, same encoder network as the ae, taining all of the network'
# with np.load(os.path.join(path, 'original_position.npz')) as f:
#     original_position = f['arr_0']
# with np.load(os.path.join(path, 'predicted_points.npz')) as f:
#     predicted_points = f['arr_0']

# ====== plot a fake arena and than plot the position of the rat (predicted vs labels) as dotes ======
plot_labels_vs_predictions_on_arena(original_position.detach().cpu().numpy(), predicted_points.detach().cpu().numpy(), folder_dir)
# plot_labels_vs_predictions_on_arena(original_position, predicted_points, folder_dir)
