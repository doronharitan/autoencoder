from utils_local import setting_parameters, loading_plus_preprocessing_data
import os
from torchvision.utils import save_image


# ====== setting the HP param dict =======
folder_dir, hp_parameters_dict, device = setting_parameters(use_folder_dir=False) #todo mofule the code
# ====== loading the data and pre-processing it =========
dataloader_dict = loading_plus_preprocessing_data(hp_parameters_dict, device, visualize_low_dim=True)

for batch_num, local_batch in enumerate(dataloader_dict):
    save_image(local_batch, os.path.join(folder_dir, './Images/%d_batch_num.png' % (batch_num)))