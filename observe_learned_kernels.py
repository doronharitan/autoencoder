from utils_local import setting_parameters, save_latent_space_to_file
import os
import math
import torch
import matplotlib.pyplot as plt


# ====== setting the HP param dict =======
folder_dir, hp_parameters_dict, device = setting_parameters()  #todo fix the folder dir
# ====== load the model =======
model_path = os.path.join(hp_parameters_dict['dir_model'], hp_parameters_dict['file_name_model'])
model = torch.load(model_path)
model.eval()

# getting the weight tensor data for 1st layer
kernel_tensor = model.encoder[0].weight.data.detach().cpu().squeeze(1) #todo cheack what is the model.featuers
# ====== Plotting the kernel ======
num_subplots = kernel_tensor.shape[0]
n_cols = 8
n_rows = math.ceil(num_subplots/n_cols)
fig, ax = plt.subplots(ncols= n_cols, nrows=n_rows, figsize=(10, 10))
for row in range(n_rows):
    for col in range(n_cols):
        ax[row, col].imshow(kernel_tensor[col + (row * n_cols)], cmap='gray')
plt.savefig(os.path.join(folder_dir, 'first filter.png'))

# ====== plot the feature map pattern ======


