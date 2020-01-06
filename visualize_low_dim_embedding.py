from utils_visualization import *
from utils_local import setting_parameters

def main():
    # ====== setting the HP param dict =======
    folder_dir, hp_parameters_dict, device = setting_parameters(use_folder_dir=False)  # todo mofule the code
    # ====== get the latent space of the AE ======
    if hp_parameters_dict['extract_latent_space'] and hp_parameters_dict['get_latent_space_method'] != 'UMAP':
        get_latent_space(hp_parameters_dict, folder_dir)
    # ====== open the latent_space npz file and run dim reduction=======
    if hp_parameters_dict['run_dim_reduction']:
        run_dim_reduction(folder_dir, hp_parameters_dict)
    # ====== visualize low dim embedding ======
    if hp_parameters_dict['run_visualization_on_dim_reduction']:
        run_visualization_on_dim_reduction(hp_parameters_dict, folder_dir, color_sample_according_to_dict)


if __name__ == "__main__":
    main()
