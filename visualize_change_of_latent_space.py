from utils_local import setting_parameters
from utils_local import get_latent_space, get_visualize_latent_space_dim_reduction

def main():
    # ====== setting the HP param dict =======
    folder_dir, hp_parameters_dict = setting_parameters(use_folder_dir=True)  # todo mofule the code
    # ====== creating and saving the last latent space, from test mode model =======
    if hp_parameters_dict['run_all_umap_dim_reduction_options']:
        for option in ['last', 'first', 'every_epoch']:
            get_visualize_latent_space_dim_reduction(hp_parameters_dict, folder_dir, option)
    else:
        get_visualize_latent_space_dim_reduction(hp_parameters_dict, folder_dir,
                 hp_parameters_dict['umap_dim_reduction_fit_according_to_specific_epoch'])



if __name__ == "__main__":
    main()




