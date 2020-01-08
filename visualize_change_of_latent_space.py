from utils_local import setting_parameters, get_visualize_latent_space_dim_reduction

def main():
    # ====== setting the HP param dict =======
    folder_dir, args = setting_parameters(use_folder_dir=True)
    # ====== creating and saving the last latent space, from test mode model =======
    if args['umap_dim_reduction_fit_according_to_specific_epoch'] == 'All':
        for fit_umap_according_to_epoch in ['last', 'first', 'every_epoch']:
            get_visualize_latent_space_dim_reduction(args, folder_dir, fit_umap_according_to_epoch)
    else:
        get_visualize_latent_space_dim_reduction(args, folder_dir,
                 args['umap_dim_reduction_fit_according_to_specific_epoch'])



if __name__ == "__main__":
    main()




