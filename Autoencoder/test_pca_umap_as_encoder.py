from utils_local import setting_parameters, \
    save_latent_space_to_file, get_visualize_latent_space_dim_reduction, create_folder_if_needed, load_model,\
    get_latent_space_pca_umap_test_mode, test_model_with_labels
import os
from torch import nn
import torch

def main():
    # ====== setting the args and folder_dir =======
    folder_dir, args = setting_parameters(use_folder_dir=True, mode='test')
    # ====== loading the data and pre-processing it =========
    dataloader = get_latent_space_pca_umap_test_mode(args, folder_dir)
    save_latent_space_to_file(dataloader.dataset.tensors[0].cpu().numpy(), os.path.join(folder_dir, 'Latent_space_arrays_fc1'),
                              args['dim_reduction_algo'], method=args['dim_reduction_algo'])
    # ====== load the model =======
    model = load_model(args['checkpoint_to_load'], args)
    criterion = nn.MSELoss()
    # ====== run prediction ======
    save_images_path = os.path.join(folder_dir, 'Images')
    create_folder_if_needed(save_images_path)
    with torch.no_grad():
        test_loss, original_images, output_images = test_model_with_labels(model, dataloader, criterion, 'test', save_images_path)
        test_loss_avr = test_loss / len(dataloader.dataset)
        print('Test loss %.5f' % test_loss_avr)
    # ====== plot results analysis ======
    # ====== the following would be plot: 1. movies of all the images 2. the change of the umap according to the set fit ======
    images_dict = {'Original_images': original_images, 'AE_output_images' : output_images,
                   'Original_and_AE_output_images': torch.cat([original_images, output_images], axis=-1)}
    image_arrays_names = list(images_dict.keys())
    image_arrays_names.append('Umap')
    for image_arrays_name in image_arrays_names:
        save_video_path = os.path.join(folder_dir, 'Videos')
        create_folder_if_needed(save_video_path)
        if image_arrays_name != 'Umap':
            images_dict[image_arrays_name] = images_dict[image_arrays_name].squeeze(1).numpy()
            # create_video(os.path.join(save_video_path, '{}.mp4'.format(image_arrays_name)),
            #              images_dict[image_arrays_name], fps=60, rgb=False)
        else:
            get_visualize_latent_space_dim_reduction(args, folder_dir,
                'last',  images_for_plot=images_dict['AE_output_images'],
                                                     model=model, mode= args['dim_reduction_algo'])

if __name__ == "__main__":
    main()