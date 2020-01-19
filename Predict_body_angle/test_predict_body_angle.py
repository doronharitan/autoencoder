from utils_local import setting_parameters, \
    get_visualize_latent_space_dim_reduction, load_model, \
    loading_plus_preprocessing_data_with_labels, test_model_with_labels, plot_labels_vs_predictions_on_arena
from torch import nn
import torch

def main():
    # ====== setting the args and folder_dir =======
    folder_dir, args = setting_parameters(use_folder_dir=True, mode='test')
    # ====== loading the data and pre-processing it =========
    dataloader = loading_plus_preprocessing_data_with_labels(args)
    # ====== load the model =======
    model = load_model(args['checkpoint_to_load'], args)
    criterion = nn.MSELoss()
    # ====== run prediction ======
    with torch.no_grad():
        test_loss, ___, output_labels = test_model_with_labels(model, dataloader, criterion, 'test', None)
        test_loss_avr = test_loss / len(dataloader)
        print('Test loss %.5f' % test_loss_avr)
    get_visualize_latent_space_dim_reduction(args, folder_dir,
                'last',  images_for_plot=dataloader.dataset.tensors[0].squeeze(1).cpu().numpy(),
                                                     model=model, mode= args['dim_reduction_algo'])
    plot_labels_vs_predictions_on_arena(dataloader.dataset.tensors[1].detach().cpu().numpy(),
                                        output_labels.detach().cpu().numpy(), folder_dir)


if __name__ == "__main__":
    main()