import argparse


parser = argparse.ArgumentParser(description='Convolutional AutoEncoder for image noise reduction')
parser.add_argument('--run_goal', default='', type=str,
                    help='What is the goal of the run, Would be saved in the setting information for each run')
parser.add_argument('--train_data_dir', default=r'D:\Autoencoder\train data',
                    type=str, help='The directory of the train/test data')
parser.add_argument('--meta_data_file_name', default='mdata_for_mami.npy',
                    type=str, help='The name of the meta data file (located in the train data dir)')
parser.add_argument('--file_name', default='ims_for_doron.npz', type=str,
                    help='The name of the file containing the data')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--batch_size_latent_space', default=128, type=int,
                    help='Batch size for the analysis of the latent space')
parser.add_argument('--seed', default=42, type=int,
                    help='initializes the pseudorandom number generator on the same number (default:42)')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs')
parser.add_argument('--split_size', default=0.2, type=int, help='set the size of the split size between validation '
                                                                'data and train data')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate (default:5e-4')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project '
                         'dir, if debug the info would be saved in debug folder(default:True). options: "True", "False"'
                         'or "debug"')
parser.add_argument('--max_pixel_value', default=255.0, type=float,
                    help='max pixel value, would be used for the rescaling of the images')
parser.add_argument('--latent_space_dim', default=16, type=int, help='For AE: The dim of the encoder FC output '
                                                                     'When we use Dim reduction to encode the images '
                                                                     'this param would contol the low dim the algo will '
                                                                     'reduce to (default:16)')
parser.add_argument('--save_model_checkpoints', default=True, type=bool,
                    help='should we save the model checkpoints during the run')
parser.add_argument('--checkpoint_interval', default=5, type=int, help='Interval between saving model checkpoints')
parser.add_argument('--save_latent_space', default=True, type=bool,
                    help='Should we save the latent space during the run?')
parser.add_argument('--checkpoint_path', default=r'C:\Users\Doron\PycharmProjects\autoencoder\20200119-132746\model check points',
                    type=str, help='Optional path to checkpoint model')
parser.add_argument('--checkpoint_to_load', default='model_25_epoch.pth.tar',
                    type=str, help='The name of the model we want to load')
parser.add_argument('--checkpoint_latent_space_interval', default=3, type=int,
                    help='Interval between saving latent_space checkpoints')
parser.add_argument('--val_check_interval', default=5, type=int, help='Interval between running validation test')
parser.add_argument('--load_checkpoint', default=False, type=bool,
                    help='Loading a checkpoint and continue training with it')
parser.add_argument('--dim_reduction_algo', default='', type=str,
                    help='The algorithm that is used for the latent space dim reduction in the '
                         'train_encoder_pca_umap.py script. options: UMAP or PCA')
parser.add_argument('--extract_latent_space', default=True, type=bool,
                    help='In the analysis should we extract the latent space?')
parser.add_argument('--extract_latent_space_fc2', default=False, type=bool,
                    help='In the analysis should we extract the latent space of the 2nd FC layer'
                         ' (the beginning of the decoder?')
parser.add_argument('--umap_dim_reduction_fit_according_to_specific_epoch', default='last', type=str,
                    help='according to which epoch to fit the umap? '
                         'Options: every_epoch, first, last, fit to alternative latent space, All')
parser.add_argument('--alternative_latent_space_to_fit_dir',
                    default=r'./SOTA - 16D, alex data, save latent space for batch/Latent_space_arrays_fc1', type=str,
                    help='What is the dir of the alternative latent space')
parser.add_argument('--analysis_latent_space_stop_index',
                    default='All', help='enable an early stop of'
                                        ' latent_space analysis in a case we dont want to plot all of the '
                                        'latent space saved')
parser.add_argument('--save_plots_or_only_create_movie',
                    default=False,type=bool, help='in the dim reduction analysis do we want '
                                                  'to save each plot or do we want to create only the video?')



