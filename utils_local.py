import json
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import time
import umap
from sklearn.decomposition import PCA
import joblib
import os
from natsort import natsorted
import numpy as np
import math
from tqdm import tqdm
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from args import parser
from models.autoencoder_model import Autoencoder
from models.pca_umap_decoder_model import DimReductionDecoder

color_datapoints_according_to_specific_condition_dict = {7: 'Angle of the body',
                                  10: 'distance from the center of the arena',
                                  11: 'polar representation of rat location'}

# color_datapoints_according_to_specific_all_condition_dict = {6: 'Angle of the head', 7: 'Angle of the body',
#                                       10: 'distance from the center of the arena',
#                                       11: 'polar representation of rat location',
#                                       12: 'Did the rat inserted noise into port'}


specific_images_to_plot = [154, 85, 226]

def setting_parameters(use_folder_dir=False, mode=None):
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_folder_dir:
        if mode is None:
            folder_dir = os.path.normpath(args.checkpoint_path + os.sep + os.pardir)
        else:
            folder_dir = os.path.join(os.path.normpath(args.checkpoint_path + os.sep + os.pardir), mode)
            create_folder_if_needed(folder_dir)
    else:
        if args.open_new_folder != 'False':
            folder_dir = open_new_folder(args.open_new_folder)
        else:
            folder_dir = os.getcwd()
    args_dict = vars(args)
    save_setting_info(args_dict, device, folder_dir)
    print('The HP parameters that were used in this run are:', args)
    print('The code run on:', device)
    print('The folder all of the data is saved on is:', folder_dir)
    args_dict['device'] = device
    return folder_dir, args_dict


def open_new_folder(open_folder_status):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug'
    folder_dir = os.path.join(os.getcwd(), folder_name)
    create_folder_if_needed(folder_dir)
    return folder_dir


def save_setting_info(hp_parameters, device, folder_dir):
    setting_file_name = os.path.join(folder_dir, 'setting_info.txt')
    i = 0
    while os.path.exists(setting_file_name):
        setting_file_name = os.path.join(folder_dir, 'setting_info_{}.txt'.format(i))
        i += 1
    with open(setting_file_name, 'w') as f:
        json.dump(hp_parameters, f, indent=2)
        json.dump('\n' + str(device), f)


def create_folder_if_needed(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def loading_plus_preprocessing_data(args, split_to_train_val=False):
    images = load_plus_scale_images(args)
    if split_to_train_val:
        images_train, images_val = train_test_split(images, test_size=args['split_size'],
                                                random_state=args['seed'])
        dataloader_dict = {
                index: DataLoader(torch.tensor(data, device=args['device']).float().unsqueeze(1),
                                  batch_size=args['batch_size'],
                                  shuffle=True)
                for index, data in enumerate([images_train, images_val])}
    else:
            dataset = torch.tensor(images, device=args['device']).float().unsqueeze(1)
            dataloader_dict = DataLoader(dataset, batch_size=args['batch_size_latent_space'],
                                    shuffle=False)
    return dataloader_dict


def load_plus_scale_images(args):
    if '.npy' in  args['file_name']:
        images = np.load(os.path.join(args['train_data_dir'], args['file_name']))
    else:
        with np.load(os.path.join(args['train_data_dir'], args['file_name'])) as training_data:
            # images = training_data['images']
            images = training_data['arr_0']
    # ====== scale images ========
    images = images / args['max_pixel_value']
    return images


def train_model(model, dataloader_dict, optimizer, criterion, epoch, save_latent_space, folder_dir, checkpoint_latent_space_interval):
    train_loss, loss_reg = 0.0, 0.0
    train_dataloader = dataloader_dict[0]
    if save_latent_space:
        save_latent_space_every_batches = len(train_dataloader) // checkpoint_latent_space_interval
    for num_batch, local_batch in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()                   # clear the gradients of all optimized variables
        loss, __ = prediction_step(model, local_batch, criterion)
        train_loss += loss.item()
        loss.backward()                         # compute the gradients
        optimizer.step()                        # update the weights with the gradients
        if save_latent_space and num_batch % save_latent_space_every_batches == 0:
            get_latent_space(None, folder_dir, model=model, epoch=epoch,
                             batch=num_batch, dataloader_all_data=dataloader_dict['all_data'])
    return train_loss


def test_model(model, test_dataloader, criterion, epoch, save_images_path):
    val_loss = 0.0
    model.eval()
    # ===== save output images and original images if the test mode is true ======
    if epoch == 'test':
        output_images, original_images = [], []
    for batch_num, local_images_batch in enumerate(test_dataloader):
        loss, output = prediction_step(model, local_images_batch, criterion, mode='test')
        if epoch == 'test':
            output_images += [output.detach().cpu()]
            original_images += [local_images_batch.detach().cpu()]
        val_loss += loss.item()
    # ====== Visualize the output images ======
    if criterion is None:
        pass
    else:
        visualize_output_images(local_images_batch, output, save_images_path, epoch)
    # ====== if test mode return images and output else return only loss======
    if epoch == 'test':
        return val_loss, torch.cat(original_images), torch.cat(output_images)
    else:
        return val_loss


def visualize_output_images(local_images_batch, output, save_images_path, epoch):
    output_img_tensor_to_plot = torch.cat((local_images_batch[:8], output[:8]))
    save_image(output_img_tensor_to_plot, os.path.join(save_images_path, '{}_epoch.png'.format(epoch)))


def prediction_step(model, dataset, criterion, labels=None, mode='train'):
    if mode != 'train':
        with torch.no_grad():
            outputs = model(dataset)
    else:
        outputs = model(dataset)
    # ====== Calculate loss =======
    if criterion is None:
        loss = torch.tensor(0)
    elif labels is None:
        loss = criterion(outputs, dataset)
    else:
        loss = criterion(outputs, labels)
    return loss, outputs.detach()


def train_model_with_labels(model, dataloader_dict, optimizer, criterion, epoch, save_latent_space, folder_dir,
                            checkpoint_latent_space_interval):
    train_loss, loss_reg = 0.0, 0.0
    train_dataloader = dataloader_dict[0]
    if save_latent_space:
        save_latent_space_every_batches = len(train_dataloader) // checkpoint_latent_space_interval
    for num_batch, (local_images_batch, local_labels_batch) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()                   # clear the gradients of all optimized variables
        loss, __ = prediction_step(model, local_images_batch, criterion, labels=local_labels_batch)
        train_loss += loss.item()
        loss.backward()                         # compute the gradients
        optimizer.step()                        # update the weights with the gradients
        if save_latent_space and num_batch % save_latent_space_every_batches == 0:
            get_latent_space(None, folder_dir, model=model, epoch=epoch,
                             batch=num_batch, dataloader_all_data=dataloader_dict['all_data'])
    return train_loss


def test_model_with_labels(model, test_dataloader, criterion, epoch, save_images_path):
    val_loss = 0.0
    model.eval()
    # ===== save output images and original images if the test mode is true ======
    for batch_num, (local_images_batch, local_labels_batch) in enumerate(test_dataloader):
        loss, output = prediction_step(model, local_images_batch, criterion, labels=local_labels_batch ,mode='test')
        val_loss += loss.item()
    # ====== Visualize the output images ======
    if local_images_batch.shape[1] > 1:
        output_img_tensor_to_plot = torch.cat((local_labels_batch[:8], output[:8]))
    else:
        output_img_tensor_to_plot = torch.cat((local_images_batch[:8], output[:8]))
    save_image(output_img_tensor_to_plot, os.path.join(save_images_path, '{}_epoch.png'.format(epoch)))
    return val_loss


def get_latent_space(args, folder_dir, fit_umap_according_to_epoch=None, fc2_mode=False, model=None,
                        epoch='test', batch=None, dataloader_all_data=None):
    save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays_{}'. format('fc2' if fc2_mode else 'fc1'))
    create_folder_if_needed(save_latent_space_folder)
    dataloader_all_data = set_dataloader(dataloader_all_data, args, folder_dir)  # todo extract to a new function the pca option
    #  ====== load the relevant model and predict the latent space =======
    if model is None:
        # sort the model saved according to checkpoints
        model_list = set_model_list(args['checkpoint_path'], fit_umap_according_to_epoch)
        for model_name in model_list:
            model = load_model(model_name, args)
            if args['dim_reduction_algo'] == 'PCA' and fc2_mode == False:
                epoch = 'PCA'
            else:
                epoch = model_name.split('.pth.tar')[0].split('model_')[1]
            extract_latent_space_prediction(dataloader_all_data,
                                        save_latent_space_folder, fc2_mode, epoch, batch, model)
    #  ====== use the forwarded model to get the latent space =======
    elif model is not None:
        extract_latent_space_prediction(dataloader_all_data,
                                    save_latent_space_folder, fc2_mode, epoch, batch, model)


def set_dataloader(dataloader_all_data, args, folder_dir):
    if args['dim_reduction_algo'] == 'PCA':
        dataset = load_npz_file(os.path.join(folder_dir, 'Latent_space_arrays_fc1'), 'latent_space_PCA.npz', mode='transform')
        dataloader = DataLoader(torch.tensor(dataset).to(args['device']),
                                batch_size=args['batch_size_latent_space'], shuffle=False)
    elif dataloader_all_data is None:
        dataloader = loading_plus_preprocessing_data(args)
    else:
        dataloader = dataloader_all_data
    return dataloader


def set_model_list(checkpoint_path, fit_umap_according_to_epoch):
    model_list = natsorted([model_name for model_name in os.listdir(checkpoint_path)])
    if fit_umap_according_to_epoch == 'first':
        model_list = [model_list[0]]
    elif fit_umap_according_to_epoch != 'every_epoch':
        model_list = [model_list[-1]]
    return model_list


def load_model(model_name, args):
    # ==== set the model according latnet space encoder options - 'PCA', 'UMAP' or 'AE encoder'========
    checkpoint = torch.load(os.path.join(args['checkpoint_path'], model_name))
    # if checkpoint['model_name'] != 'Autoencoder':
    if args['dim_reduction_algo'] == 'PCA':
        model = DimReductionDecoder(args['pca_umap_dim_reduction']).to(args['device'])
    else:
        model = Autoencoder(args['latent_space_dim']).to(args['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def extract_latent_space_prediction(dataloader, save_latent_space_folder,
                                fc2_mode, epoch, batch, model):
    # ====== check if the latent space file exists else create them =======
    if os.path.exists(os.path.join(save_latent_space_folder, 'latent_space_{}.npz'.format(epoch))):
        pass
    else:
        # ====== run forward pass till the latent_space ======
        # ====== move model to eval mode with no grads and predict the latent space ======
        model.eval()
        latent_space_arrays = []
        with torch.no_grad():
            for local_images_batch in dataloader:
                latent_space = model.forward_latent_space(local_images_batch, fc2_mode)
                latent_space_arrays += [latent_space.detach().cpu()]
            save_latent_space_to_file(latent_space_arrays, save_latent_space_folder, epoch, batch=batch)


def save_latent_space_to_file(latent_space, save_latent_space_folder, epoch='last', method='AE_model', batch=None):
    if batch is None:
        file_path = os.path.join(save_latent_space_folder, 'latent_space_{}.npz'.format(epoch))
    else:
        file_path = os.path.join(save_latent_space_folder, 'latent_space_{}_epoch_{}_batch.npz'.format(epoch,
                                                                                                       batch))  # todo switch to the new format '{} {}'.format('one', 'two')
    if method == 'PCA' or method == 'UMAP':
        latent_space_array = latent_space
    else:
        latent_space_array = torch.cat(latent_space).numpy()
    np.savez_compressed(file_path, latent_space_array)


def save_loss_info_into_a_file(train_loss, val_loss, folder_dir, epoch, lr):
    file_name = os.path.join(folder_dir, 'loss_per_epoch.txt')
    with open(file_name, 'a+') as f:
        f.write('{} Epoch: Train Loss {:.6f}, Validation loss {:.6f},  lr %.8f\n'
                .format(epoch, train_loss, val_loss, lr))


def create_video(save_video_path, plot_arrays, fps=5, rgb=True):
    if 'Original_and_AE_output_images' in save_video_path:
        h_fig = plt.figure(figsize=(8, 4))
    else:
        h_fig = plt.figure(figsize=(4, 4))  # todo change back to 8,8?
    h_ax = h_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    h_ax.set_axis_off()
    if rgb:
        h_im = h_ax.matshow(plot_arrays[0])
    else:
        h_im = h_ax.matshow(plot_arrays[0], cmap='gray')
    h_im.set_interpolation('none')
    h_ax.set_aspect('equal')
    n_frames = len(plot_arrays)
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Rat movie {}'.format(save_video_path.split('.')[0]), artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with tqdm(total=n_frames) as pbar:
        with writer.saving(h_fig, save_video_path, dpi=300):
            for i in range(n_frames):
                h_im.set_array(plot_arrays[i])
                writer.grab_frame()
                pbar.update(1)


def get_visualize_latent_space_dim_reduction(args, folder_dir, fit_umap_according_to_epoch, images_for_plot=None, model=None, mode=None):
    fc1_mode, fc2_mode = args['extract_latent_space'], args['extract_latent_space_fc2']
    for index, fc_mode in enumerate([fc1_mode, fc2_mode]):
        which_fc = 'fc{}'.format(index + 1)
        if which_fc == 'fc2' and (
            fit_umap_according_to_epoch == 'every_epoch' or fit_umap_according_to_epoch == 'first'):
            print('Saving umap models is very memory expansive,'
                  'Thus, I canceled the option for fitting fc2 output to a umap according to every epoch'
                  'and according to first epoch. If you do want to use it cancel the relevant if')
            pass
        elif fc_mode:
            fc_2_mode_to_pass = False if which_fc == 'fc1' else True
            # ====== if the latent space file exists the code will not create a new one, saves time =======
            get_latent_space(args, folder_dir, fit_umap_according_to_epoch=fit_umap_according_to_epoch,
                             fc2_mode=fc_2_mode_to_pass, model=model)
            # ====== set the path from where we should read the latent space files =======
            latent_space_dir = os.path.join(folder_dir, 'Latent_space_arrays_{}'.format(which_fc))
            latent_space_files = natsorted([file for file in os.listdir(latent_space_dir)])
            if args is not None and args['analysis_latent_space_stop_index'] != 'All':
                latent_space_files = latent_space_files[: args['analysis_latent_space_stop_index']]
            # ====== if mode equal to test move the alternative latent space dir to be the one
            # that was extracted last by the model in the training mode=====
            if mode == 'test':
                args['alternative_latent_space_to_fit_dir'] = os.path.join(
                    os.path.normpath(args['checkpoint_path'] + os.sep +
                                     os.pardir), 'Latent_space_arrays_{}'.format(which_fc))
            # ====== fit and transform 2D umap to the specific latent space ======
            umap_embedding = fit_transform_umap_according_to_specific_requirement\
                (args, fit_umap_according_to_epoch, latent_space_files, folder_dir, which_fc, latent_space_dir)
            # ===== visualize the umap embeddings ======
            visualize_umap_embeddings(args, folder_dir, which_fc,
                                          umap_embedding, fit_umap_according_to_epoch, images_for_plot)
        else:
            continue


def fit_transform_umap_according_to_specific_requirement(args, fit_umap_according_to_epoch, latent_space_files,
                                               folder_dir, which_fc, latent_space_dir):
    if fit_umap_according_to_epoch != 'every_epoch':
        latent_space_to_fit = set_which_latent_space_to_fit(fit_umap_according_to_epoch,
                                                            latent_space_files, args['alternative_latent_space_to_fit_dir'])
        # ====== fit 2D umap to the specific latent space ======
        run_dim_reduction(folder_dir, latent_space_dir, which_fc,
                          fit_umap_according_to_epoch=fit_umap_according_to_epoch,
                          only_fit_umap=True,
                          latent_space_file_name=latent_space_to_fit)
        # ===== transform the relevant latent space to the umap fit ======
        umap_embedding = extract_dim_reduction_embeddings\
            (folder_dir, args, latent_space_dir, which_fc, fit_umap_according_to_epoch, latent_space_files)
    elif fit_umap_according_to_epoch == 'every_epoch':
        # ===== fit and transform the relevant latent space to the umap dim reduction  ======
        umap_embedding = extract_dim_reduction_embeddings \
            (folder_dir, args, latent_space_dir, which_fc, fit_umap_according_to_epoch, latent_space_files,
             only_fit_umap=True)
    return umap_embedding


def set_which_latent_space_to_fit(fit_umap_according_to_epoch, latent_space_files, alternative_latent_space_to_fit_dir):
    if fit_umap_according_to_epoch == 'first':
        latent_space_to_fit = latent_space_files[0]
    elif fit_umap_according_to_epoch == 'last':
        latent_space_to_fit = latent_space_files[-1]
    elif fit_umap_according_to_epoch == 'fit to alternative latent space':
        last_latent_space = \
            natsorted([model_name for model_name in os.listdir(alternative_latent_space_to_fit_dir)])[-1]
        latent_space_to_fit = os.path.join(alternative_latent_space_to_fit_dir, last_latent_space)
    return latent_space_to_fit


def visualize_umap_embeddings(args, folder_dir, which_fc,
                              umap_embedding, fit_umap_according_to_epoch, images_for_plot):
    meta_data = np.load(os.path.join(args['train_data_dir'],
                                     args['meta_data_file_name']))
    save_plots_folder_dir = os.path.join(folder_dir, 'UMAP Plots {}'.format(which_fc))
    if args['save_plots_or_only_create_movie']:
        create_folder_if_needed(save_plots_folder_dir)
    save_video_path = os.path.join(folder_dir, 'Videos {}'.format(which_fc))
    create_folder_if_needed(save_video_path)
    images_to_plot_dim_reduction = extract_dim_reduction_images_to_plot(umap_embedding)
    if images_for_plot is None:
        images_to_plot = extract_images_to_plot(save_video_path, args, folder_dir)
    else:
        images_to_plot = np.stack([images_for_plot[specific_images_to_plot[0]],
                                   images_for_plot[specific_images_to_plot[1]], images_for_plot[specific_images_to_plot[2]]])
    axis_limits = calculate_max_min_axis(umap_embedding) #set that it would work for 3d as well
    dots_to_plot_line = extract_dots_for_line(images_to_plot_dim_reduction, axis_limits)
    for index, color_type in color_datapoints_according_to_specific_condition_dict.items():
        plot_name = '2D UMAP, according to {} fitting, color {}'.format(fit_umap_according_to_epoch,
                                                                        color_type)
        # ===== if we have only one frame to plot change the format to figure ====
        file_type = '.avi' if len(umap_embedding) > 1 else '.png'
        save_video_path_dim = os.path.join(save_video_path, plot_name + file_type)
        if os.path.exists(save_video_path_dim):
            pass
        else:
            color_array = meta_data[:, index]
            cmap = set_cmap(color_type)
            # ====== if fit to alternative latent space load the umap of the alternative latent space to be plot on =======
            if fit_umap_according_to_epoch == 'fit to alternative latent space':
                umap_embedding = load_npz_file(os.path.join(folder_dir,'Umap_embedding_{}'.format(which_fc)),
                                   '2D_UMAP_alternative_embedding_fit.npz',
                                   mode='fit to alternative latent space')
                umap_embedding = [umap_embedding]
            plot_umap_embedding(axis_limits, umap_embedding, color_array, cmap, dots_to_plot_line,
                                images_to_plot, images_to_plot_dim_reduction,  save_video_path_dim,
                                args['save_plots_or_only_create_movie'], save_plots_folder_dir)


def plot_umap_embedding(axis_limits, dim_reduction_results, color_array, cmap, dots_to_plot_line, images_to_plot,
                        images_to_plot_dim_reduction, save_video_path, save_plots, save_plots_path):  # todo expand it also to 3D
    # ====== set and create if needed the folder where we will save the videos ======
    h_fig = plt.figure(figsize=(7, 8))  # todo see how I can save the plots while doing the movie
    # ====== plot the umap =======
    h_ax_1 = set_axis(h_fig, 'umap', x_min=axis_limits[0], x_max=axis_limits[1], y_min=axis_limits[2], y_max=axis_limits[3])
    h_im_1 = h_ax_1.scatter(dim_reduction_results[-1][:, 0], dim_reduction_results[-1][:, 1],
                            s=0.7, alpha=0.5, c=color_array, cmap=cmap)
    h_ax_1.set_aspect('equal')
    # ===== set the measurements of the colorbar ======
    divider = make_axes_locatable(h_ax_1)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    h_fig.colorbar(h_im_1, cax=cax)
    # ======= translate the color_array so we can color the circle representing each image in the correct color =======
    cNorm = colors.Normalize(vmin=0, vmax=color_array.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    color_circle = [scalarMap.to_rgba(color_array[specific_images_to_plot[0]]),
                    scalarMap.to_rgba(color_array[specific_images_to_plot[1]]),
                    scalarMap.to_rgba(color_array[specific_images_to_plot[2]])]
    # ====== plot each circle above and each image, will be done only once=======
    for i in range(3):
        h_ax = set_axis(h_fig, 'rat_image_{}'.format(i + 1))
        circle_image = plt.Circle((25, -4), 2, color=color_circle[i], clip_on=False)
        h_ax.add_artist(circle_image)
        h_ax.matshow(images_to_plot[i], cmap='gray')
    # ===== plots lines from mini figure to the images on the side ======
    h_ax_2 = set_axis(h_fig, 'umap', x_min=axis_limits[0], x_max=axis_limits[1], y_min=axis_limits[2],
                      y_max=axis_limits[3])
    h_line_dict = {
        i: h_ax_2.plot(dots_to_plot_line[-1][:, 0:2][i], dots_to_plot_line[-1][:, 2:4][i], alpha=0.5, c='gray')[0]
        for i in range(3)}
    # ====== plot specific images dim reduction dots=======
    h_ax_3 = set_axis(h_fig, 'umap', x_min=axis_limits[0], x_max=axis_limits[1], y_min=axis_limits[2], y_max=axis_limits[3])
    h_im_2 = h_ax_3.scatter(images_to_plot_dim_reduction[-1][:, 0], images_to_plot_dim_reduction[-1][:, 1],
                            s=50,
                            c=color_circle, edgecolors='black')

    if len(dots_to_plot_line) == 1:
        plt.savefig(save_video_path, dpi=300)
    else:
        n_frames = len(dim_reduction_results)
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Rat movie {}'.format(save_video_path.split('.')[0]), artist='Matplotlib')
        writer = FFMpegWriter(fps=30, metadata=metadata)
        with tqdm(total=n_frames) as pbar:
            with writer.saving(h_fig, save_video_path, dpi=300):  # change from 600 dpi
                for i in range(n_frames):
                    # ===== re-plot the umap scatter plot only if the mode is not alternative latent space =====
                    if len(dim_reduction_results) > 1:
                        h_im_1.set_offsets(dim_reduction_results[i])
                        h_ax_1.set_title('Network update: {}'.format(i), x=0.46, fontsize=16)
                    for index, h_line in h_line_dict.items():
                        h_line.remove()
                        h_line_dict[index] = h_ax_2.plot(dots_to_plot_line[i][:, 0:2][index], dots_to_plot_line[i][:, 2:4][index], alpha=0.5, c='gray')[0]
                    h_im_2.set_offsets(images_to_plot_dim_reduction[i])
                    writer.grab_frame()
                    pbar.update(1)
                    if save_plots or (n_frames - 1 == i):
                        plt.savefig(os.path.join(save_plots_path, save_video_path.split('.')[0] + '.png'), dpi=300)


def set_cmap(color_type):
    if 'Angle' in color_type or 'polar' in color_type:
        cmap = 'hsv_r'
    elif 'noise' in color_type:
        cmap = 'cool'
    else:
        cmap = 'Spectral'
    return cmap


def run_dim_reduction(folder_dir, latent_space_dir, which_fc, fit_umap_according_to_epoch='last', only_fit_umap=False,
                      only_transform_umap=False,
                      latent_space_file_name='latent_space_last.npz'):
    # ====== run 2D UMAP on the data and save the result =======
    if only_fit_umap and only_transform_umap:
        fit_umap(latent_space_dir, folder_dir, which_fc,
                 latent_space_file_name, fit_umap_according_to_epoch)
        umap_embedding = transform_umap(latent_space_dir, folder_dir,
                                             fit_umap_according_to_epoch, which_fc,
                                             latent_space_file_name)
        return umap_embedding
    elif only_fit_umap:
        fit_umap(latent_space_dir, folder_dir, which_fc,
                 latent_space_file_name, fit_umap_according_to_epoch)
        pass
    elif only_transform_umap:
        umap_embeddings = transform_umap(latent_space_dir, folder_dir,
                                             fit_umap_according_to_epoch, which_fc,
                                             latent_space_file_name)
        return umap_embeddings


def fit_umap(latent_space_dir, folder_dir, which_fc, latent_space_file_name, fit_umap_according_to_epoch):
    # ====== load the latent space data if it was not created by UMAP =====
    latent_space_name = latent_space_file_name.split('latent_space_')[1].split('.npz')[0]
    umap_fit_models_folder_dir = os.path.join(folder_dir, 'Umap_models_{}'.format(which_fc))
    create_folder_if_needed(umap_fit_models_folder_dir)
    if fit_umap_according_to_epoch == 'fit to alternative latent space':
        model_name = 'UMAP_model_2D_fit{}_{}.sav'.format('_' + latent_space_name, fit_umap_according_to_epoch)
    else:
        model_name = 'UMAP_model_2D_fit{}.sav'.format('_' + latent_space_name)
    if os.path.exists(os.path.join(umap_fit_models_folder_dir, model_name)):
        pass
    else:
        latent_space_matrix = load_npz_file(latent_space_dir, latent_space_file_name,
                                            mode='fit')
        umap_model = umap.UMAP(random_state=56, n_components=2).fit(latent_space_matrix)
        joblib.dump(umap_model, os.path.join(umap_fit_models_folder_dir, model_name))
        if fit_umap_according_to_epoch == 'fit to alternative latent space':
            get_umap_embedding_alternative_umap(latent_space_matrix, umap_model, folder_dir, which_fc)


def transform_umap(latent_space_dir, folder_dir, fit_umap_according_to_epoch, which_fc, latent_space_file_name):
    latent_space_matrix = load_npz_file(latent_space_dir, latent_space_file_name, mode='transform')
    epoch_details_for_file_name = latent_space_file_name.split('latent_space_')[1].split('.npz')[0]
    embedding_folder_dir = os.path.join(folder_dir, 'Umap_embedding_{}'.format(which_fc))
    create_folder_if_needed(embedding_folder_dir)
    umap_embeddings_path = os.path.join(embedding_folder_dir,
                                        '2D_UMAP_embedding_fit_{}_{}.npz'.format(fit_umap_according_to_epoch,
                                                                                  epoch_details_for_file_name))
    if os.path.exists(umap_embeddings_path):
        umap_embedding = load_npz_file(umap_embeddings_path, None, 'load_embeddings')
    else:
        umap_model = open_umap_models(folder_dir, which_fc, fit_umap_according_to_epoch, epoch=epoch_details_for_file_name)
        umap_embedding = umap_model.transform(latent_space_matrix)
        np.savez_compressed(umap_embeddings_path, umap_embedding)
    return umap_embedding


def open_umap_models(folder_dir, which_fc, option, epoch=None):
    umap_models_folder_dir = os.path.join(folder_dir, 'Umap_models_{}'.format(which_fc))
    umap_model_name_list = natsorted([umap_model_name for umap_model_name in os.listdir(umap_models_folder_dir)])
    if option == 'last':
        umap_model_name = umap_model_name_list[-1]
    elif option == 'first':
        umap_model_name = umap_model_name_list[0]
    elif option == 'every_epoch':
        umap_model_name = [umap_model_name for umap_model_name in umap_model_name_list if
                           (epoch in umap_model_name and 'alternative' not in umap_model_name)][0]
    elif option == 'fit to alternative latent space':
        umap_model_name = [umap_model_name for umap_model_name in umap_model_name_list if
                           'alternative' in umap_model_name][0]
    umap_model = joblib.load(os.path.join(umap_models_folder_dir, umap_model_name))
    return umap_model



def get_umap_embedding_alternative_umap(latent_space_matrix, umap_model, folder_dir, which_fc):
    embeeding_folder_dir = os.path.join(folder_dir, 'Umap_embedding_{}'.format(which_fc))
    create_folder_if_needed(embeeding_folder_dir)
    umap_embeddings_path = os.path.join(embeeding_folder_dir,
                                        '2D_UMAP_alternative_embedding_fit.npz')
    umap_embedding = umap_model.transform(latent_space_matrix)
    np.savez_compressed(umap_embeddings_path, umap_embedding)



def extract_dim_reduction_embeddings(folder_dir, args, latent_space_dir, which_fc, fit_umap_according_to_epoch, latent_space_files,
                                 only_fit_umap=False):
    umap_embedding_list = []
    for file_name in latent_space_files:
        umap_embedding = run_dim_reduction(folder_dir, latent_space_dir, which_fc,
                                            fit_umap_according_to_epoch=fit_umap_according_to_epoch,
                                           only_fit_umap=only_fit_umap, only_transform_umap=True,
                                            latent_space_file_name=file_name)
        umap_embedding_list += [umap_embedding]
    return umap_embedding_list


def extract_dim_reduction_images_to_plot(dim_reduction_results):
    image_to_plot_dim_reduction = []
    for i in range(len(dim_reduction_results)):
        dim_reduction_image_1 = np.expand_dims(dim_reduction_results[i][specific_images_to_plot[0]], axis=0)
        dim_reduction_image_2 = np.expand_dims(dim_reduction_results[i][specific_images_to_plot[1]], axis=0)
        dim_reduction_image_3 = np.expand_dims(dim_reduction_results[i][specific_images_to_plot[2]], axis=0)
        image_to_plot_dim_reduction += [
            np.concatenate((dim_reduction_image_1, dim_reduction_image_2, dim_reduction_image_3))]
    return image_to_plot_dim_reduction


def set_axis(h_fig, mode, x_min=None, x_max=None, y_min=None, y_max=None):
    if mode == 'umap':
        h_ax = h_fig.add_axes([0.15, 0.1, 0.78, 0.78])
        # ======= set max and min for ax, remove ticks and the boxes axises ======
        h_ax.set_xlim(x_min, x_max)
        h_ax.set_ylim(y_min, y_max)
        h_ax.spines['top'].set_visible(False)
        h_ax.spines['right'].set_visible(False)
        h_ax.spines['bottom'].set_visible(False)
        h_ax.spines['left'].set_visible(False)
        h_ax.set_xlabel('UMAP 1', fontsize=11)
        h_ax.set_ylabel('UMAP 2', fontsize=11)
    elif mode == 'rat_image_1':
        h_ax = h_fig.add_axes([0.07, 0.2, 0.15, 0.15])
        h_ax.set_axis_off()
    elif mode == 'rat_image_2':
        h_ax = h_fig.add_axes([0.07, 0.70, 0.15, 0.15])
        h_ax.set_axis_off()
    elif mode == 'rat_image_3':
        h_ax = h_fig.add_axes([0.7, 0.7, 0.15, 0.15])
        h_ax.set_axis_off()
    h_ax.get_xaxis().set_ticks([])
    h_ax.get_yaxis().set_ticks([])
    return h_ax


def extract_dots_for_line(images_to_plot_dim_reduction, axis_limits):
    rat_images_x = np.array([[0.8 * axis_limits[0]], [0.8 * axis_limits[0]], [0.56 * axis_limits[1]]])
    rat_images_y = np.array([[0.493 * axis_limits[2]], [0.713 * axis_limits[3]], [0.713 * axis_limits[3]]])
    dots_to_plot_line = []
    for i in range(len(images_to_plot_dim_reduction)):
        x_dim_reduction = np.expand_dims(images_to_plot_dim_reduction[i][:, 0], axis=1)
        y_dim_reduction = np.expand_dims(images_to_plot_dim_reduction[i][:, 1], axis=1)
        dots_to_plot_line += [np.concatenate((rat_images_x, x_dim_reduction, rat_images_y, y_dim_reduction), axis=1)]
    return dots_to_plot_line


def extract_images_to_plot(save_folder_dir, args, folder_dir):
    # ====== set model to test =====
    model_to_test = set_model_list(args['checkpoint_path'], 'last')[0]
    model = load_model(model_to_test, args)
    # ====== set which images we want to plot =====
    # ===== if the feature were extracted by PCA than load the pca as the images (network input) ====
    if args['dim_reduction_algo'] == 'PCA':
        images = load_npz_file(os.path.join(folder_dir,'Latent_space_arrays_fc1'), 'latent_space_PCA.npz', mode='transform')
        images = torch.FloatTensor(images).unsqueeze(1)
        images_to_plot = torch.cat([images[specific_images_to_plot[0]],
                                    images[specific_images_to_plot[1]],
                                    images[specific_images_to_plot[2]]]).to(args['device'])
    else:
        images = load_plus_scale_images(args)
        images = torch.FloatTensor(images).unsqueeze(1)
        images_to_plot = torch.cat([images[specific_images_to_plot[0]],
                                    images[specific_images_to_plot[1]],
                                    images[specific_images_to_plot[2]]]).to(args['device']).unsqueeze(1)
    dataloader_images_to_plot = DataLoader(images_to_plot, batch_size=3,
                                           shuffle=False)
    # ====== extract de-noised images ====
    with torch.no_grad():
        __, ___, predicted_images = test_model(model, dataloader_images_to_plot, None, 'test', save_folder_dir)
    return predicted_images.squeeze(1)


def calculate_max_min_axis(array_to_boundaries_on):  # todo expand to 3D
    if type(array_to_boundaries_on) == list:
        array_to_boundaries_on = np.array(array_to_boundaries_on)
        max_axis = array_to_boundaries_on.max(axis=1).max(axis=0)
        min_axis = array_to_boundaries_on.min(axis=1).min(axis=0)
    else:
        max_axis, min_axis = array_to_boundaries_on.max(axis=0), array_to_boundaries_on.min(axis=0)
    x_max, x_min = max_axis[0], min_axis[0]
    y_max, y_min = max_axis[1], min_axis[1]
    # ====== choosingg according to which set of boundaries we will set the axis =======
    max_axis = math.ceil(max(y_max, x_max))
    min_axis = math.floor(min(y_min, x_min))
    absolute_value = max(abs(max_axis), abs(min_axis))
    max_axis = absolute_value
    min_axis = -1 * absolute_value
    # ====== adding factor to the boundary =======
    for element in [max_axis, min_axis]:
        factor = 0.5 if element >= 0 else -0.5
        element += factor
    return max_axis, min_axis, max_axis, min_axis



def load_npz_file(file_dir, file_name, mode):
    if mode == 'load_embeddings':
        load_file_path = file_dir
    else:
        load_file_path = os.path.join(file_dir, file_name)
    with np.load(load_file_path) as f:
        np_array = f['arr_0']
    return np_array


def get_latent_space_pca(args, folder_dir):
    dataloader_dict = loading_plus_preprocessing_data(args, split_to_train_val=True)
    pca = PCA(n_components=args['pca_umap_dim_reduction'])
    images, pca_results = [], []
    for index, dataloader in dataloader_dict.items():
        dataset_array = dataloader.dataset.cpu().numpy()
        dataset_array_flatten = dataset_array.reshape(dataset_array.shape[0], -1)
        if index == 0:
            pca.fit(dataset_array_flatten)
            # ====== calculate variance ratios and plot it ======
            variance = pca.explained_variance_ratio_
            var_explained_by_dim = np.round(np.cumsum(pca.explained_variance_ratio_ * 100), decimals=3)
            extract_pca_data_to_file(folder_dir, variance, var_explained_by_dim, 00)
        pca_result = pca.transform(dataset_array_flatten)
        pca_results += [torch.tensor(pca_result, device=args['device']).float()]
    # ====== return dataset where the x is the PCA reduction results and in the y is the mapped features
    # that would be used to calculate the loss ======
    dataset = {index: TensorDataset(pca_results[index], dataloader_dict[index].dataset) for index in
                   range(len(dataloader_dict))}
    # ===== save the PCA latent space into a file =====
    save_latent_space_into_file(args, pca, folder_dir, 'PCA')
    return dataset


def save_latent_space_into_file(args, model, folder_dir, method):
    dataloader = loading_plus_preprocessing_data(args, split_to_train_val=False)
    dataset_array = dataloader.dataset.cpu().numpy()
    dataset_array_flatten = dataset_array.reshape(dataset_array.shape[0], -1)
    all_data_results = model.transform(dataset_array_flatten)
    save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays_fc1')
    create_folder_if_needed(save_latent_space_folder)
    save_latent_space_to_file(all_data_results, save_latent_space_folder, epoch=method, method=method)
    if args['pca_umap_dim_reduction'] == 2:
        dataset_array = dataloader.dataset.cpu().squeeze(1).numpy().swapaxes(1, 2)
        visualize_umap_embeddings(args, folder_dir, 'fc1',
                                  [all_data_results], method, dataset_array)


def extract_pca_data_to_file(folder_dir, variance, var_explained_by_dim,
                             n_components_95):  # todo fix, where do we use it?
    file_path = os.path.join(folder_dir, 'pca_data.txt')
    plot_pca_var_per_componets(var_explained_by_dim, folder_dir)
    n_componets_array = np.arange(len(variance))
    variance_per_component = np.array(list(zip(n_componets_array, variance)))
    var_explained_by_dim = np.array(list(zip(n_componets_array, var_explained_by_dim)))
    with open(file_path, 'w') as f:
        f.write('For PCA which explains 95 percent of the variance we need to set %d components\n' % n_components_95)
        np.savetxt(f, variance_per_component, delimiter=',')
        np.savetxt(f, var_explained_by_dim, delimiter=',')


def plot_pca_var_per_componets(var_explained_by_dim, folder_dir):
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Components')
    plt.xticks(np.arange(len(var_explained_by_dim)), np.arange(len(var_explained_by_dim)) + 1 )
    plt.title('PCA Analysis with %d Components' % (var_explained_by_dim.shape[0]))
    plt.plot(var_explained_by_dim)
    plt.savefig(os.path.join(folder_dir, 'PCA Analysis with %d Components.png' % (var_explained_by_dim.shape[0])))
    plt.close()


def get_latent_space_umap(args, folder_dir):
    dataloader_dict = loading_plus_preprocessing_data(args, split_to_train_val=True)
    umap_model = umap.UMAP(random_state=42, n_components=args['pca_umap_dim_reduction'])
    umap_results = []
    for index, dataloader in dataloader_dict.items():
        dataset_array = dataloader.dataset.cpu().numpy()
        dataset_array_flatten = dataset_array.reshape(dataset_array.shape[0], -1)
        if index == 0:
            umap_model.fit(dataset_array_flatten)
            joblib.dump(umap_model, os.path.join(folder_dir, 'UMAP_model.sav'))
        umap_result = umap_model.transform(dataset_array_flatten)
        umap_results += [torch.tensor(umap_result, device=args['device']).float()]
    # ====== return dataset where the x is the UMAP reduction results and in the y is the mapped features
    # that would be used to calculate the loss ======
    dataset = {index: TensorDataset(umap_results[index], dataloader_dict[index].dataset) for index in
               range(len(dataloader_dict))}
    # ===== save the PCA latent space into a file =====
    save_latent_space_into_file(args, umap_model, folder_dir, 'UMAP')
    return dataset