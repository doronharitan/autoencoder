import json
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import time
import umap
from sklearn.decomposition import PCA
import joblib
from PIL import Image
import os
from natsort import natsorted
import numpy as np
import math
from tqdm import tqdm
import random
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from args import parser
from models.autoencoder_model import Autoencoder

color_datapoints_according_to_specific_condition_dict = {7: 'Angle of the body',
                                  10: 'distance from the center of the arena',
                                  11: 'polar representation of rat location'}

color_datapoints_according_to_specific_condition_dict = {6: 'Angle of the head', 7: 'Angle of the body',
                                      10: 'distance from the center of the arena',
                                      11: 'polar representation of rat location',
                                      12: 'Did the rat inserted noise into port'}



def setting_parameters(use_folder_dir=False, mode=None):
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_folder_dir:
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
            get_latent_space_ae(None, folder_dir, model=model, epoch=epoch,
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
    output_img_tensor_to_plot = torch.cat((local_images_batch[:8], output[:8]))
    save_image(output_img_tensor_to_plot, os.path.join(save_images_path, '{}_epoch.png'.format(epoch)))
    # ====== if test mode return images and output else return only loss======
    if epoch == 'test':
        return val_loss, torch.cat(original_images), torch.cat(output_images)
    else:
        return val_loss


def prediction_step(model, dataset, criterion, labels=None, mode='train'):
    if mode != 'train':
        with torch.no_grad():
            outputs = model(dataset)
    else:
        outputs = model(dataset)
    # ====== Calculate loss =======
    if labels is None:
        loss = criterion(outputs, dataset)
    else:
        loss = criterion(outputs, labels)
    return loss, outputs.detach()


def get_latent_space_ae(args, folder_dir, fit_umap_according_to_epoch=None, fc2_mode=False, model=None,
                        epoch='test', batch=None, dataloader_all_data=None):
    save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays_{}'. format('fc2' if fc2_mode else 'fc1'))
    create_folder_if_needed(save_latent_space_folder)
    dataloader_all_data = set_dataloader(dataloader_all_data, args)  # todo extract to a new function the pca option
    #  ====== load the relevant model and predict the latent space =======
    if model is None:
        # sort the model saved according to checkpoints
        model_list = set_model_list(args['checkpoint_path'], fit_umap_according_to_epoch)
        for model_name in model_list:
            extract_latent_space_prediction(model_name, args['checkpoint_path'], dataloader_all_data,
                                        save_latent_space_folder, fc2_mode, epoch, batch)
    #  ====== use the forwarded model to get the latent space =======
    elif model is not None:
        extract_latent_space_prediction(None, None, dataloader_all_data,
                                    save_latent_space_folder, fc2_mode, epoch, batch, model=model)


def set_dataloader(dataloader_all_data, args):
    if dataloader_all_data is None:
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


def extract_latent_space_prediction(model_name, model_dir, dataloader, save_latent_space_folder,
                                fc2_mode, epoch, batch, model=None):
    if model_name != None:
        epoch = model_name.split('.pth.tar')[0].split('model_')[1]
    # ====== check if the latent space file exists else create them =======
    if os.path.exists(os.path.join(save_latent_space_folder, 'latent_space_{}.npz'.format(epoch))):
        pass
    else:
        # ====== run forward pass till the latent_space ======
        # ====== load model if needed ======
        if model is None:
            model_path = os.path.join(model_dir, model_name)
            model = torch.load(model_path)
        # ====== move model to eval mode with no grads and predict the latent space ======
        model.eval()
        latent_space_arrays = []
        with torch.no_grad():
            for local_images_batch in dataloader:
                encoder_output = model.encoder(local_images_batch)
                batch_size, num_filters, w, h = encoder_output.shape
                fc_input = encoder_output.view(batch_size, num_filters * h * w)
                latent_space = model.fc_1(fc_input)
                if fc2_mode:
                    latent_space = model.fc_2(latent_space)
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
            get_latent_space_ae(args, folder_dir, fit_umap_according_to_epoch=fit_umap_according_to_epoch,
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
    save_video_path = os.path.join(folder_dir, 'Videos')
    create_folder_if_needed(save_video_path)
    images_to_plot_dim_reduction = extract_dim_reduction_images_to_plot(umap_embedding)
    if images_for_plot is None:
        images_to_plot = extract_images_to_plot(save_video_path, args)
    else:
        images_to_plot = np.stack([images_for_plot[226], images_for_plot[85], images_for_plot[154]])
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
    color_circle = [scalarMap.to_rgba(color_array[226]), scalarMap.to_rgba(color_array[85]),
                    scalarMap.to_rgba(color_array[154])]
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
                    if save_plots:
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
        umap_model = umap.UMAP(random_state=34, n_components=2).fit(latent_space_matrix)
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
        dim_reduction_image_1 = np.expand_dims(dim_reduction_results[i][226], axis=0)
        dim_reduction_image_2 = np.expand_dims(dim_reduction_results[i][85], axis=0)
        dim_reduction_image_3 = np.expand_dims(dim_reduction_results[i][154], axis=0)
        image_to_plot_dim_reduction += [
            np.concatenate((dim_reduction_image_1, dim_reduction_image_2, dim_reduction_image_3))]
    return image_to_plot_dim_reduction


def set_axis(h_fig, mode, x_min=None, x_max=None, y_min=None, y_max=None):
    if mode == 'umap':
        h_ax = h_fig.add_axes([0.15, 0.1, 0.78, 0.78])
        # ======= set max and min for ax, remove ticks and the boxes axises ======
        h_ax.set_xlim(x_min, x_max)
        h_ax.set_ylim(y_min, y_max)
        # h_ax.spines['top'].set_visible(False)
        # h_ax.spines['right'].set_visible(False)
        # h_ax.spines['bottom'].set_visible(False)
        # h_ax.spines['left'].set_visible(False)
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
    # h_ax.get_xaxis().set_ticks([])
    # h_ax.get_yaxis().set_ticks([])
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


def extract_images_to_plot(save_folder_dir,args):
    images = load_plus_scale_images(args)
    images = torch.FloatTensor(images).unsqueeze(1)
    images_to_plot = torch.cat([images[226], images[85], images[154]]).to(args['device']).unsqueeze(1)
    dataloader_images_to_plot = DataLoader(images_to_plot, batch_size=2,
                                           shuffle=False)
    model_to_test = set_model_list(args['checkpoint_path'], 'last')[0]
    model = Autoencoder(args['latent_space_dim']).to(args['device'])
    checkpoint = torch.load(os.path.join(args['checkpoint_path'], model_to_test))
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        __, ___, predicted_images = test_model(model, dataloader_images_to_plot, nn.MSELoss(), 'test', save_folder_dir)
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
    min_axis = math.ceil(min(y_min, y_max))
    # ====== adding factor to the boundary =======
    for element in [max_axis, min_axis]:
        factor = 0.5 if element >= 0 else -0.5
        element += factor

    return max_axis, min_axis, max_axis, min_axis



def load_npz_file(file_dir, file_name, mode):
    if mode == 'fit':
        load_file_path = file_name
    elif mode == 'load_embeddings':
        load_file_path = file_dir
    else:
        load_file_path = os.path.join(file_dir, file_name)
    with np.load(load_file_path) as f:
        np_array = f['arr_0']
    return np_array
