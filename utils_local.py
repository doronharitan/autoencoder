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

color_sample_according_to_dict = {7: 'Angle of the body',
                                  10: 'distance from the center of the arena',
                                  11: 'polar representation of rat location'}

color_sample_according_to_dict_all = {6: 'Angle of the head', 7: 'Angle of the body',
                                      10: 'distance from the center of the arena',
                                      11: 'polar representation of rat location',
                                      12: 'Did the rat inserted noise into port'}


# todo erase the colab commendes
# todo add for how many dimentions I want it to work? in the utils


def setting_parameters(use_folder_dir=False):
    hp_parameters_dict = create_hp_dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_folder_dir:
            folder_dir = os.path.normpath(hp_parameters_dict['dir_model'] + os.sep + os.pardir)
    else:
        if hp_parameters_dict['open_new_folder_status'] != 'False':
            folder_dir = open_new_folder(hp_parameters_dict['open_new_folder_status'])
        else:
            folder_dir = os.getcwd()  # run in colab r'./'
    save_setting_info(hp_parameters_dict, device, folder_dir)
    print('The HP parameters that were used in this run are:', hp_parameters_dict)
    print('The code run on:', device)
    print('The folder all of the data is saved on is:', folder_dir)
    hp_parameters_dict['device'] = device
    return folder_dir, hp_parameters_dict


def open_new_folder(open_folder_status):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug'
    folder_dir = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
        os.makedirs(os.path.join(folder_dir, 'Images'))
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


def load_plus_scale_images(hp_parameters_dict):
    # images = np.load(os.path.join(hp_parameters_dict['train_data_dir'], hp_parameters_dict['file_name']))
    with np.load(os.path.join(hp_parameters_dict['train_data_dir'], hp_parameters_dict['file_name'])) as training_data:
        # images = training_data['images']
        images = training_data['arr_0']
    # ====== data pre-processing ========
    images = images / hp_parameters_dict['max_pix_value']
    return images


def loading_plus_preprocessing_data_visualization(hp_parameters_dict,
                                                  encoder_mode='AE model'):  # todo module it and change the name of the varible - visualize_low_dim
    images = load_plus_scale_images(hp_parameters_dict)
    if encoder_mode == 'PCA':
        images = images.reshape(images.shape[0], -1)
        return images
    else:
        dataset = torch.tensor(images, device=hp_parameters_dict['device']).float().unsqueeze(1)
        dataloader = DataLoader(dataset, batch_size=hp_parameters_dict['batch_size_latent_space'],
                                shuffle=False)
        return dataloader


def loading_plus_preprocessing_data(hp_parameters_dict,
                                    encoder_mode='AE model'):  # todo module it and change the name of the varible - visualize_low_dim
    images = load_plus_scale_images(hp_parameters_dict)
    images_train, images_val = train_test_split(images, test_size=hp_parameters_dict['val_split'],
                                                random_state=hp_parameters_dict['seed'])
    if encoder_mode == 'PCA' or encoder_mode == 'UMAP':  # Todo remove it from dataloader we don't need it
        images_train = images_train.reshape(images_train.shape[0], -1)
        images_val = images_val.reshape(images_val.shape[0], -1)
        dataloader_dict = {index: images for index, images in
                           enumerate([images_train, images_val])}  # todo change the name dataloader_dicr?
    else:
        dataloader_dict = {
            index: DataLoader(torch.tensor(data, device=hp_parameters_dict['device']).float().unsqueeze(1),
                              batch_size=hp_parameters_dict['batch_size'],
                              shuffle=True)
            for index, data in enumerate([images_train, images_val])}
    return dataloader_dict


def loading_plus_preprocessing_data_with_labels(hp_parameters_dict, mode='train'):
    images = load_plus_scale_images(hp_parameters_dict)
    # labels = np.load(os.path.join(hp_parameters_dict['train_data_dir'],
    #                               hp_parameters_dict['meta_data_file_name']))[:, 8:10]
    labels_neck = np.load(
        os.path.join(hp_parameters_dict['train_data_dir'], hp_parameters_dict['meta_data_file_name']))[:, 2:4]
    labels_base = np.load(
        os.path.join(hp_parameters_dict['train_data_dir'], hp_parameters_dict['meta_data_file_name']))[:, :2]
    labels = labels_neck - labels_base
    dataset = TensorDataset(torch.tensor(images, device=hp_parameters_dict['device']).float().unsqueeze(1),
                            torch.tensor(labels, device=hp_parameters_dict['device']).float())
    if mode == 'train':
        len_train_datase = round(len(dataset) * (1 - hp_parameters_dict['val_split']))
        len_data_to_split = [len_train_datase, len(dataset) - len_train_datase]
        training_dataset, val_dataset = random_split(dataset, len_data_to_split)
        dataloader_dict = {index: DataLoader(dataset, batch_size=hp_parameters_dict['batch_size'], shuffle=True)
                           for index, dataset in enumerate([training_dataset, val_dataset])}
        return dataloader_dict
    elif mode == 'test':
        dataloader = DataLoader(dataset, batch_size=hp_parameters_dict['batch_size_latent_space'],
                                shuffle=False)
        return dataloader


def save_loss_info_into_a_file(train_loss, val_loss, folder_dir, epoch, lr):
    file_name = os.path.join(folder_dir, 'loss_per_epoch.txt')
    with open(file_name, 'a+') as f:
        f.write('{}D Epoch: Train Loss {:.6f}, Validation loss {:.6f},  lr %.8f\n'
                .format(epoch, train_loss, val_loss, lr))


def train_model(model, train_images, optimizer, criterion, epoch, hp_parameters_dict, folder_dir,
                dataloader_all_data=None):
    train_loss, loss_reg = 0.0, 0.0
    save_space_every_batches = len(train_images) // 3
    for num_batch, local_batch in enumerate(train_images):
        model.train()
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        loss, __ = prediction_step(model, local_batch, criterion)
        train_loss += loss.item()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights with the gradients
        if hp_parameters_dict['save_latent_space'] and num_batch % save_space_every_batches == 0:
            get_latent_space(hp_parameters_dict, folder_dir, model=model, epoch=epoch,
                             batch=num_batch, dataloader_all_data=dataloader_all_data)
    return train_loss


def test_model(model, test_images, criterion, epoch, folder_dir):
    val_loss = 0.0
    model.eval()
    if epoch == 'test':
        output_images, original_images = [], []
    for batch_num, local_batch in enumerate(test_images):
        loss, output = prediction_step(model, local_batch, criterion, mode='val')
        if epoch == 'test':
            output_images += [output.detach().cpu()]
            original_images += [local_batch.detach().cpu()]
        val_loss += loss.item()
    # ====== Visualize the output images ======
    output_img_tensor = torch.cat((local_batch[:8], output[:8]))
    save_image(output_img_tensor, os.path.join(folder_dir, '{}_epoch.png'.format(epoch)))
    # ====== if test mode return images and output else return only loss======
    if epoch == 'test':
        original_images = torch.cat(original_images)
        output_images = torch.cat(output_images)
        return val_loss, original_images, output_images
    else:
        return val_loss


def prediction_step(model, dataset, criterion, labels=None, mode='train'):
    if mode != 'train':
        with torch.no_grad():
            outputs = model(dataset)
    else:
        outputs = model(dataset)
    if labels is None:
        loss = criterion(outputs, dataset)
    else:
        loss = criterion(outputs, labels)
    return loss, outputs.detach()


def train_model_pca_umap(model, train_images, optimizer, criterion):
    train_loss, loss_reg = 0.0, 0.0
    for num_batch, local_batch in enumerate(train_images):
        model.train()
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        loss, __ = prediction_step_pca_umap(model, local_batch, criterion)
        train_loss += loss.item()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights with the gradients
    return train_loss


def test_model_pca_umap(model, test_images, criterion, epoch, folder_dir):
    val_loss = 0.0
    model.eval()
    for batch_num, local_batch in enumerate(test_images):
        loss, output = prediction_step_pca_umap(model, local_batch, criterion, mode='val')
        ___, local_batch = local_batch
        val_loss += loss.item()
    # ====== Visualize the output images ======
    output_img_tensor = torch.cat((local_batch[:8], output[:8]))
    save_image(output_img_tensor, os.path.join(folder_dir, './Images/%d_epoch.png' % epoch))
    return val_loss


def prediction_step_pca_umap(model, dataset, criterion, mode='train'):
    dataset, images = dataset
    if mode != 'train':
        with torch.no_grad():
            outputs = model.forward_pca_umap(dataset)
    else:
        outputs = model.forward_pca_umap(dataset)
    loss = criterion(outputs, images)
    return loss, outputs.detach()


def train_model_predicted_position(model, train_images, optimizer, criterion, epoch, folder_dir, save_latent_space,
                                   dataloader_all_data=None, transfer_learning=False):
    train_loss = 0.0
    save_space_every_batches = len(train_images) // 3
    for num_batch, (local_batch, local_labels) in enumerate(train_images):
        model.train()
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        loss, ___ = prediction_step(model, local_batch, criterion, local_labels)
        train_loss += loss.item()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights with the gradients
        if save_latent_space and num_batch % save_space_every_batches == 0:
            save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays_fc1')
            create_folder_if_needed(save_latent_space_folder)
            run_latent_space_prediction_predict_position(dataloader_all_data, save_latent_space_folder,
                                                         model, epoch, num_batch, transfer_learning)
    return train_loss


def test_model_predicted_position(model, test_images, criterion, folder_dir, epoch):
    val_loss = 0.0
    model.eval()
    if epoch == 'test':
        labels_positions, predicated_positions = [], []
    for local_batch, local_labels in test_images:
        loss, outputs = prediction_step(model, local_batch, criterion, local_labels, mode='val')
        val_loss += loss.item()
        if epoch == 'test':
            labels_positions += [local_labels.detach().cpu()]
            predicated_positions += [outputs.detach().cpu()]
    # ====== plot the label, the prediction on the input image =======
    if epoch == 'test':
        plot_labels_vs_predictions_on_arena(labels_positions, predicated_positions, folder_dir)
    else:
        plot_labels_vs_predictions_on_arena(local_labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                                            folder_dir, epoch)
    if epoch == 'test':
        labels_positions = torch.cat(labels_positions)
        predicated_positions = torch.cat(predicated_positions)
        return val_loss, labels_positions, predicated_positions
    else:
        return val_loss


def get_latent_space(hp_parameters_dict, folder_dir, option=None, fc2_mode=None, model=None,
                     epoch=None, batch=None, dataloader_all_data=None):
    # ====== loading the data and pre-processing it =========
    if hp_parameters_dict['get_latent_space_method'] == 'AE_model':
        get_latent_space_ae(hp_parameters_dict, folder_dir, option, fc2_mode,
                            model=model, epoch=epoch, batch=batch, dataloader_all_data=dataloader_all_data)
    elif hp_parameters_dict['get_latent_space_method'] == 'PCA':  # todo check code
        get_latent_space_pca_visualization(hp_parameters_dict, folder_dir)
    elif hp_parameters_dict['get_latent_space_method'] == 'PCA_decoder':  # todo check code
        dataloader_pca = get_latent_space_pca_visualization(hp_parameters_dict,
                                                            encoder_mode='PCA_decoder')
        get_latent_space_ae(hp_parameters_dict, latent_space_pca=dataloader_pca)  # todo 3


def get_latent_space_ae(hp_parameters_dict, folder_dir, option, fc2_mode, model=None, latent_space_pca=None, epoch=None,
                        batch=None, dataloader_all_data=None):
    if fc2_mode:
        save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays_fc2')
    else:
        save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays_fc1')
    create_folder_if_needed(save_latent_space_folder)
    dataloader = set_dataloader(latent_space_pca, dataloader_all_data, hp_parameters_dict)
    #  ====== load the relevant model and get the latent space =======
    if model is None:
        # sort the model saved according to checkpoints
        model_list = natsorted([model_name for model_name in os.listdir(hp_parameters_dict['dir_model'])])
        if option == 'first':
            model_list = [model_list[0]]
        elif option != 'every_epoch':
            model_list = [model_list[-1]]
        for model_name in model_list:
            run_latent_space_prediction(model_name, hp_parameters_dict['dir_model'], dataloader, latent_space_pca,
                                        save_latent_space_folder, fc2_mode)
    #  ====== use the forwarded model to get the latent space =======
    elif model is not None:
        run_latent_space_prediction(None, None, dataloader, latent_space_pca,
                                    save_latent_space_folder, fc2_mode, model=model, epoch=epoch, batch=batch)


def get_latent_space_pca(hp_parameters_dict, device, folder_dir):  # todo change?
    dataloader_dict = loading_plus_preprocessing_data(hp_parameters_dict, device,
                                                      encoder_mode='PCA')  # Todo run it without scale data
    pca = PCA(n_components=hp_parameters_dict['pca_dim_reduction'])
    images, pca_results = [], []  # todo change it and make it prettier
    for index, dataset in dataloader_dict.items():
        if index == 0:
            pca.fit(dataset)
            variance = pca.explained_variance_ratio_  # calculate variance ratios
            var_explained_by_dim = np.round(np.cumsum(pca.explained_variance_ratio_ * 100), decimals=3)
            extract_pca_data_to_file(folder_dir, variance, var_explained_by_dim, 00)
        pca_result = pca.transform(dataset)
        pca_results += [torch.tensor(pca_result, device=device).float()]
        images += [
            torch.tensor(dataset, device=device).float().view(dataset.shape[0], 50, 50).unsqueeze(1)]  # todo change it
    dataset = {index: TensorDataset(pca_results[index], images[index]) for index in range(len(images))}
    return dataset


def get_latent_space_umap(hp_parameters_dict, device, folder_dir):  # todo change?
    dataloader_dict = loading_plus_preprocessing_data(hp_parameters_dict, device,
                                                      encoder_mode='UMAP')  # Todo run it without scale data
    umap_model = umap.UMAP(random_state=42, n_components=hp_parameters_dict['latent_space_dim'])
    images, umap_results = [], []  # todo change it and make it prettier
    for index, dataset in dataloader_dict.items():
        if index == 0:
            umap_model.fit(dataset)
            joblib.dump(umap_model, os.path.join(folder_dir, 'UMAP_model.sav'))
        umap_result = umap_model.transform(dataset)
        umap_results += [torch.tensor(umap_result, device=device).float()]
        images += [
            torch.tensor(dataset, device=device).float().view(dataset.shape[0], 50, 50).unsqueeze(1)]  # todo change it
    dataset = {index: TensorDataset(umap_results[index], images[index]) for index in range(len(images))}
    return dataset


def set_dataloader(latent_space_pca, dataloader_all_data, hp_parameters_dict):
    if latent_space_pca is None:
        if dataloader_all_data is None:
            dataloader = loading_plus_preprocessing_data_visualization(hp_parameters_dict)
        else:
            dataloader = dataloader_all_data
    else:
        dataloader = latent_space_pca
    return dataloader


def run_latent_space_prediction(model_name, model_dir, dataloader_dict, latent_space_pca, save_latent_space_folder,
                                fc2_mode, model=None, epoch=None, batch=None):
    if model_name != None:
        epoch = model_name.split('.pth.tar')[0].split('model_')[1]
    if os.path.exists(os.path.join(save_latent_space_folder, 'latent_space_{}.npz'.format(epoch))):
        pass
    else:
        # ====== run forward pass till the latent_space ======
        # load model if needed
        if model is None:
            model_path = os.path.join(model_dir, model_name)  # todo here
            model = torch.load(model_path)
        # move model to eval mode with no grads and predict the latent space
        model.eval()
        latent_space_arrays = []
        with torch.no_grad():
            for local_batch in dataloader_dict:
                if latent_space_pca is None:
                    encoder_output = model.encoder(local_batch)
                    batch_size, num_filters, w, h = encoder_output.shape
                    fc_input = encoder_output.view(batch_size, num_filters * h * w)
                    latent_space = model.fc_1(fc_input)
                    if fc2_mode:
                        latent_space = model.fc_2(latent_space)
                else:
                    latent_space = local_batch
                latent_space_arrays += [latent_space.detach().cpu()]
            save_latent_space_to_file(latent_space_arrays, save_latent_space_folder, epoch, batch=batch)


def run_latent_space_prediction_predict_position(dataloader, save_latent_space_folder,
                                                 model, epoch, batch, transfer_learning=False):
    # ====== predict the latent_space ======
    model.eval()
    latent_space_arrays = []
    with torch.no_grad():
        for local_batch, local_label in dataloader:
            if transfer_learning:
                latent_space = model.ae_encoder(local_batch)
            else:
                encoder_output = model.encoder(local_batch)
                batch_size, num_filters, w, h = encoder_output.shape
                fc_input = encoder_output.view(batch_size, num_filters * h * w)
                latent_space = model.fc_1(fc_input)
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
    plt.title('PCA Analysis with %d Components' % (var_explained_by_dim.shape[0]))
    plt.plot(var_explained_by_dim)
    plt.savefig(os.path.join(folder_dir, 'PCA Analysis with %d Components.png' % (var_explained_by_dim.shape[0])))
    plt.close()


def get_latent_space_pca_visualization(hp_parameters_dict, folder_dir,
                                       encoder_mode='PCA'):  # todo fix, where do we use it?
    save_latent_space_folder = os.path.join(folder_dir, 'Latent_space_arrays')
    if not os.path.exists(save_latent_space_folder):
        os.makedirs(save_latent_space_folder)
    images = loading_plus_preprocessing_data_visualization(hp_parameters_dict,
                                                           encoder_mode='PCA')  # Todo run it without scale data
    pca = PCA(n_components=hp_parameters_dict['pca_dim_reduction'])
    pca_95 = PCA(0.95)
    pca_results = pca.fit_transform(images)
    if encoder_mode == 'PCA_decoder':
        pca_results = torch.tensor(pca_results, device=hp_parameters_dict['device']).float()
        dataloader_pca = DataLoader(pca_results, batch_size=hp_parameters_dict['batch_size'], shuffle=False)
        return dataloader_pca
    else:
        pca_results_95 = pca_95.fit_transform(images)
        n_components_95 = pca_results_95.shape[1]
        variance = pca.explained_variance_ratio_  # calculate variance ratios
        var_explained_by_dim = np.round(np.cumsum(pca.explained_variance_ratio_ * 100), decimals=3)
        save_latent_space_to_file(pca_results, save_latent_space_folder, method='PCA')
        extract_pca_data_to_file(folder_dir, variance, var_explained_by_dim, n_components_95)


def get_latent_space_umap_visualization(hp_parameters_dict, device, folder_dir):  # todo fix, where do we use it?
    dataloader_dict = loading_plus_preprocessing_data_visualization(hp_parameters_dict, device, encoder_mode='UMAP')
    for local_batch in dataloader_dict:
        local_batch = local_batch.squeeze(1)
        latent_space_matrix = local_batch.view(local_batch.shape[0], local_batch.shape[1] * local_batch.shape[2]).cpu()
        save_latent_space_to_file(latent_space_matrix, folder_dir, method='UMAP')
    return latent_space_matrix


def get_visualize_latent_space_dim_reduction(hp_parameters_dict, folder_dir, option, output_images=None):
    fc1_mode, fc2_mode = hp_parameters_dict['extract_latent_space'], hp_parameters_dict[
        'extract_latent_space_fc2']  # todo change for informative name
    for index, fc_mode in enumerate([fc1_mode, fc2_mode]):
        which_fc = 'fc{}'.format(index + 1)
        if which_fc == 'fc2' and (
                option == 'every_epoch' or option == 'first'):  # todo to save umap models is t expansive had to shrink it down
            pass
        elif fc_mode:
            # ====== check if the latent space file exists else create them =======
            fc_2_mode_to_pass = False if which_fc == 'fc1' else True
            get_latent_space(hp_parameters_dict, folder_dir, option, fc_2_mode_to_pass)
            # ====== set the path from where we should read the latent space files =======
            latent_space_dir = os.path.join(folder_dir, 'Latent_space_arrays_{}'.format(which_fc))
            latent_space_files = natsorted([file for file in os.listdir(latent_space_dir)])
            latent_space_files = latent_space_files[:420]  # todo an hack remove
            # ====== fit and transform 2/3D umap to the specific latent space ======
            if option != 'every_epoch':
                latent_space_to_fit = set_which_latent_space_to_fit(hp_parameters_dict, option,
                                                                    latent_space_files)
                # ====== fit 2/3D umap to the specific latent space ======
                run_dim_reduction(folder_dir, hp_parameters_dict, latent_space_dir, which_fc, option=option,
                                  only_fit_umap=True,
                                  latent_space_file_name=latent_space_to_fit)  # todo add load file option if I have the file already..so it wouldn't take me so muh time
                # ===== transform the relevant latent space to the  umap fit ======
                umap_2d_embedding, umap_3d_embedding = get_dim_reduction_embeddings \
                    (folder_dir, hp_parameters_dict, latent_space_dir, which_fc, option, latent_space_files)
            elif option == 'every_epoch':
                # ===== fit and transform the relevant latent space to the umap dim reduction  ======
                umap_2d_embedding, umap_3d_embedding, file_name_details_for_plot_titles = get_dim_reduction_embeddings \
                    (folder_dir, hp_parameters_dict, latent_space_dir, which_fc, option, latent_space_files,
                     only_fit_umap=True)
            # ===== visualize the umap embeddings ======
            if option != 'fit to alternative latent space':
                visualize_umap_embeddings(hp_parameters_dict, folder_dir, which_fc,
                                          umap_embeddings=[umap_2d_embedding, umap_3d_embedding],
                                          option=option)
            else:
                visualize_umap_embeddings_tracking_on_alternative_space(hp_parameters_dict, folder_dir, which_fc,
                                                                        [umap_2d_embedding[0], umap_3d_embedding[0]],
                                                                        option, output_images)
        else:
            continue


def set_which_latent_space_to_fit(hp_parameters_dict, option, latent_space_files):
    if option == 'first':
        latent_space_to_fit = latent_space_files[0]
    elif option == 'last':
        latent_space_to_fit = latent_space_files[-1]
    elif option == 'fit to alternative latent space':
        last_latent_space = \
            natsorted([model_name for model_name in os.listdir(hp_parameters_dict['alternative_latent_space_to_fit'])])[
                -1]
        latent_space_to_fit = os.path.join(hp_parameters_dict['alternative_latent_space_to_fit'], last_latent_space)
    return latent_space_to_fit


def get_dim_reduction_embeddings(folder_dir, hp_parameters_dict, latent_space_dir, which_fc, option, latent_space_files,
                                 only_fit_umap=False):
    umap_2d_embedding, umap_3d_embedding, file_name_details_for_plot_titles = [], [], []
    for file_name in latent_space_files:
        umap_embeddings = run_dim_reduction(folder_dir, hp_parameters_dict, latent_space_dir, which_fc,
                                            option=option, only_fit_umap=only_fit_umap, only_transform_umap=True,
                                            latent_space_file_name=file_name)  # todo fix questions
        umap_2d_embedding += [umap_embeddings[0]]
        umap_3d_embedding += [umap_embeddings[1]]
        # file_name_details_for_plot_titles += [file_name.split('.npz')[0].split('latent_space_')[1]]
    return umap_2d_embedding, umap_3d_embedding


def run_dim_reduction(folder_dir, hp_parameters_dict, latent_space_dir, which_fc, option='last', only_fit_umap=False,
                      only_transform_umap=False,
                      latent_space_file_name='latent_space_last.npz'):
    # ====== load the latent space data =====
    if hp_parameters_dict['get_latent_space_method'] == 'UMAP':
        latent_space_matrix = get_latent_space_umap_visualization(hp_parameters_dict)  # todo check if it is OK?
    if hp_parameters_dict['dim_reduction_algo'] == 't-SNE':
        run_tsne(latent_space_matrix, folder_dir)
    elif hp_parameters_dict['dim_reduction_algo'] == 'UMAP':
        # ====== run 2/3D UMAP on the data and save the result =======
        if only_fit_umap and only_transform_umap:
            fit_umap(latent_space_dir, folder_dir, which_fc,
                     latent_space_file_name, option)
            umap_embeddings = get_umap_embedding(latent_space_dir, folder_dir,
                                                 option, which_fc,
                                                 latent_space_file_name)
            return umap_embeddings
        elif only_fit_umap:
            fit_umap(latent_space_dir, folder_dir, which_fc,
                     latent_space_file_name, option)
            pass
        elif only_transform_umap:
            umap_embeddings = get_umap_embedding(latent_space_dir, folder_dir,
                                                 option, which_fc,
                                                 latent_space_file_name)
            return umap_embeddings


def fit_umap(latent_space_dir, folder_dir, which_fc, latent_space_file_name, option):
    # ====== load the latent space data if it was not created by UMAP =====
    latent_space_name = latent_space_file_name.split('latent_space_')[1].split('.npz')[0]
    umap_fit_models_folder_dir = os.path.join(folder_dir, 'Umap_models_{}'.format(which_fc))
    create_folder_if_needed(umap_fit_models_folder_dir)
    for dim in range(2, 4):
        if option == 'fit to alternative latent space':
            model_name = 'UMAP_model_{}D_fit{}_{}.sav'.format(dim, '_' + latent_space_name, option)
        else:
            model_name = 'UMAP_model_{}D_fit{}.sav'.format(dim, '_' + latent_space_name)
        if os.path.exists(os.path.join(umap_fit_models_folder_dir, model_name)):
            pass
        else:
            latent_space_matrix = load_npz_file(latent_space_dir, latent_space_file_name, option, mode='fit')
            umap_model = umap.UMAP(random_state=34, n_components=dim).fit(latent_space_matrix)
            joblib.dump(umap_model, os.path.join(umap_fit_models_folder_dir, model_name))
            if option == 'fit to alternative latent space':
                get_umap_embedding_alternative_umap(latent_space_matrix, umap_model, folder_dir, dim, which_fc)


def get_umap_embedding_alternative_umap(latent_space_matrix, umap_model, folder_dir, dim, which_fc):
    embeeding_folder_dir = os.path.join(folder_dir, 'Umap_embedding_{}'.format(which_fc))
    create_folder_if_needed(embeeding_folder_dir)
    umap_embeddings_path = os.path.join(embeeding_folder_dir,
                                        '{}D_UMAP_alternative_embedding_fit.npz'.format(dim))
    umap_embedding = umap_model.transform(latent_space_matrix)
    np.savez_compressed(umap_embeddings_path, umap_embedding)


def load_npz_file(file_dir, file_name, option, mode):
    if option == 'fit to alternative latent space' and mode == 'fit':
        load_file_path = file_name
    elif mode == 'load_embeddings':
        load_file_path = file_dir
    else:
        load_file_path = os.path.join(file_dir, file_name)
    with np.load(load_file_path) as f:
        np_array = f['arr_0']
    return np_array


def open_umap_models(folder_dir, which_fc, option, dim, epoch=None):
    umap_models_folder_dir = os.path.join(folder_dir, 'Umap_models_{}'.format(which_fc))
    umap_model_name_list = natsorted([umap_model_name for umap_model_name in os.listdir(umap_models_folder_dir)
                                      if (str(dim) + 'D') in umap_model_name])
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


def get_umap_embedding(latent_space_dir, folder_dir, option, which_fc, latent_space_file_name):
    latent_space_matrix = load_npz_file(latent_space_dir, latent_space_file_name, option, mode='transform')
    epoch_details_for_file_name = latent_space_file_name.split('latent_space_')[1].split('.npz')[0]
    embedding_folder_dir = os.path.join(folder_dir, 'Umap_embedding_{}'.format(which_fc))
    create_folder_if_needed(embedding_folder_dir)
    umap_embeddings = []
    for i in range(2, 4):
        umap_embeddings_path = os.path.join(embedding_folder_dir,
                                            '{}D_UMAP_embedding_fit_{}_{}.npz'.format(i, option,
                                                                                      epoch_details_for_file_name))
        if os.path.exists(umap_embeddings_path):
            umap_embedding = load_npz_file(umap_embeddings_path, None, option, 'load_embeddings')
        else:
            umap_model = open_umap_models(folder_dir, which_fc, option, dim=i, epoch=epoch_details_for_file_name)
            umap_embedding = umap_model.transform(latent_space_matrix)
            np.savez_compressed(umap_embeddings_path, umap_embedding)
        umap_embeddings += [umap_embedding]
    return umap_embeddings


def visualize_umap_embeddings(hp_parameters_dict, folder_dir, which_fc,
                              umap_embeddings=None, option=None):
    meta_data = np.load(os.path.join(hp_parameters_dict['train_data_dir'],
                                     hp_parameters_dict['meta_data_file_name']))  # todo can I not import it every time
    if umap_embeddings is None:  # todo need to understand this code anf if needed for tsne?
        data_folder = os.path.join(folder_dir, 'Dim_reduction')
        for file_name in os.listdir(
                data_folder):  # in case I ran the dim reduction with different n_neighbors parameter
            if hp_parameters_dict['dim_reduction_algo'] in file_name:
                algo_name = file_name.split('D ')[1].split(' ')[0]
                dim_reduction_results = np.load(os.path.join(data_folder, file_name), allow_pickle=True)
                plot_umap_embedding(meta_data, dim_reduction_results, folder_dir, file_name,
                                    algo_name)  # todo understand if we need the single plot option if not erase
    else:
        # set and create if needed the folder where we will save the videos and plots
        save_plots_folder_dir = os.path.join(folder_dir, 'UMAP Plots {}'.format(which_fc))
        create_folder_if_needed(save_plots_folder_dir)
        for dim, umap_embedding in enumerate(umap_embeddings):
            if dim == 0:  # todo also for 3D, expand the code, maybe do plot for 3d function?
                plot_umap_embedding(meta_data, umap_embedding, save_plots_folder_dir, hp_parameters_dict,
                                    dim='{}D'.format(dim + 2),
                                    save_plots=hp_parameters_dict['save_plots'], mode='multipule_plots',
                                    # todo do i need this mode?
                                    option=option)
            else:
                continue


def plot_umap_embedding(meta_data, dim_reduction_results, folder_dir, hp_parameters_dict, dim, algo_name='UMAP',
                        save_plots=False, mode='single_plot', option=None):  # todo expand it also to 3D
    # ====== set and create if needed the folder where we will save the videos ======
    save_video_path = os.path.join(folder_dir, 'Videos')
    create_folder_if_needed(save_video_path)
    x_max, x_min, y_max, y_min = get_max_min_axis(dim_reduction_results)
    for index, color_type in color_sample_according_to_dict.items():
        plot_name = '{} UMAP, according to {} fitting, color {}'.format(dim, option, color_type)
        save_video_path_dim = os.path.join(save_video_path, plot_name + '.avi')
        if os.path.exists(save_video_path_dim):
            pass
        else:
            color_array = meta_data[:, index]
            cmap = set_cmap(color_type)
            if mode == 'single_plot':  # todo understand if we need the single plot option if not erase
                plot_single_plot(dim, dim_reduction_results, color_array, cmap,
                                 epoch, algo_name, color_type, folder_dir, option)
            else:
                images_to_plot_dim_reduction = get_dim_reduction_images_to_plot(dim_reduction_results)
                images_to_plot = get_images_to_plot(hp_parameters_dict, save_video_path)
                dots_to_plot_line = get_dots_for_line(images_to_plot_dim_reduction)
                h_fig = plt.figure(figsize=(7, 8))  # todo see how I can save the plots while doing the movie
                h_ax_1 = set_axis(h_fig, 'umap', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                # plot the first ax
                h_im_1 = h_ax_1.scatter(dim_reduction_results[-1][:, 0], dim_reduction_results[-1][:, 1],
                                        s=0.7, alpha=0.5, c=color_array, cmap=cmap)
                h_ax_1.set_aspect('equal')
                h_ax_2 = set_axis(h_fig, 'umap', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                h_line_dict = {i: h_ax_2.plot(dots_to_plot_line[-1][:,0:2][i], dots_to_plot_line[-1][:,2:4][i], alpha=0.5, c='gray')[0]
                               for i in range(3)}
                # set the measurements of the colorbar
                divider = make_axes_locatable(h_ax_1)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                h_fig.colorbar(h_im_1, cax=cax)
                # translate the color_array to colors so we can color the circle in the correct color
                cNorm = colors.Normalize(vmin=0, vmax=color_array.max())
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
                color_circle = [scalarMap.to_rgba(color_array[226]), scalarMap.to_rgba(color_array[85]),
                                scalarMap.to_rgba(color_array[154])]
                for i in range(3):
                    h_ax = set_axis(h_fig, 'rat_image_{}'.format(i + 1))
                    circle_image = plt.Circle((25, -4), 2, color=color_circle[i], clip_on=False)
                    h_ax.add_artist(circle_image)
                    h_ax.matshow(images_to_plot[i], cmap='gray')
                h_ax_5 = set_axis(h_fig, 'umap', x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                h_im_2 = h_ax_5.scatter(images_to_plot_dim_reduction[-1][:, 0], images_to_plot_dim_reduction[-1][:, 1],
                                        s=50,
                                        c=color_circle, edgecolors='black')

                # n_frames = len(dim_reduction_results)
                n_frames = 270
                FFMpegWriter = manimation.writers['ffmpeg']
                metadata = dict(title='Rat movie {}'.format(save_video_path_dim.split('.')[0]), artist='Matplotlib')
                writer = FFMpegWriter(fps=30, metadata=metadata)
                with tqdm(total=n_frames) as pbar:
                    with writer.saving(h_fig, save_video_path_dim, dpi=300):  # change from 600 dpi
                        for i in range(n_frames):
                            h_im_1.set_offsets(dim_reduction_results[i])
                            h_ax_1.set_title('Network update: {}'.format(i), x=0.46, fontsize=16)
                            for index, h_line in h_line_dict.items():
                                h_line.remove()
                                h_line_dict[index] = h_ax_2.plot(dots_to_plot_line[i][:,0:2][index], dots_to_plot_line[i][:,2:4][index], alpha=0.5, c='gray')[0]
                            h_im_2.set_offsets(images_to_plot_dim_reduction[i])
                            writer.grab_frame()
                            pbar.update(1)
                            if save_plots:
                                plt.savefig(os.path.join(folder_dir, plot_name + '.png'), dpi=600)


def set_cmap(color_type):
    if 'Angle' in color_type or 'polar' in color_type:
        cmap = 'hsv_r'
    elif 'noise' in color_type:
        cmap = 'cool'
    else:
        cmap = 'Spectral'
    return cmap


def plot_single_plot(dim, dim_reduction_results, color_array, cmap, epoch, algo_name, color_type, folder_dir,
                     option, axis_boundaries=None):  # todo understand If i need it and if so why?
    dim = 2 if '2D' in dim else 3
    save_plot_dir = os.path.join(folder_dir,
                                 '{}D_{}_epoch_{}_{}_{}.png'.format(dim, epoch, algo_name, color_type, option))
    if dim == 2:
        fig, ax = plt.subplots()
        umap_plot = ax.scatter(dim_reduction_results[:, 0], dim_reduction_results[:, 1],
                               s=0.7, alpha=0.5, c=color_array, cmap=cmap)
        if axis_boundaries != None:
            ax.set_xlim(axis_boundaries[0], axis_boundaries[1])
            ax.set_ylim(axis_boundaries[2], axis_boundaries[3])
        fig.colorbar(umap_plot)
        plt.gca().set_aspect('equal', 'datalim')

    else:
        pass  # todo restore 3D
        fig, ax = plt.subplots()
        ax = Axes3D(fig)
        umap_plot = ax.scatter(dim_reduction_results[:, 0], dim_reduction_results[:, 1],
                               dim_reduction_results[:, 2], s=0.7, alpha=0.5,
                               c=color_array, cmap=cmap)
        fig.set_tight_layout(False)
        fig.colorbar(umap_plot)
    # plt.title('{}D {} epoch {}, {}'.format(dim, epoch, algo_name, color_type)) #todo restore?
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(save_plot_dir) #todo restore
    if epoch is not None:
        fig.canvas.draw()
        umap_plot_array = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                          fig.canvas.tostring_rgb())
        plt.close()
        return umap_plot_array
    plt.close()


def create_video(save_video_path, plot_arrays, fps=5, rgb=True):
    if 'Side_by_side' in save_video_path:
        h_fig = plt.figure(figsize=(8, 4))
    else:
        h_fig = plt.figure(figsize=(8, 8))  # todo change back to 8,8?
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
        with writer.saving(h_fig, save_video_path, dpi=600):
            for i in range(n_frames):
                h_im.set_array(plot_arrays[i])
                writer.grab_frame()
                pbar.update(1)


def visualize_umap_embeddings_tracking_on_alternative_space(hp_parameters_dict, folder_dir, which_fc,
                                                            umap_embeddings, option,
                                                            output_images):  # todo see if I can mearge it with the previous visulaziation
    embeeding_folder_dir = os.path.join(folder_dir,
                                        'Umap_embedding_{}'.format(which_fc))  # todo get better name than alterative
    alternative_embedding_file_names = [embedding_file_name for embedding_file_name in os.listdir(embeeding_folder_dir)
                                        if 'D_UMAP_alternative_embedding_fit.npz' in embedding_file_name]
    umap_alternative_embeddings = []
    for alternative_embedding_file_name in alternative_embedding_file_names:
        path = os.path.join(embeeding_folder_dir, alternative_embedding_file_name)
        umap_embedding = load_npz_file(path, None, option, 'load_embeddings')
        umap_alternative_embeddings += [umap_embedding]
    meta_data = np.load(os.path.join(hp_parameters_dict['train_data_dir'],
                                     hp_parameters_dict['meta_data_file_name']))  # todo can I not import it every time
    save_video_path = os.path.join(folder_dir, 'Videos')
    create_folder_if_needed(save_video_path)
    for dim, umap_embedding in enumerate(umap_embeddings):
        if dim == 0:  # todo just for now, erase when done
            x_max_axis, x_min_axis, y_max_axis, y_min_axis = get_max_min_axis(umap_alternative_embeddings[dim])
            # if dim + 2 == 3:
            #     z_max_axis = max_per_axis[2] #todo fix
            for index, color_type in color_sample_according_to_dict.items():
                save_video_path_dim = os.path.join(save_video_path,
                                                   '{}D UMAP, according to {} fitting, color {}'
                                                   .format(dim + 2, option, color_type))
                if os.path.exists(save_video_path_dim + '.avi'):
                    pass
                else:
                    if os.path.exists(save_video_path + '.npz'):
                        pass
                    else:
                        color_array = meta_data[:, index]
                        cmap = set_cmap(color_type)
                        umap_plot_arrays = []
                        umap_alternative_embedding = umap_alternative_embeddings[dim]
                        # umap_embedding = umap_embedding[:200]  # todo erase
                        with tqdm(total=len(umap_embedding)) as pbar:
                            for image_num, umap_embedding_point in enumerate(umap_embedding):
                                if dim + 2 == 2:
                                    fig = plt.figure()
                                    ax1 = fig.add_axes([0.22, 0.1, 0.8, 0.85])
                                    umap_plot = ax1.scatter(umap_alternative_embedding[:, 0],
                                                            umap_alternative_embedding[:, 1],
                                                            s=0.7, alpha=0.5, c=color_array, cmap=cmap)
                                    fig.colorbar(umap_plot)
                                    ax1.scatter(umap_embedding_point[0], umap_embedding_point[1],
                                                s=50, c='black')
                                    ax1.set_xlim(x_min_axis, x_max_axis)
                                    ax1.set_ylim(y_min_axis, y_max_axis)
                                    plt.xticks([])
                                    plt.yticks([])
                                    plt.box(False)
                                    ax2 = fig.add_axes([0.01, 0.70, 0.26, 0.26])
                                    ax2.imshow(output_images[image_num], cmap='gray')
                                else:
                                    umap_alternative_embedding = umap_alternative_embeddings[dim]
                                    fig, ax = plt.subplots()
                                    ax = Axes3D(fig)
                                    umap_plot = ax.scatter(umap_alternative_embedding[:, 0],
                                                           umap_alternative_embedding[:, 1],
                                                           umap_alternative_embedding[:, 2], s=0.7, alpha=0.5,
                                                           c=color_array, cmap=cmap)
                                    fig.set_tight_layout(False)
                                    fig.colorbar(umap_plot)
                                    ax.scatter(umap_embedding_point[0], umap_embedding_point[1],
                                               umap_embedding_point[2], s=50,
                                               c='black')
                                    # ax.get_zlim(z_max_axis)
                                plt.xticks([])
                                plt.yticks([])
                                fig.canvas.draw()
                                umap_plot_array = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                                                  fig.canvas.tostring_rgb())
                                umap_plot_arrays += [umap_plot_array]
                                plt.close()
                                pbar.update(1)
                            # np.savez_compressed(save_video_path_dim + '.npz', np.array(umap_plot_arrays))
                    create_video(save_video_path_dim + '.avi', umap_plot_arrays, fps=60)  # todo fix


def get_max_min_axis(array_to_boundaries_on):  # todo expand to 3D
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


def create_folder_if_needed(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def run_tsne(latent_space_matrix, folder_dir):
    data_folder = os.path.join(folder_dir, 'Dim_reduction')
    create_folder_if_needed(data_folder)
    # ====== run 2/3D t-SNE on the data and save the result =======
    for i in range(2, 4):
        tsne_results = TSNE(n_components=i).fit_transform(latent_space_matrix)
        np.save(os.path.join(data_folder, '%iD t-SNE repersentation.npy' % (i)), tsne_results)


def plot_labels_vs_predictions_on_arena(original_position, predicted_points, folder_dir, epoch='last'):
    image_save_dir = os.path.join(folder_dir, 'Images')
    # calculate the arena diameter
    arena_diameter_from_labels = original_position.max(axis=0)
    # normalize the positions
    original_position_norm = original_position / arena_diameter_from_labels  # normalize to 1
    predicted_points_norm = predicted_points / arena_diameter_from_labels
    # plot the circle object
    # ====== plot label vs predicted on the arena ======
    # plot on only part of the data, choose randomly on which
    k = 25 if len(original_position) >= 25 else len(original_position) < 25
    random_indexes = random.choices(np.arange(len(original_position_norm)), k=k)
    with tqdm(total=len(random_indexes)) as pbar:
        n_rows = 5 if k == 25 else round(math.sqrt(k))
        n_cols = n_rows
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True)
        fig.suptitle('labels_vs_predictions_on_arena')
        for row in range(n_rows):
            for col in range(n_cols):
                circle = plt.Circle((0, 0), 1, alpha=0.5, color='black')  # todo find a way to reuse the circle
                # circle = plt.Circle((0.5, 0.5), 0.5, alpha=0.5, color='black')  # todo find a way to reuse the circle
                ax[row, col].add_artist(circle)
                # ax[row, col].set_xlim(-0.05, 1.05)
                # ax[row, col].set_ylim(-0.05, 1.05)
                ax[row, col].set_xlim(-1.05, 1.05)
                ax[row, col].set_ylim(-1.05, 1.05)
                ax[row, col].scatter(original_position_norm[row * n_cols + col][0],
                                     original_position_norm[row * n_cols + col][1], color='red', s=5)
                ax[row, col].scatter(predicted_points_norm[row * n_cols + col][0],
                                     predicted_points_norm[row * n_cols + col][1], color='blue', s=5)
                # plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                pbar.update(1)
    plt.savefig(os.path.join(image_save_dir, 'labels_vs_predictions_on_arena_{}_epoch.png'.format(epoch)), dpi=600)


def get_images_to_plot(hp_parameters_dict, save_folder_dir):
    images = load_plus_scale_images(hp_parameters_dict)
    images = torch.FloatTensor(images).unsqueeze(1)
    images_to_plot = torch.cat([images[226], images[85], images[154]]).to(hp_parameters_dict['device']).unsqueeze(1)
    dataloader_images_to_plot = DataLoader(images_to_plot, batch_size=2,
                                           shuffle=False)
    # model_list = natsorted([model_name for model_name in os.listdir(hp_parameters_dict['dir_model'])])
    model_list = natsorted([model_name for model_name in os.listdir('D:\Autoencoder\AE\SOTA - 16D, alex data, save latent space for batch\model check points')])
    # ae_model = torch.load(os.path.join(hp_parameters_dict['dir_model'], model_list[-1])).to(
    #     hp_parameters_dict['device'])
    ae_model = torch.load(os.path.join('D:\Autoencoder\AE\SOTA - 16D, alex data, save latent space for batch\model check points', model_list[-1])).to(
        hp_parameters_dict['device'])
    criterion = nn.MSELoss()
    with torch.no_grad():
        __, ___, predicted_images = test_model(ae_model, dataloader_images_to_plot, criterion, 'test', save_folder_dir)
    return predicted_images.squeeze(1)


def get_dim_reduction_images_to_plot(dim_reduction_results):
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
        # set max and min for ax, remove ticks and the boxes
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


def get_dots_for_line(images_to_plot_dim_reduction):
    rat_images_x = np.array([[-9.4], [-9.4], [8.5]])
    rat_images_y = np.array([[-5.4], [10.3], [10.3]])
    # rat_images_x = np.array([[-12.7], [-12.8], [9.1]])
    # rat_images_y = np.array([[-8], [11.5], [11.4]])
    dots_to_plot_line = []
    for i in range(len(images_to_plot_dim_reduction)):
        x_dim_reduction = np.expand_dims(images_to_plot_dim_reduction[i][:, 0], axis=1)
        y_dim_reduction = np.expand_dims(images_to_plot_dim_reduction[i][:, 1], axis=1)
        dots_to_plot_line += [np.concatenate((rat_images_x, x_dim_reduction, rat_images_y, y_dim_reduction), axis=1)]

    return dots_to_plot_line
