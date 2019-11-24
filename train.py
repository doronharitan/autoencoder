import numpy as np
from utils_local import *
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from autoencoder_model import Autoencoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import os

def main():
    # ====== setting the hp parameters =====
    folder_dir, hp_parameters_dict, device = setting_parameters()
    tensorboard_writer = SummaryWriter(folder_dir)
    torch.manual_seed(hp_parameters_dict['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ====== loading the data ========= #this is the problem!added r before the string
    with np.load(os.path.join(hp_parameters_dict['dir'], hp_parameters_dict['file_name'])) as training_data:
        # images = training_data['images']
        images = training_data['arr_0']

    # ====== data pre-processing ========
    images = images / hp_parameters_dict['max_pix_value']
    images_train, images_val = train_test_split(images, test_size=hp_parameters_dict['val_split'],
                                                random_state=hp_parameters_dict['seed'])
    # load the datasets to dataloader_dict while trasnforming it to float 32 tensor.
    dataloader_dict = {index: DataLoader(torch.tensor(data, device=device).float().unsqueeze(1),
                                         batch_size=hp_parameters_dict['batch_size'],
                                         shuffle=True)
                       for index, data in enumerate([images_train, images_val])}

    # ====== visualize validation data =======
    img_to_plot = dataloader_dict['1'][:16]
    save_image(img_to_plot, os.path.join(folder_dir, './Images/row val data.png'))

    # ====== initializing the model, the loss and the optimizer function =======
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_parameters_dict['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    # ===== train the model ======
    for epoch in range(hp_parameters_dict['num_epochs']):
        start_epoch = time.time()
        train_loss = train_model(model, dataloader_dict[0], optimizer, criterion)
        if (epoch % hp_parameters_dict['val_check_interval']) == 0:
            val_loss = test_model(model, dataloader_dict[1], criterion, epoch, folder_dir)
            scheduler.step()
            # write to tensorboard
            train_loss_avr = train_loss / len(dataloader_dict[0])
            val_loss_avr = val_loss / len(dataloader_dict[1])
            tensorboard_writer.add_scalars('train/val loss',
                                           {'train_loss': train_loss_avr, 'val loss': val_loss_avr},
                                           epoch)
            # print status in console
            end_epoch = time.time()
            print('%d Epoch: Train Loss %.4f, Validation loss %.4f, time %.4f, lr %.8f'
                  % (epoch, train_loss_avr, val_loss_avr, end_epoch - start_epoch, scheduler.get_lr()[0]))
            save_loss_info_into_a_file(train_loss_avr, val_loss_avr, folder_dir, epoch)
    tensorboard_writer.close()
    torch.save(model, os.path.join(folder_dir, 'model.pth.tar'))


def train_model(model, train_images, optimizer, criterion):
    train_loss, loss_reg = 0.0, 0.0
    model.train()
    for local_batch in train_images:
        optimizer.zero_grad()                                           # clear the gradients of all optimized variables
        loss, __ = predection_step(model, local_batch, criterion)
        train_loss += loss.item()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights with the gradients
    return train_loss


def test_model(model, test_images, criterion, epoch, folder_dir):
    val_loss = 0.0
    model.eval()
    for batch_num,local_batch in enumerate(test_images):
        loss, output = predection_step(model, local_batch, criterion, mode='val')
        val_loss += loss.item()
    output_img_tensor = torch.cat((local_batch[:8], output[0:8]))
    save_image(output_img_tensor, os.path.join(folder_dir, './Images/%d_epoch.png'%(epoch)))
    return val_loss


def predection_step(model, dataset, criterion, mode='train'):
    if mode != 'train':
        with torch.no_grad():
            outputs, ___ = model(dataset)
    else:
        outputs, fc_1_output = model(dataset)
    print(criterion(outputs, dataset))
    loss = criterion(outputs, dataset)  # what is the the square distance between each pixel in the original dataset comapre to the output picture
    return loss, outputs.detach()

if __name__ == "__main__":
    main()
