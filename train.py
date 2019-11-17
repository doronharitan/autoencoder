import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os

# todo:
#Do I have test data? ask Alex

def main():
    # ====== setting the hp parameters =====
    folder_dir, hp_parameters_dict, device = setting_parameters()
    tensorboard_writer = SummaryWriter(folder_dir)

    # ====== loading the data =========
    with np.load(os.path.join(hp_parameters_dict['dir'], hp_parameters_dict['file_name'])) as training_data:
        images = training_data['images']
        # todo, change the names of the paramters and the function
    visualizing_images(images)  # todo - change, for example visualize images

    # ====== data pre-processing ========
    images = images / hp_parameters_dict['max_pix_value']
    images_train, images_val = train_test_split(images, test_size=hp_parameters_dict['val_split'],
                                      random_state=hp_parameters_dict['seed'])
    # load the datasets to dataloader_dict while trasnforming it to float 32 tensor.
    dataloader_dict = {index: DataLoader(torch.tensor(data, device=device).float().unsqueeze(1), batch_size=hp_parameters_dict['batch_size'],
                          shuffle=True)
        for index, data in enumerate([images_train, images_val])}

    # ====== initializing the model, the loss and the optimizer function =======
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_parameters_dict['lr'])

    # ===== train the model ======
    for epoch in range(hp_parameters_dict['num_epochs']):
        train_loss, val_loss = 0.0, 0.0
        train_loss = train_model(model, dataloader_dict[0], optimizer, criterion)
        if (epoch % hp_parameters_dict['val_check_interval']) == 0:
            val_loss = test_model(model, dataloader_dict[1], criterion, epoch)
            # write to tensorboard
            train_loss_avr = train_loss/len(dataloader_dict[0])
            val_loss_avr = val_loss/len(dataloader_dict[1])
            tensorboard_writer.add_scalars('train/val loss',
                           {'train_loss': train_loss_avr, 'val loss': val_loss_avr},
                           epoch)
            #print status in console
            print('%d Epoch: Train Loss %.4f, Validation loss %.4f' %(epoch, train_loss_avr, val_loss_avr))
    tensorboard_writer.close()
    torch.save(model, folder_dir)

def train_model(model, train_images, optimizer, criterion):
    train_loss = 0.0
    model.train()
    for local_batch in train_images:
        optimizer.zero_grad()                                           # clear the gradients of all optimized variables
        loss = predection_step(model, local_batch, criterion)
        train_loss += loss.item()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights with the gradients
    return train_loss


def test_model(model, test_images, criterion, epoch):
    val_loss = 0.0
    model.eval()
    for batch_num, local_batch in enumerate(test_images):
        loss = predection_step(model, local_batch, criterion, batch_num, epoch, mode='val')
        val_loss += loss.item()
    return val_loss


def predection_step(model, dataset, criterion, batch_num=None, epoch=None, mode='train'):
    if mode != 'train':
        with torch.no_grad():
            outputs = model(dataset)
            visualizing_images(dataset.squeeze(1).cpu().detach().numpy(),
                               compared_data=outputs.squeeze(1).cpu().detach().numpy(), batch_num=batch_num, epoch=epoch)
    else:
        outputs = model(dataset)
    loss = criterion(outputs, dataset)  # what is the the square distance between each pixel in the original dataset comapre to the output picture
    return loss

if __name__ == "__main__":
    main()
