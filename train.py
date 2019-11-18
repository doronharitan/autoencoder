import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from autoencoder_model import Autoencoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    # ====== setting the hp parameters =====
    folder_dir, hp_parameters_dict, device = setting_parameters()
    tensorboard_writer = SummaryWriter(folder_dir)
    torch.manual_seed(hp_parameters_dict['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ====== loading the data =========
    with np.load(os.path.join(hp_parameters_dict['dir'], hp_parameters_dict['file_name'])) as training_data:
        images = training_data['images']
    # visualizing_images(images, folder_dir)

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

    start = time.time()
    # ===== train the model ======
    output_stack = 0          # would be use later to visualize the images
    for epoch in range(hp_parameters_dict['num_epochs']):
        train_loss, loss_reg = train_model(model, dataloader_dict[0], optimizer, criterion)
        if (epoch % hp_parameters_dict['val_check_interval']) == 0:
            val_loss, output_stack = test_model(model, dataloader_dict[1], criterion, output_stack)
            # write to tensorboard
            train_loss_avr = train_loss/len(dataloader_dict[0])
            loss_reg_avr = loss_reg/ len(dataloader_dict[0])
            val_loss_avr = val_loss/len(dataloader_dict[1])
            tensorboard_writer.add_scalars('train/val loss',
                           {'train_loss': train_loss_avr, 'val loss': val_loss_avr},
                           epoch)
            #print status in console
            print('%d Epoch: Train Loss %.4f, Validation loss %.4f, reg loss %.4f'
                  %(epoch, train_loss_avr, val_loss_avr, loss_reg_avr))
    start_1 = time.time()
    visualizing_images(images_val, folder_dir,
                       compared_data=output_stack.squeeze(1).cpu().detach().numpy(), batch_size=hp_parameters_dict['batch_size'],
                       epoch_interval = hp_parameters_dict['val_check_interval'])
    end_1 = time.time()
    print(end_1 - start_1, 'figures')
    tensorboard_writer.close()
    torch.save(model, os.path.join(folder_dir, 'model.pth.tar'))
    end = time.time()
    print(end-start)


def train_model(model, train_images, optimizer, criterion):
    train_loss, loss_reg = 0.0, 0.0
    model.train()
    for local_batch in train_images:
        optimizer.zero_grad()                                           # clear the gradients of all optimized variables
        loss, _ = predection_step(model, local_batch, criterion)
        # L1_regularization_FC_1 = torch.cat([x.view(-1) for x in model.fc_1.parameters()])
        # lambda_1 = 0.01
        # loss_reg += lambda_1 * torch.norm(L1_regularization_FC_1,1)
        train_loss += loss.item()
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights with the gradients
    return train_loss, loss_reg


def test_model(model, test_images, criterion, output_stack):
    val_loss = 0.0
    model.eval()
    for batch_num,local_batch in enumerate(test_images):
        loss, output = predection_step(model, local_batch, criterion, mode='val')
        val_loss += loss.item()
        if type(output_stack) == int and batch_num == 0:
            output_stack = output
        else:
            output_stack = torch.cat((output_stack, output))
    return val_loss, output_stack


def predection_step(model, dataset, criterion, mode='train'):
    if mode != 'train':
        with torch.no_grad():
            outputs = model(dataset)
    else:
        outputs = model(dataset)
    loss = criterion(outputs, dataset)  # what is the the square distance between each pixel in the original dataset comapre to the output picture
    return loss, outputs

if __name__ == "__main__":
    main()
