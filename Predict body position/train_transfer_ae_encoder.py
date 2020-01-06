import time
from utils_local import setting_parameters, loading_plus_preprocessing_data_with_labels, \
    train_model_predicted_position, test_model_predicted_position, save_loss_info_into_a_file
import torch
# from transfer_ae_encoder_model import Predict_body_position
# from transfer_ae_encoder_unfreezed_f1_model import Predict_body_position
from transfer_ae_only_encoder_model import Predict_body_position
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import os

def main():
    # ====== setting the hp parameters =====
    folder_dir, hp_parameters_dict = setting_parameters()
    tensorboard_writer = SummaryWriter(folder_dir)
    torch.manual_seed(hp_parameters_dict['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataloader_dict = loading_plus_preprocessing_data_with_labels(hp_parameters_dict)
    if hp_parameters_dict['save_latent_space']:
        dataloader_dict['all_data'] = loading_plus_preprocessing_data_with_labels(hp_parameters_dict, suffle=False)

    # ====== visualize validation data =======
    img_to_plot = dataloader_dict[1].dataset[:16][0]
    save_image(img_to_plot, os.path.join(folder_dir, './Images/row val data.png'))

    # ====== initializing the model, the loss and the optimizer function =======
    model = Predict_body_position(hp_parameters_dict['dir_model']).to(hp_parameters_dict['device'])
    encoder_mode = 'predict_position'
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_parameters_dict['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    # ===== train the model ======
    len_train_dataset = dataloader_dict[0].dataset.__len__()
    len_val_dataset = dataloader_dict[1].dataset.__len__()
    save_images_folder = os.path.join(folder_dir, 'Images')
    for epoch in range(hp_parameters_dict['num_epochs']):
        start_epoch = time.time()
        if hp_parameters_dict['save_latent_space']:
            train_loss = train_model_predicted_position(model, dataloader_dict[0], optimizer, criterion, epoch,
                                folder_dir, hp_parameters_dict['save_latent_space'], dataloader_dict['all_data'],
                                                        transfer_learning=True)
        else:
            train_loss = train_model_predicted_position(model, dataloader_dict[0], optimizer, criterion, epoch,
                                    folder_dir, hp_parameters_dict['save_latent_space'])
        if (epoch % hp_parameters_dict['val_check_interval']) == 0:
            val_loss = test_model_predicted_position(model, dataloader_dict[1], criterion, folder_dir, epoch)
            scheduler.step()
            # write to tensorboard
            train_loss_avr = train_loss / len_train_dataset
            val_loss_avr = val_loss / len_val_dataset
            tensorboard_writer.add_scalars('train/val loss',
                                           {'train_loss': train_loss_avr, 'val loss': val_loss_avr},
                                           epoch)
            save_loss_info_into_a_file(train_loss_avr, val_loss_avr, folder_dir, epoch)
            end_epoch = time.time()
            # print status in console
            print('{} Epoch: Train Loss {:.5f}, Validation loss {:.5f} time {:.2f}, lr {:.8f}'
                  .format(epoch, train_loss_avr,  val_loss_avr, end_epoch - start_epoch,
                          scheduler.get_lr()[0]))
    save_model_dir = os.path.join(folder_dir, 'model check points')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    torch.save(model, os.path.join(save_model_dir, 'model_{}_epoch.pth.tar'.format(epoch)))
    tensorboard_writer.close()


if __name__ == "__main__":
    main()
