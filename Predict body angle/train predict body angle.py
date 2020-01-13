import time
from utils_local import setting_parameters, loading_plus_preprocessing_data_with_labels, \
    train_model_with_labels, test_model_with_labels, save_loss_info_into_a_file, create_folder_if_needed
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import os
from models.predict_body_angle_model import Predict_body_angle


def main():
    # ====== setting the hp parameters =====
    folder_dir, args = setting_parameters()
    tensorboard_writer = SummaryWriter(folder_dir)
    # ===== set a seed for the run ======
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ====== load dataset ======
    dataloader_dict = loading_plus_preprocessing_data_with_labels(args, split_to_train_val=True)
    # ===== if args['save_latent_space'] is True create a dataloader of all of the data =======
    if args['save_latent_space']:
        dataloader_dict['all_data'] = loading_plus_preprocessing_data_with_labels(args)

    # ====== visualize validation data =======
    img_to_plot = dataloader_dict[1].dataset.dataset.tensors[1][:16]
    save_images_path = os.path.join(folder_dir, 'Images')
    create_folder_if_needed(save_images_path)
    save_image(img_to_plot, os.path.join(save_images_path, 'row val data.png'))

    # ====== initializing the model, the loss and the optimizer function =======
    model = Predict_body_angle(args['latent_space_dim']).to(args['device'])
    if args['load_checkpoint']:
        checkpoint = torch.load(os.path.join(args['checkpoint_path'], args['checkpoint_to_load']))
        model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    # ===== train the model ======
    save_model_dir = os.path.join(folder_dir, 'model check points')
    create_folder_if_needed(save_model_dir)
    for epoch in range(args['epochs']):
        start_epoch = time.time()
        checkpoint_latent_space_interval = args['checkpoint_latent_space_interval'] if args['save_latent_space'] else None
        train_loss = train_model_with_labels(model, dataloader_dict, optimizer, criterion, epoch,
                                             args['save_latent_space'], folder_dir, checkpoint_latent_space_interval)
        if (epoch % args['val_check_interval']) == 0:
            val_loss = test_model_with_labels(model, dataloader_dict[1], criterion, epoch, None)
            scheduler.step()
            # ====== write to tensorboard =======
            train_loss_avr = train_loss / len(dataloader_dict[0])
            val_loss_avr = val_loss / len(dataloader_dict[1])
            tensorboard_writer.add_scalars('train/val loss',
                                           {'train_loss': train_loss_avr, 'val loss': val_loss_avr},
                                           epoch)
            save_loss_info_into_a_file(train_loss_avr, val_loss_avr, folder_dir, epoch,  scheduler.get_lr()[0])
            end_epoch = time.time()
            # ====== print status in console =======
            print('{} Epoch: Train Loss {:.5f}, Validation loss {:.5f} time {:.2f}, lr {:.8f}'
                  .format(epoch, train_loss_avr,  val_loss_avr, end_epoch - start_epoch,
                          scheduler.get_lr()[0]))
            if args['save_model_checkpoints']:
                if (epoch % args['checkpoint_interval']) == 0:
                    model_dict = {'model_state_dict': model.state_dict(), 'model_name': model._get_name()}
                    torch.save(model_dict, os.path.join(save_model_dir, 'model_{}_epoch.pth.tar'.format(epoch)))
    model_dict = {'model_state_dict': model.state_dict(), 'model_name': model._get_name()}
    torch.save(model_dict, os.path.join(save_model_dir, 'model_{}_epoch.pth.tar'.format(epoch)))
    tensorboard_writer.close()


if __name__ == "__main__":
    main()
