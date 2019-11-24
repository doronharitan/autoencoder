import time
import os
import json
import torch


def create_hp_dict():
    DIR =  r'C:\Users\Doron\Desktop\Autoencoder\train data' # the directory where you can find the train data r'./data'
    FILE_NAME = 'ims_for_doron.npz'                         # The name of the file containing the data
    BATCH_SIZE = 64                                         # 'mini-batch size
    SEED = 42                                               #
    VAL_SPLIT = 0.2                                         # the split of train and val
    NUM_EPOCHS = 500                                        # number of total epochs
    LR = 1e-3                                               # initial learning rate
    OPEN_NEW_FOLDER_STATUS = 'True'                        # open a new folder for saving the run info, if false the info would be saved in the project dir, if debug the info would be saved in debug folder(default:True)
    MAX_PIX_VALUE = 255.0                                   # max pixsel value, would be used for the rescaling of the images
    VAL_CHECK_INTERBAL = 5                                  # Epochs interval between running validation test
    RUN_GOAL = '50 pix, SOTA network, no sparse.'
    PLOT_ALL_EPOCH_OUTPUTS = False                          # Do we want to plot figuers for al of the epochs or just the last one
    hp_parameters = {'dir': DIR , 'file_name': FILE_NAME,
                      'batch_size': BATCH_SIZE, 'seed': SEED, 'val_split': VAL_SPLIT,
                     'num_epochs': NUM_EPOCHS, 'lr': LR, 'open_new_folder_status': OPEN_NEW_FOLDER_STATUS, 'max_pix_value': MAX_PIX_VALUE,
                     'val_check_interval': VAL_CHECK_INTERBAL, 'run_goal': RUN_GOAL, 'plot_all_epoch_outputs': PLOT_ALL_EPOCH_OUTPUTS}
    return hp_parameters


def setting_parameters():
    hp_parameters_dict = create_hp_dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if hp_parameters_dict['open_new_folder_status'] != 'False':
        folder_dir = open_new_folder(hp_parameters_dict['open_new_folder_status'])
    else:
        folder_dir = os.getcwd()  #run in colab r'./'
    save_setting_info(hp_parameters_dict, device, folder_dir)
    print('The HP parameters that were used in this run are:', hp_parameters_dict)
    print('The code run on:', device)
    print('The folder all of the data is saved on is:', folder_dir)
    return folder_dir, hp_parameters_dict, device


def open_new_folder(open_folder_status):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug'
    folder_dir = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
        os.makedirs(os.path.join(folder_dir,'Images'))
    return folder_dir


def save_setting_info(hp_parameters, device, folder_dir):
    setting_file_name = os.path.join(folder_dir, 'setting_info.txt')
    with open(setting_file_name, 'w') as f:
        json.dump(hp_parameters, f, indent=2)
        json.dump('\n' + str(device), f)


def save_loss_info_into_a_file(train_loss,val_loss, folder_dir, epoch):
    file_name = os.path.join(folder_dir, 'loss_per_epoch.txt')
    with open(file_name,'a+') as f:
        f.write('%d Epoch: Train Loss %.4f, Validation loss %.4f\n'
                  %(epoch, train_loss, val_loss))