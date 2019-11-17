import matplotlib.pyplot as plt
import time
import os
import json
import torch

def create_hp_dict():
    DIR = r'C:\Users\Doron\Desktop\Autoencoder\train data' # the directory where you can find the train data
    FILE_NAME = 'rat_unaug_db_50pix.npz'                   # The name of the file containing the data
    BATCH_SIZE = 64                                        # 'mini-batch size
    SEED = 42                                              #
    VAL_SPLIT = 0.2                                        # the split of train and val
    NUM_EPOCHS = 300                                       # number of total epochs
    LR = 1e-3                                              # initial learning rate
    OPEN_NEW_FOLDER_STATUS = 'debug'                       # open a new folder for saving the run info, if false the info would be saved in the project dir, if debug the info would be saved in debug folder(default:True)
    MAX_PIX_VALUE = 255.0                                  # max pixsel value, would be used for the rescaling of the images
    VAL_CHECK_INTERBAL = 5                                 # Epochs interval between running validation test
    hp_parameters = {'dir': DIR , 'file_name': FILE_NAME,
                      'batch_size': BATCH_SIZE, 'seed': SEED, 'val_split': VAL_SPLIT,
                     'num_epochs': NUM_EPOCHS, 'lr': LR, 'open_new_folder_status': OPEN_NEW_FOLDER_STATUS, 'max_pix_value': MAX_PIX_VALUE,
                     'val_check_interval': VAL_CHECK_INTERBAL}
    return hp_parameters

def setting_parameters():
    hp_parameters_dict = create_hp_dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if hp_parameters_dict['open_new_folder_status'] != 'False':
        folder_dir = open_new_folder(hp_parameters_dict['open_new_folder_status'])
    else:
        folder_dir = os.getcwd()
    save_setting_info(hp_parameters_dict, device, folder_dir)
    print('The HP parameters that were used in this run are:', hp_parameters_dict)
    print('The code run on:', device)
    print('The folder all of the data is saved on is:', folder_dir)
    return folder_dir, hp_parameters_dict, device

def open_new_folder(open_folder_status):
    if open_folder_status == 'True':
        folder_name = time.strftime("%Y%m%d-%H%M%S")
    else:
        folder_name = 'debug' #todo add when it is False
    folder_dir = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir

def save_setting_info(hp_parameters, device, folder_dir):
    setting_file_name = os.path.join(folder_dir, 'setting_info.txt')
    with open(setting_file_name, 'w') as f:
        json.dump(hp_parameters, f, indent=2)
        json.dump('\n' + str(device), f)

def visualizing_images(images, compared_data=[], batch_num=None, epoch=None):
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10), sharey=True)
    data_len = len(images)
    for row in range(nrows):
        if (row + 1) % 2 == 0 and len(compared_data) > 0:
            for col in range(ncols):
                ax[row,col].imshow(compared_data[col], cmap='gray')
                ax[row, col].set_title('After ' + str(col))
        elif (row + 1) % 2 == 0:
            for col in range(ncols):
                ax[row,col].imshow(images[data_len - col - 1], cmap='gray')
                ax[row, col].set_title('Before ' + str(data_len - col - 1))
        else:
            for col in range(ncols):
                ax[row,col].imshow(images[col], cmap='gray')
                ax[row, col].set_title('Before ' + str(col))
    if len(compared_data) == 0:
        plt.savefig('untouched_data', bbox_inches="tight")
    else:
        plt.savefig('before_vs_after_%dbatch_%depoch'%(batch_num, epoch), bbox_inches="tight")
        plt.close('all')
