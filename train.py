# import
import numpy as np
from plot import plot_data
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

# import torchvision.transforms as transforms
from torchvision import datasets

# setting hyper parameters (#hidden size, code size, optimizer, loss, epochs, path)
# Should I use the arg parser?
hp_parameters = {'dir' : r'C:\Users\Doron\Desktop\Autoencoder\train data', 'file_name' : 'rat_unaug_db_50pix.npz', 'code_size': 10, 'batch': 64,
                 'seed' : 42 , 'val_size' : 0.2 , 'test_small_size' : 0.005, 'num_epochs' : 300 , 'lr' : 1e-3}

def main():
    # set a standard random seed for reproducible results
    seed = hp_parameters['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if hp_parameters['mnist']:

    # else:
    # loading the data , rescale and divide to batches.
    with np.load(hp_parameters['dir'] + '\\' + hp_parameters['file_name'] ) as training_data:
        X_full_data = training_data['images']

    # visualizing the data - print the first 5 images and the last 5 images
    plot_data(X_full_data)

    X_full_data = X_full_data/ 255.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('This code is running on', device)

    # splitting the data to train and validation (Or Should I Use cross validation?)
    X_train, X_val = train_test_split(X_full_data, test_size=hp_parameters['val_size'], random_state=hp_parameters['seed'])
    X_train, X_test_small = train_test_split(X_train, test_size=hp_parameters['test_small_size'], random_state=hp_parameters['seed'])

    #Change the data to tensor and load the data + split it to batches
    dataloader = {index : DataLoader(torch.tensor(data,device=device).float().unsqueeze(1), batch_size=hp_parameters['batch'], shuffle=True)
                  for index,data in enumerate([X_train, X_val, X_test_small])}

    # initilazing the model
    if device.type == 'cuda':
        model = Autoencoder().cuda()
    else:
        model = Autoencoder()
    print('The model articture is:', model)
    # specify loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_parameters['lr'])
    writer = SummaryWriter()
    for epoch in range(hp_parameters['num_epochs']):
        start = time.time()
        train_loss, val_loss = 0.0, 0.0 # Do I need it?
        for data in dataloader[0]: # this is for each batch in each training set
            optimizer.zero_grad() #clear the gradients of all optimized variables
            outputs = model(data)
            loss = criterion(outputs, data) #what is the the square distance between each pixel in the original dataset comapre to the output picture
            train_loss += loss.item()
            loss.backward() #compute the gradients
            optimizer.step() #update the paramters with the gradients
        for data in dataloader[1]:
            outputs = model(data)
            loss = criterion(outputs, data) #what is the the square distance between each pixel in the original dataset comapre to the output picture
            val_loss += loss.item()
        # writer.add_scalar('train loss' , train_loss/len(dataloader[0]), epoch)
        # writer.add_scalar('val loss' , val_loss/len(dataloader[1]), epoch)
        writer.add_scalars('train/val loss', {'train_loss':  train_loss/len(dataloader[0]), 'val loss': val_loss/len(dataloader[1])}, epoch)
        end = time.time()
        if epoch % 5 == 0:
            print(epoch)
            print('train loss', train_loss/len(dataloader[0]))
            print('val loss', val_loss/len(dataloader[1]))
            print(end - start)
            print ('============')
        if epoch == hp_parameters['num_epochs']-1:
            plot_data(data.squeeze(1).cpu().detach().numpy(),compared_data=outputs.squeeze(1).cpu().detach().numpy())
    writer.close()


if __name__ == "__main__":
    main()










