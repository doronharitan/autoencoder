from torch import nn
import torch

class Predict_body_position(nn.Module):  # Todo change output sizes
    def __init__(self, model_path):
        super(Predict_body_position, self).__init__()
        self.ae_encoder = Autoencoder(model_path)
        # changing the last FC layer to an output with the size we need. this layer is un freezed
        self.fc_2 = nn.Linear(128*9*9, 2)


    def forward(self, x):  #todo make it prettier
        encoder_output = self.ae_encoder(x)
        # unfreezed layer
        fc_2_output = self.fc_2(encoder_output)
        return fc_2_output

class Autoencoder(nn.Module):  # Todo change output sizes
    def __init__(self, model_path):
        super(Autoencoder, self).__init__()
        self.ae_model = torch.load(model_path)
        # freezing all of the layers.
        for param in self.ae_model.parameters():
            param.requires_grad = False


    def forward(self, x):
        # freezed layers
        encoder_output = self.ae_model.encoder(x)
        batch_size, num_filters, w, h = encoder_output.shape
        encoder_output = encoder_output.view(batch_size, num_filters * h * w)
        return encoder_output