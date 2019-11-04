import torch
from torch import nn


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # the size of the input is (50,50)
        # Encoder #chnage the numbers to be changeable?
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0), #B output (16,48,48)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),   #B output (16, 24, 24)
            nn.Conv2d(16, 8, kernel_size=3, stride=3, padding=0), #B output (8,8,8)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2) #B output (8,4,4)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2), #B output (8,9,9)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=3, padding=1),  # B output (16,25,25)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),  # B output (1,50,50)
            nn.ReLU(True)
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

