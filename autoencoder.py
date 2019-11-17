import torch
from torch import nn


class Autoencoder(nn.Module):  # todo how to write the name of the class correct
    def __init__(self):
        super(Autoencoder, self).__init__()
        # the size of the input is (50,50)
        # Encoder #chnage the numbers to be changeable?
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),  # B output (32,48,48)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # B output (16, 24, 24)
            nn.Conv2d(32, 16, kernel_size=3, stride=1),  # B output (8,22,22)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # B output (8,11,11)
            nn.Conv2d(16, 8, kernel_size=3, stride=1),  # B output (8,9,9)
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )                                           #todo - try to add FC, maybe comapre the preformance using ROC

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=1),  # B output (8,11,11)
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),  # B output (16,22,22)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1),  # B output (1,24,24)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # B output (16,48,48)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1),  # B output (16,50,50)
            # nn.ReLU(True),
            nn.Tanh()               #Todo Check why to use this and not relu
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

