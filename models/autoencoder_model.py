from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, latent_space_dim):
        super(Autoencoder, self).__init__()
        # the size of the input is (1,50,50)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1),  # B output (64,48,48)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # B output (64, 24, 24)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # B output (64,22,22)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # B output (64,11,11)
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # B output (128,9,9) #change to 128/256
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(128*9*9, latent_space_dim),  #
            nn.BatchNorm1d(latent_space_dim),
            nn.ReLU(True)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(latent_space_dim, 128*9*9),
            nn.BatchNorm1d(128*9*9),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(

                        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),  # B output (8,11,11)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # B output (16,22,22)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),  # B output (1,24,24) #change to higher?
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # B output (16,48,48)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1),  # B output (16,50,50)
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        batch_size, num_filters, w, h = encoder_output.shape
        fc_input = encoder_output.view(batch_size, num_filters * h * w)
        fc_1_output = self.fc_1(fc_input)
        fc_2_output = self.fc_2(fc_1_output)
        decoder_input = fc_2_output.view(batch_size, num_filters, h, w)
        decoder_output = self.decoder(decoder_input)
        return decoder_output

    def forward_latent_space(self, x, fc2_mode):
        encoder_output = self.encoder(x)
        batch_size, num_filters, w, h = encoder_output.shape
        fc_input = encoder_output.view(batch_size, num_filters * h * w)
        fc_1_output = self.fc_1(fc_input)
        if fc2_mode:
            fc_2_output = self.fc_2(fc_1_output)
            return fc_2_output
        else:
            return fc_1_output



