from torch import nn


class DimReductionDecoder(nn.Module):
    def __init__(self, latent_space_dim):
        super(DimReductionDecoder, self).__init__()
        # the size of the input is (1,50,50)
        self.fc = nn.Sequential(
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
        batch_size = x.shape[0]
        fc_output = self.fc(x)
        decoder_input = fc_output.view(batch_size, 128, 9, 9)
        decoder_output = self.decoder(decoder_input)
        return decoder_output


    def forward_latent_space(self, x, fc2_mode):
        if fc2_mode:
            fc_output = self.fc(x)
            return fc_output
        else:
            return x