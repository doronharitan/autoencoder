from torch import nn


class Predict_body_angle(nn.Module):
    def __init__(self, latent_space_dim):
        super(Predict_body_angle, self).__init__()
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
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # B output (128,9,9)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(128 * 9 * 9, latent_space_dim),  #
            nn.BatchNorm1d(latent_space_dim),
            nn.ReLU(True)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(latent_space_dim, 2))

    def forward(self, x):
        encoder_output = self.encoder(x)
        batch_size, num_filters, w, h = encoder_output.shape
        fc_input = encoder_output.view(batch_size, num_filters * h * w)
        fc_1_output = self.fc_1(fc_input)
        fc_2_output = self.fc_2(fc_1_output)
        return fc_2_output

    def forward_latent_space(self, x, fc2_mode=False):
        encoder_output = self.encoder(x)
        batch_size, num_filters, w, h = encoder_output.shape
        fc_input = encoder_output.view(batch_size, num_filters * h * w)
        fc_1_output = self.fc_1(fc_input)
        return fc_1_output
