import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU()

        # Spatial dropout to zero entire channels
        self.dropout2d = nn.Dropout2d(dropout)

        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)

        x = self.dropout2d(x)

        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(input_channel, out_channel, dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Spatial dropout after pooling
        self.pool_dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        skip, x = self.conv_block(x), self.conv_block(x)  # adjust skip connections
        p = self.maxpool(x)
        p = self.pool_dropout(p)
        return skip, p


class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, dropout):
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        # Spatial dropout after concatenation
        self.dropout2d = nn.Dropout2d(dropout)
        self.conv_block = ConvBlock(output_channel * 2, output_channel, dropout)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout2d(x)
        x = self.conv_block(x)
        return x


class Unet(nn.Module):
    def __init__(self, input_channel=6, base_filters=64, dropout=0.1):
        super(Unet, self).__init__()
        # Encoders
        self.encoder_1 = Encoder(input_channel, base_filters, dropout)
        self.encoder_2 = Encoder(base_filters, base_filters * 2, dropout)
        self.encoder_3 = Encoder(base_filters * 2, base_filters * 4, dropout)
        self.encoder_4 = Encoder(base_filters * 4, base_filters * 8, dropout)

        # Bottleneck with additional spatial dropout
        self.bottleneck = nn.Sequential(
            ConvBlock(base_filters * 8, base_filters * 16, dropout),
            nn.Dropout2d(dropout)
        )

        # Decoders
        self.decoder_1 = Decoder(base_filters * 16, base_filters * 8, dropout)
        self.decoder_2 = Decoder(base_filters * 8, base_filters * 4, dropout)
        self.decoder_3 = Decoder(base_filters * 4, base_filters * 2, dropout)
        self.decoder_4 = Decoder(base_filters * 2, base_filters, dropout)

        # Final convolution (logits output)
        self.final_conv = nn.Conv2d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoding path
        s1, p1 = self.encoder_1(x)
        s2, p2 = self.encoder_2(p1)
        s3, p3 = self.encoder_3(p2)
        s4, p4 = self.encoder_4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoding path
        d4 = self.decoder_1(b, s4)
        d3 = self.decoder_2(d4, s3)
        d2 = self.decoder_3(d3, s2)
        d1 = self.decoder_4(d2, s1)

        # Output logits
        return self.final_conv(d1)
