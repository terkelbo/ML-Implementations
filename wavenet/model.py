import math

import pytorch_lightning as pl
import torch
from torch import nn


class WaveNet(nn.Module):
    def __init__(
        self,
        layers: int,
        k: int,
        number_of_channels: int,
        number_of_conditionals: int = 0,
    ) -> None:
        """
        Defines the models as described in the paper.
        `layers` is the number of layers in the model and `k` is filter size.

        The dilations are defined as 2^i where i is the layer number - 1.
        """
        super().__init__()

        self.layers = layers
        self.k = k
        self.number_of_channels = number_of_channels
        self.number_of_conditionals = number_of_conditionals
        self.dilations = [2**i for i in range(layers)]

        # first layer for the main dataset and the conditionals
        self.first_layers = []
        for _ in range(number_of_conditionals + 1):
            _layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=number_of_channels,
                    kernel_size=k,
                    dilation=self.dilations[0],
                )
            )
            if number_of_channels > 1:
                # if there are more than one channels, we add a 1x1 conv
                # to reduce the number of channels to 1
                _layer.append(
                    nn.Conv1d(
                        in_channels=number_of_channels,
                        out_channels=1,
                        kernel_size=1,
                    )
                )
            self.first_layers.append(_layer)

        # the skip connections for the main dataset and the conditionals
        self.skip_connections = []
        for _ in range(number_of_conditionals + 1):
            self.skip_connections.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=1,
                )
            )

        # create the layes of dilated convolutions
        self.dilated_convolutions: list[nn.Sequential] = []
        for dilation in self.dilations[1:]:
            sequential = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=number_of_channels,
                    kernel_size=k,
                    dilation=dilation,
                )
            )

            # if there are more than one channels, we add a 1x1 conv
            # to reduce the number of channels to 1
            if number_of_channels > 1:
                sequential.append(
                    nn.Conv1d(
                        in_channels=number_of_channels,
                        out_channels=1,
                        kernel_size=1,
                    )
                )
            self.dilated_convolutions.append(sequential)

        # initialize the weights with zero mean gaussian
        # with standard deviation sqrt(2/k)
        for sequential in self.dilated_convolutions:
            for layer in sequential:
                nn.init.normal_(
                    layer.weight, 0.0, math.sqrt(2 / (number_of_channels * k))
                )

        # create the final layer
        self.final_layer = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        # main dataset
        x_conv_first = self.first_layers[0](x[:, 0, :, :])
        skip = self.skip_connections[0](x[:, 0, :, :])

        x = x_conv_first + skip[:, :, -x_conv_first.shape[2] :]

        # conditionals
        for i in range(1, self.number_of_conditionals + 1):
            x_conv_first = self.first_layers[i](x[:, i, :, :])
            skip = self.skip_connections[i](x[:, i, :, :])
            x = x_conv_first + skip[:, :, -x_conv_first.shape[2] :]

        for i in range(self.layers - 1):
            # apply the dilated convolution
            x_conv = self.dilated_convolutions[i](x)

            # apply relu
            x_conv = torch.relu(x_conv)

            # apply the residual connection
            # NOTE: Not really described well in the paper
            # how to deal with the different sizes of the input and output
            # of the dilated convolution.
            # I think this is the correct way to do it. The paper says
            # that they implement it the same way as in the WaveNet paper.
            # Which is the same as this.
            x = x_conv + x[:, :, -x_conv.shape[2] :]

        # apply the final layer
        x = self.final_layer(x)

        return x


class WaveNetLightning(pl.LightningModule):
    def __init__(
        self, model: WaveNet, learning_rate: float, weight_decay: float
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(  # pylint: disable=unused-argument
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)

        # the loss used in the paper
        # mean absolute error (l2 regularization is added in the optimizer)
        loss = torch.mean(torch.abs(y - y_hat))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
