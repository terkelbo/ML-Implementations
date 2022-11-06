import math

import pytorch_lightning as pl
import torch
from torch import nn


class WaveNet(nn.Module):
    def __init__(self, layers: int, k: int) -> None:
        """
        Defines the models as described in the paper.
        `layers` is the number of layers in the model and `k` is filter size.

        The dilations are defined as 2^i where i is the layer number - 1.
        """
        super().__init__()

        self.layers = layers
        self.k = k

        self.dilations = [2**i for i in range(layers)]

        # create the layes of dilated convolutions
        self.dilated_convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    dilation=dilation,
                    padding=0,
                )
                for dilation in self.dilations
            ]
        )

        # initialize the weights with zero mean gaussian
        # with standard deviation sqrt(2/k)
        for layer in self.dilated_convolutions:
            nn.init.normal_(layer.weight, 0.0, math.sqrt(2 / k))

        # create the residual connections
        self.residual_connections = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=1) for _ in range(layers)]
        )

        # create the final layer
        self.final_layer = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        for i in range(self.layers):
            # apply the dilated convolution
            x = self.dilated_convolutions[i](x)

            # apply relu
            x = torch.relu(x)

            # apply the residual connection
            x = self.residual_connections[i](x) + x

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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
