import pytorch_lightning as pl
import torch.utils.data

from wavenet.data import apply_data_preprocessing, get_data, transform_data_to_torch
from wavenet.model import WaveNet, WaveNetLightning


def main() -> None:
    # parameters
    layers = 4
    k = 2
    receptive_field = 2 ** (layers - 1) * k
    learning_rate = 0.001
    weight_decay = 0.001

    # data
    df = get_data()
    df = apply_data_preprocessing(df)
    X, y = transform_data_to_torch(df, receptive_field=receptive_field)
    X = X.float()
    y = y.float()

    # model
    wave_net = WaveNet(layers, k)
    wave_net = wave_net.float()
    model = WaveNetLightning(
        wave_net, learning_rate=learning_rate, weight_decay=weight_decay
    )

    # train
    # make dataloader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=X.shape[0]
    )
    trainer = pl.Trainer(max_epochs=2000)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
