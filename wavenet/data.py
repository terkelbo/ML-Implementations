import pandas as pd
import torch


def get_data() -> pd.DataFrame:
    """
    Returns the dataframe containing the data.
    """
    df = pd.read_csv("wavenet/raw_data/HistoricalPrices.csv")
    # remove leading space in column name
    df.columns = df.columns.str.strip()

    # get the Date and Close columns
    df = df[["Date", "Close"]]

    # rename the columns
    df.columns = ["ds", "y"]

    # convert the date column to datetime
    df["ds"] = pd.to_datetime(df["ds"])

    # sort by date
    df = df.sort_values("ds")

    # make a dataframe where all the dates are present
    # then left merge with the original dataframe
    # this will fill in the missing dates with NaN
    # then fill the NaN with the previous value
    df_data = pd.DataFrame(
        {"ds": pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")}
    )
    df_data = df_data.merge(df, how="left", on="ds")

    # fill the NaN with the previous value
    df_data["y"] = df_data["y"].fillna(method="ffill")

    return df_data


def apply_data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the preprocessing steps to the dataframe.
    1. Differences the data
    2. Subtracts the mean
    3. Divides by the standard deviation
    """
    df["y"] = df["y"].diff().fillna(0)
    df["y"] = (df["y"] - df["y"].mean()) / df["y"].std()
    return df


def transform_data_to_torch(
    df: pd.DataFrame, receptive_field: int, *df_conditions: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes the dataset and all the conditional datasets of the same size and transforms them into
    it into a torch tensor of shape (n, number_of_conditions + 1, 1, receptive_field).
    It pads the left side of the dataset with the amount of zeros the
    size of the receptive field.

    Returns a tuple of tensors (X, y).
    """
    # pads the left side of the dataset with the amount of zeros the
    # size of the receptive field
    padded = torch.zeros(1, receptive_field)
    padded = torch.cat((padded, torch.tensor(df["y"].values).unsqueeze(0)), dim=1)

    padded_conditions = []
    for df_condition in df_conditions:
        padded_condition = torch.zeros(1, receptive_field)
        padded_condition = torch.cat(
            (padded_condition, torch.tensor(df_condition["y"].values).unsqueeze(0)),
            dim=1,
        )
        padded_conditions.append(padded_condition)

    # create the sliding window of size receptive_field
    # and stack them into a tensor
    X = torch.stack([padded[:, i : i + receptive_field] for i in range(len(df))], dim=0)

    # create the sliding window for the conditions
    X_conditions = []
    for padded_condition in padded_conditions:
        X_condition = torch.stack(
            [padded_condition[:, i : i + receptive_field] for i in range(len(df))],
            dim=0,
        )
        X_conditions.append(X_condition)

    # stack the conditions with the main dataset
    if X_conditions:
        X = torch.stack([X] + X_conditions, dim=1)
    else:
        X = X.unsqueeze(1)

    # create the target tensor
    y = torch.tensor(df["y"].values)
    return X, y
