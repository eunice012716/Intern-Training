import os
import hashlib
import tarfile
import zipfile
import requests

import torch
import numpy as np
import pandas as pd
from torch import nn
from d2l import torch as d2l

DATA_HUB = dict()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"

DATA_HUB["kaggle_house_train"] = (
    DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce",
)

DATA_HUB["kaggle_house_test"] = (
    DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
)


def download(name, cache_dir=os.path.join("..", "data")):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    file_name = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(file_name):
        sha1 = hashlib.sha1()
        with open(file_name, "rb", encoding="UTF-8") as input_file:
            while True:
                data = input_file.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return file_name  # Hit cache
    print(f"Downloading {file_name} from {url}...")
    request_get = requests.get(url, stream=True, verify=True)
    with open(file_name, "wb", encoding="UTF-8") as output_file:
        output_file.write(request_get.content)
    return file_name


def download_and_extract_zip_file(name, folder=None):
    """Download and extract a zip/tar file."""
    file_name = download(name)
    base_dir = os.path.dirname(file_name)
    data_dir, ext = os.path.splitext(file_name)
    if ext == ".zip":
        fp = zipfile.ZipFile(file_name, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(file_name, "r")
    else:
        assert False, "Only zip/tar files can be extracted."
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all(DATA_HUB):
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    """
    This leads to the following root-mean-squared-error between
    the logarithm of the predicted price and the logarithm of the label price
    """
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(nn.MSELoss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(
    net,
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    """
    Defining the Training Loop
    """
    train_log_rmse, test_log_rmse = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            train_loss = nn.MSELoss(net(X), y)
            train_loss.backward()
            optimizer.step()
        train_log_rmse.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_log_rmse.append(log_rmse(net, test_features, test_labels))
    return train_log_rmse, test_log_rmse


def get_data_in_k_fold_cross_validation_procedure(k, i, X, y):
    """
    returns the  i-th  fold of the data in a  K-fold cross-validation procedure.
    It proceeds by slicing out the  i-th  segment as validation data and returning the rest as training data.
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(
    k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size
):
    """
    The training and verification error averages are returned when we train  K  times in the  K -fold cross-validation.
    """
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_data_in_k_fold_cross_validation_procedure(
            k, i, X_train, y_train
        )
        net = get_net()
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size
        )
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(
                list(range(1, num_epochs + 1)),
                [train_ls, valid_ls],
                xlabel="epoch",
                ylabel="rmse",
                xlim=[1, num_epochs],
                legend=["train", "valid"],
                yscale="log",
            )
        print(
            f"fold {i + 1}, train log rmse {float(train_ls[-1]):f}, "
            f"valid log rmse {float(valid_ls[-1]):f}"
        )
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(
    train_features,
    test_feature,
    train_labels,
    test_data,
    num_epochs,
    lr,
    weight_decay,
    batch_size,
):
    """
    train and predict on Kaggle
    """
    net = get_net()
    train_ls, _ = train(
        net,
        train_features,
        train_labels,
        None,
        None,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
    d2l.plot(
        np.arange(1, num_epochs + 1),
        [train_ls],
        xlabel="epoch",
        ylabel="log rmse",
        xlim=[1, num_epochs],
        yscale="log",
    )
    print(f"train log rmse {float(train_ls[-1]):f}")
    # Apply the network to the test set
    preds = net(test_features).detach().numpy()
    # Reformat it to export to Kaggle
    test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    train_data = pd.read_csv(download("kaggle_house_train"))
    test_data = pd.read_csv(download("kaggle_house_test"))

    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # If test data were inaccessible, mean and standard deviation could be
    # calculated from training data
    numeric_features = all_features.dtypes[
        all_features.dtypes != "object"
    ].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    # After standardizing the data all means vanish, hence we can set missing
    # values to 0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
    # creates an indicator feature for it
    all_features = pd.get_dummies(all_features, dummy_na=True)
    all_features.shape
    n_train = train_data.shape[0]
    train_features = torch.tensor(
        all_features[:n_train].values, dtype=torch.float32
    )
    test_features = torch.tensor(
        all_features[n_train:].values, dtype=torch.float32
    )
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
    )
    in_features = train_features.shape[1]

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(
        k,
        train_features,
        train_labels,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
    print(
        f"{k}-fold validation: avg train log rmse: {float(train_l):f}, "
        f"avg valid log rmse: {float(valid_l):f}"
    )

    train_and_pred(
        train_features,
        test_features,
        train_labels,
        test_data,
        num_epochs,
        lr,
        weight_decay,
        batch_size,
    )
