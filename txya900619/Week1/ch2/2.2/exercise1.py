import pandas as pd
import numpy as np


def get_random_na_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """random insert na to dataframe"""
    na_probability = 0.2
    return dataframe.mask(np.random.random(dataframe.shape) < na_probability)


def process_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """delete column that have most na, and fill na with mean"""

    na_count = dataframe.isnull().sum(axis=0).tolist()
    most_na_col = na_count.index(max(na_count))
    result = dataframe.drop(dataframe.columns[most_na_col], axis=1)
    return result.fillna(result.mean())


if __name__ == "__main__":
    with open("example_dataset.csv", "r", encoding="utf-8") as example_dataset:
        dataframe = pd.read_csv(example_dataset)
        inputs, outputs = (
            get_random_na_dataframe(dataframe.iloc[:, :-1]),
            dataframe.iloc[:, -1],
        )
        print("raw inputs:\n", inputs.head())
        print()

        inputs = process_data(inputs)
        print("processed inputs:\n", inputs.head())
        print()
