import pandas as pd
import numpy as np


def get_random_na_dataframe(dataFrame: pd.DataFrame) -> pd.DataFrame:
    # random insert na to dataframe
    na_probability = 0.2
    return dataFrame.mask(np.random.random(dataFrame.shape) < na_probability)


def process_data(dataFrame: pd.DataFrame) -> pd.DataFrame:
    # delete column that have most na, and fill na with mean

    naCount = dataFrame.isnull().sum(axis=0).tolist()
    mostNaCol = naCount.index(max(naCount))
    result = dataFrame.drop(dataFrame.columns[mostNaCol], axis=1)
    return result.fillna(result.mean())


if __name__ == "__main__":
    with open("example_dataset.csv", "r", encoding="utf-8") as example_dataset:
        dataFrame = pd.read_csv(example_dataset)
        inputs, outputs = (
            get_random_na_dataframe(dataFrame.iloc[:, :-1]),
            dataFrame.iloc[:, -1],
        )
        print("raw inputs:\n", inputs.head())
        print()

        inputs = process_data(inputs)
        print("processed inputs:\n", inputs.head())
        print()
