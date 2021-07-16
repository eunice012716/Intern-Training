import pandas as pd
import numpy as np


def getRandomNaDataFrame(dataFrame: pd.DataFrame) -> pd.DataFrame:
    return dataFrame.mask(np.random.random(dataFrame.shape) < 0.2)


def processData(dataFrame: pd.DataFrame) -> pd.DataFrame:
    naCount = dataFrame.isnull().sum(axis=0).tolist()
    mostNaCol = naCount.index(max(naCount))
    result = dataFrame.drop(dataFrame.columns[mostNaCol], axis=1)
    return result.fillna(result.mean())


if __name__ == "__main__":
    with open("example_dataset.csv", "r") as f:
        dataFrame = pd.read_csv(f)
        inputs, outputs = (
            getRandomNaDataFrame(dataFrame.iloc[:, :-1]),
            dataFrame.iloc[:, -1],
        )
        print("raw inputs:\n", inputs.head())
        print()

        inputs = processData(inputs)
        print("processed inputs:\n", inputs.head())
        print()
