import torch

if __name__ == "__main__":
    point_A = torch.tensor([2, 7], dtype=torch.float32)
    point_B = torch.tensor([6, 4], dtype=torch.float32)
    print(
        "The distance: of travel absolute values: ",
        (point_A - point_B).abs().sum(),
    )

    print("The distance of travelling diagonally:", (point_A - point_B).norm())
