import torch

if __name__ == "__main__":
    pointA = torch.tensor(
        [1, 1], dtype=torch.float32
    )  # first element is avenue, second is street
    pointB = torch.tensor([3, 5], dtype=torch.float32)
    print(
        "from 1nd street, 1nd Avenue to 3nd street, 5nd Avenue distance you need to cover in terms is: ",
        (pointA - pointB).abs().sum(),
    )
    print()

    print("If I can fly, I can travel diagonally owo")
    print()

    print("If travel diagonally, distance will be: ", (pointA - pointB).norm())
