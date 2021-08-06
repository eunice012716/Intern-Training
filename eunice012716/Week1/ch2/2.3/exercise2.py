import torch

if __name__ == "__main__":
    A = torch.arange(15).reshape(3, 5)
    B = torch.ones(3, 5)
    A_T = A.T
    B_T = B.T
    C = (A + B).T
    print("AT + BT = (A + B)T: \n", C == A_T + B_T)
