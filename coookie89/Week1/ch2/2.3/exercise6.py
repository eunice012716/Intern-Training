# 6. Run A / A.sum(axis=1) and see what happens. Can you analyze the reason?

import torch

if __name__ == "__main__":
    A = torch.arange(20).reshape(5, 4)
    print("A==>\n", A, "\n")
    print("A.sum(axis=1)==>\n", A.sum(axis=1), "\n")
    # print("\nA / A.sum(axis=1)==>\n",A / A.sum(axis=1))

    print("因為sum(axis=1)是把row的值加在一起,會產生1*5的matrix,沒辦法跟A(5*4)做除法")

    print("\nA==>\n", A, "\n")
    print("A.sum(axis=1)==>\n", A.sum(axis=0), "\n")
    print("A / A.sum(axis=1)==>\n", A / A.sum(axis=0), "\n")

    print("如果改成sum(axis=0),把column的值加在一起,會產生1*4的matrix,就沒有問題了！", "\n")
