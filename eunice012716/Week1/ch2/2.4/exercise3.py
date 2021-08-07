if __name__ == "__main__":
    print("For example vector x = (x1, x2, x3)")
    print("f(x) = (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (1/2)")
    print()
    print(
        "the gradient of the function is \n",
        "(1/2) * (2 * x1) ** (-1/2) + (1/2) * (2 * x2) ** (-1/2) + (1/2) * (2 * x3) ** (-1/2) =",
        "x1 ** (-1/2) + x2 ** (-1/2) + x3 ** (-1/2)",
    )
