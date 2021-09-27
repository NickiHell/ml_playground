from scipy.misc import derivative

if __name__ == "__main__":
    def f(x):
        return x**2


    print(derivative(f, 1.0, dx=1e-6))
