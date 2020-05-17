import numpy as np

LOW = -4
HIGH = 4
NUM_SAMPLES = 500


def tanh_approx(x: np.ndarray) -> np.ndarray:
    return np.clip(x * (27 + np.square(x)) / (27 + 9 * np.square(x)), a_min=-1, a_max=1)


def approx_tanh():
    x = np.linspace(start=LOW, stop=HIGH, endpoint=True, num=NUM_SAMPLES)
    
    tanh = np.tanh(x)
    rmse = np.sqrt(np.average(np.square(tanh_approx(x) - tanh)))
    print('Tanh RMSE: {0:.4f}'.format(rmse))


def approx_sigmoid():
    x = np.linspace(start=LOW, stop=HIGH, endpoint=True, num=NUM_SAMPLES)
   
    sigmoid_approx = 0.5 * (tanh_approx(0.5 * x) + 1)


    sigmoid = 1.0 / (1.0 + np.exp(-x))
    # sigmoid_approx = 0.5 * (x / (1.0 + np.abs(x))) + 0.5

    rmse = np.sqrt(np.average(np.square(sigmoid_approx - sigmoid)))
    print('Sigmoid RMSE: {0:.4f}'.format(rmse))


if __name__ == '__main__':
    approx_tanh()
    approx_sigmoid()
