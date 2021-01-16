import matplotlib.pyplot as plt


#probs = [0.837756, 0.85858625, 0.8738437, 0.8876629, 0.90488964, 0.9270233, \
#         0.94432306, 0.95770884, 0.9679855, 0.97582805, 0.98178554, 0.98629534, \
#         0.9897002, 0.9922658, 0.9941962, 0.99564683, 0.99673605, 0.9975534, \
#         0.9981665, 0.9986261 ]  # FORDA
probs = [0.6840883, 0.67822635, 0.62649214, 0.57098734, 0.5232515, 0.47774157, \
         0.4325982, 0.38854706]  # PEN DIGITS


with plt.style.context('seaborn-ticks'):
    fig, ax = plt.subplots()

    xs = list(range(1, len(probs) + 1))
    ax.plot(xs, probs, marker='o')

    ax.set_xlabel('Step')
    ax.set_ylabel('Predicted Max Probability')
    ax.set_title('Predicted Max Probability for Various Step Sizes')

    plt.show()


