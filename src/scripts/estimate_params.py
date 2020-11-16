
n = 2   # Input Features
d = 22  # State Size
h = 24  # Hidden Output Units
k = 10  # Number of classes
r = 4   # Stop output hidden units

embedding = n * d + d
rnn = (2 * d)**2 + 2 * d
merge = 2 * (d * d) + d
output = (h * d + h) + (k * h + k)
pool = 2 * d + 1
stop_output = (d * r + r) + (r + 1)

total = embedding + rnn + merge + pool + stop_output + output

print('Total Parameters: {0}'.format(total))

