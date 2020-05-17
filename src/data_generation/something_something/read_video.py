import cv2
import matplotlib.pyplot as plt
import os.path

SAMPLE = 5


path = '/home/tejask/Downloads/20bn-something-something-v2/23.webm'
assert os.path.exists(path), f'The path does not exist: {path}'

cap = cv2.VideoCapture(path)
ret, frame = cap.read()

index = 0
while ret:
    if index % SAMPLE == 0:
        plt.imshow(frame)
        plt.show()

    index += 1
    ret, frame = cap.read()

print(index)
