import cv2
import os

testB_path = '../moire_testB_dataset/images'

output_path = './output/pre'

file_list = os.listdir(testB_path)
file_list.sort()


index = 0
for frame in file_list:
    print(index)
    index += 1
    im1 = cv2.imread(os.path.join(testB_path, frame))
    im2 = cv2.imread(os.path.join(output_path, frame))

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    assert h1 == h2 and w1 == w2, "im1's shape must equal im2!"