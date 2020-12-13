import numpy as np
import cv2

# # toro
# img = cv2.imread('toro.png',-1)
# print img.shape
# resized_image = cv2.resize(img, (20, 30)) 
# print resized_image.shape
# cv2.imwrite('game_toro.png',resized_image)

# tree row
img = cv2.imread('tree_row2.png', -1)
print(img.shape)
resized_image = cv2.resize(img, (30, 220))
print(resized_image.shape)
cv2.imwrite('game_tree_row2.png', resized_image)
