import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, color

image_path = '/content/drive/MyDrive/mlo/IMG-0001-00001.jpg'
image = io.imread(image_path)

gray_image = color.rgb2gray(image)

threshold = 0.5

binary_mask = gray_image > threshold

segments = segmentation.clear_border(binary_mask)

high_intensity_regions = gray_image > 0.6

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(image)
ax[0].set_title('Original Image')

ax[1].imshow(segments, cmap='gray')
ax[1].set_title('Segmented Image')

ax[2].imshow(high_intensity_regions, cmap='gray')
ax[2].set_title('High Intensity Regions')

for a in ax:
    a.axis('off')

plt.show()