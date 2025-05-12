import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, color

def save_segmentation_results(image_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        image = io.imread(image_path)
        gray_image = color.rgb2gray(image)
        threshold = 0.5
        binary_mask = gray_image > threshold
        segments = segmentation.clear_border(binary_mask)
        high_intensity_regions = gray_image > 0.6
        io.imsave(os.path.join(output_folder, f'high_intensity_region_{idx}.jpg'), high_intensity_regions.astype(np.uint8) * 255)

image_paths = ['/content/drive/MyDrive/mlo/IMG-0001-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0002-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0003-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0004-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0005-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0006-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0007-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0008-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0009-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0010-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0011-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0012-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0013-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0014-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0015-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0016-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0017-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0018-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0019-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0020-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0021-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0022-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0023-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0024-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0025-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0026-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0027-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0028-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0029-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0030-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0031-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0032-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0033-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0034-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0035-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0036-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0037-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0038-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0039-00001.jpg', '/content/drive/MyDrive/mlo/IMG-0040-00001.jpg' ]

output_folder = '/content/drive/MyDrive/MLO_ROI_HII'

save_segmentation_results(image_paths, output_folder)