from rasterio import open
from glob import glob
import numpy as np


path_to_images = "C:\\Users\\laptomon\\Documents\\python_keras\\zindi\\farming\\data\\train_large\\"
image_files = glob(path_to_images + "/*.jp2")
band_count = 13
file_band_means = np.zeros((len(image_files), band_count))
file_band_stds = np.zeros((len(image_files), band_count))

for idx, image in enumerate(image_files):
    raster_file = open(image)
    for band_idx in range(raster_file.count):
        band = raster_file.read(band_idx + 1)
        band_mean = band.mean()
        band_std = band.std()
        file_band_means[idx, band_idx] = band_mean
        file_band_stds[idx, band_idx] = band_std
global_mean = file_band_means.mean(axis=0)
global_std = file_band_stds.mean(axis=0)
res = np.stack((np.array(global_mean), np.array(global_std)), axis=0)
res = res.transpose()
np.savetxt('jp2_image_mean_std.txt', res, delimiter=',')
