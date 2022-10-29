from glob import glob
import matplotlib.pyplot as plt
from hyperspectral_image_generator import hyperspectral_image_generator_jp2

class_indices_column = 'CLASS_ID'
shape_file = "./shape_file/park_centroids.shp"
path_to_images = './images_for_notebook'
image_files = glob(path_to_images + "/*.tif")
batch_size = 4
augmentation_parameters = {'flip': True,
                            'zoom': 1.2,
                            'shift': 0.1,
                            'rotation': 10.0,
                            'sheer': 0.01,
                            'noising': None}
print(image_files)

jp2_image_generator = hyperspectral_image_generator_jp2(image_files, shape_file, class_indices_column,
                                                batch_size=batch_size,
                                                image_mean='jp2_image_mean_std.txt',
                                                rotation_range=augmentation_parameters['rotation'],
                                                horizontal_flip=augmentation_parameters['flip'],
                                                vertical_flip=augmentation_parameters['flip'],
                                                speckle_noise=augmentation_parameters['noising'],
                                                shear_range=augmentation_parameters['sheer'],
                                                scale_range=augmentation_parameters['zoom'],
                                                transform_range=augmentation_parameters['shift'],
                                                crop_size=[96,96]
                                                )

# JP2 case
data = next(jp2_image_generator)
images = data[0]
print(images.shape)
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for idx, ax in enumerate(axs):
    ax.imshow(images[idx, :, :, 1], cmap='gray')
    ax.axis('off')
plt.show()