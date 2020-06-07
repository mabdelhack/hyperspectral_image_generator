def hyperspectral_image_generator(files, class_indices, batch_size=32, image_mean=None,
                           rotation_range=0, shear_range=0, scale_range=1,
                           transform_range=0, horizontal_flip=False,
                           vertical_flip=False, crop=False, crop_size=None, filling_mode='edge',
                           speckle_noise=None):
    from skimage.io import imread
    import numpy as np
    from random import sample
    from image_functions import categorical_label_from_full_file_name, preprocessing_image_ms

    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files, batch_size)
        # get one_hot_label
        batch_Y = categorical_label_from_full_file_name(batch_files,
                                                        class_indices)
        # array for images
        batch_X = []
        # loop over images of the current batch
        for idx, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)
            if image_mean is not None:
                mean_std_data = np.loadtxt(image_mean, delimiter=',')
                image = preprocessing_image_ms(image, mean_std_data[0], mean_std_data[1])
            # process image
            image = augmentation_image_ms(image, rotation_range=rotation_range, shear_range=shear_range,
                                          scale_range=scale_range,
                                          transform_range=transform_range, horizontal_flip=horizontal_flip,
                                          vertical_flip=vertical_flip, warp_mode=filling_mode)
            if speckle_noise is not None:
                from skimage.util import random_noise
                image_max = np.max(np.abs(image), axis=(0, 1))
                image /= image_max

                image = random_noise(image, mode='speckle', var=speckle_noise)
                image *= image_max

            if crop:
                if crop_size is None:
                    crop_size = image.shape[0:2]
                image = crop_image(image, crop_size)
            # put all together
            batch_X += [image]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)


def hyperspectral_image_generator_jp2(files, shape_file, class_indices_column, batch_size=32, image_mean=None,
                                      rotation_range=0, shear_range=0, scale_range=1,
                                      transform_range=0, horizontal_flip=False,
                                      vertical_flip=False, crop_size=None, filling_mode='edge',
                                      speckle_noise=None):
    from rasterio.mask import mask
    from rasterio import open
    from shapely.geometry import box
    import geopandas as gpd
    import numpy as np
    from random import sample
    from image_functions import categorical_label_from_full_file_name, preprocessing_image_ms
    from keras.utils import to_categorical

    geometry_df = gpd.read_file(shape_file)
    centroids = geometry_df['geometry'].values
    class_indices = geometry_df[class_indices_column].values.astype(int)
    number_of_classes = class_indices.max()
    files_centroids = list(zip(files*len(centroids), list(centroids)*len(files), list(class_indices)*len(files)))
    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files_centroids, batch_size)
        # get one_hot_label

        batch_Y = []
        # array for images
        batch_X = []
        # loop over images of the current batch
        for idx, (rf, polycenter, label) in enumerate(batch_files):
            raster_file = open(rf)
            mask_polygon = box(max(polycenter.coords.xy[0][0] - raster_file.transform[0] * crop_size[0] * 2,
                                   raster_file.bounds.left),
                               max(polycenter.coords.xy[1][0] - raster_file.transform[4] * crop_size[1] * 2,
                                   raster_file.bounds.bottom),
                               min(polycenter.coords.xy[0][0] + raster_file.transform[0] * crop_size[0] * 2,
                                   raster_file.bounds.right),
                               min(polycenter.coords.xy[1][0] + raster_file.transform[4] * crop_size[1] * 2,
                                   raster_file.bounds.top))
            image, out_transform = mask(raster_file, shapes=[mask_polygon], crop=True, all_touched=True)
            image = np.transpose(image, (1, 2, 0))
            if image_mean is not None:
                mean_std_data = np.loadtxt(image_mean, delimiter=',')
                image = preprocessing_image_ms(image.astype(np.float64), mean_std_data[0], mean_std_data[1])
            # process image
            image = augmentation_image_ms(image, rotation_range=rotation_range, shear_range=shear_range,
                                          scale_range=scale_range,
                                          transform_range=transform_range, horizontal_flip=horizontal_flip,
                                          vertical_flip=vertical_flip, warp_mode=filling_mode)
            if speckle_noise is not None:
                from skimage.util import random_noise
                image_max = np.max(np.abs(image), axis=(0, 1))
                image /= image_max

                image = random_noise(image, mode='speckle', var=speckle_noise)
                image *= image_max

            image = crop_image(image, crop_size)

            # put all together
            batch_X += [image]
            batch_Y += [to_categorical(label, num_classes=number_of_classes)]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)


def augmentation_image_ms(image, rotation_range=0, shear_range=0, scale_range=1, transform_range=0,
                          horizontal_flip=False, vertical_flip=False, warp_mode='edge'):
    from skimage.transform import AffineTransform, SimilarityTransform, warp
    from numpy import deg2rad, flipud, fliplr
    from numpy.random import uniform, random_integers
    from random import choice

    image_shape = image.shape
    # Generate image transformation parameters
    rotation_angle = uniform(low=-abs(rotation_range), high=abs(rotation_range))
    shear_angle = uniform(low=-abs(shear_range), high=abs(shear_range))
    scale_value = uniform(low=abs(1 / scale_range), high=abs(scale_range))
    translation_values = (random_integers(-abs(transform_range), abs(transform_range)),
                          random_integers(-abs(transform_range), abs(transform_range)))

    # Horizontal and vertical flips
    if horizontal_flip:
        # randomly flip image up/down
        if choice([True, False]):
            image = flipud(image)
    if vertical_flip:
        # randomly flip image left/right
        if choice([True, False]):
            image = fliplr(image)

    # Generate transformation object
    transform_toorigin = SimilarityTransform(scale=(1, 1), rotation=0, translation=(-image_shape[0], -image_shape[1]))
    transform_revert = SimilarityTransform(scale=(1, 1), rotation=0, translation=(image_shape[0], image_shape[1]))
    transform = AffineTransform(scale=(scale_value, scale_value), rotation=deg2rad(rotation_angle),
                                shear=deg2rad(shear_angle), translation=translation_values)
    # Apply transform
    image = warp(image, ((transform_toorigin) + transform) + transform_revert, mode=warp_mode, preserve_range=True)
    return image


def crop_image(image, target_size):
    from numpy import ceil, floor
    x_crop = min(image.shape[0], target_size[0])
    y_crop = min(image.shape[1], target_size[1])
    midpoint = [ceil(image.shape[0] / 2), ceil(image.shape[1] / 2)]

    out_img = image[int(midpoint[0] - ceil(x_crop / 2)):int(midpoint[0] + floor(x_crop / 2)),
              int(midpoint[1] - ceil(y_crop / 2)):int(midpoint[1] + floor(y_crop / 2)),
              :]
    assert list(out_img.shape[0:2]) == target_size
    return out_img