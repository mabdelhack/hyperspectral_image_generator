def preprocessing_image_ms(x, mean, std):
    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


def categorical_label_from_full_file_name(files, class_indices):
    from keras.utils import to_categorical
    import os
    # file basename without extension
    base_name = [os.path.splitext(os.path.basename(i))[0] for i in files]
    # class label from filename
    base_name = [i.split("_")[0] for i in base_name]
    # label to indices
    image_class = [class_indices[i] for i in base_name]
    # class indices to one-hot-label
    return to_categorical(image_class, num_classes=len(class_indices))
