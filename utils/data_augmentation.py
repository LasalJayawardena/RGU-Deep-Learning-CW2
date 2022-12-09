import numpy as np

import tensorflow as tf


def mixup_augmentation(X, y, beta_con_0=0.2, beta_con_1=0.2, batch_size=512, alpha=0.1 ):
    """
    Implementation of MixUp Data Augmentation
    """

    def beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_one = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_two = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_one / (gamma_one + gamma_two)

    def mix_up(dataset_one, dataset_two, alpha=0.2):
        # Unpack the images and labels from the two datasets
        images_one, labels_one = dataset_one
        images_two, labels_two = dataset_two
        batch_size = tf.shape(images_one)[0]

        # Sample lambda from beta distribution and reshape it to do the mixup
        d_lambda = beta_distribution(batch_size, alpha, alpha)
        x_lambda = tf.reshape(d_lambda, (batch_size, 1, 1, 1))
        y_lambda = tf.reshape(d_lambda, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        images = tf.cast(images_one, tf.float32) * tf.cast(x_lambda, tf.float32) + tf.cast(images_two, tf.float32) * (1.0 - tf.cast(x_lambda, tf.float32))
        labels = tf.cast(labels_one, tf.float32) * y_lambda + tf.cast(labels_two, tf.float32) * (1.0 - y_lambda)
        return (images, labels)

    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = batch_size

    train_sub_dataset_one = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )

    train_sub_dataset_two = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )

    train_dataset = tf.data.Dataset.zip((train_sub_dataset_one, train_sub_dataset_two))

    train_dataset_mu = train_dataset.map(
        lambda sub_ds_one, sub_ds_two: mix_up(sub_ds_one, sub_ds_two, alpha=alpha), 
        num_parallel_calls=AUTO
    )

    return train_dataset_mu



def cutmix_augmentation(X,y, beta_con_0=0.2, beta_con_1=0.2, batch_size=512, IMAGE_SIZE=28, alpha=0.11, beta=0.11, image_size=28):
    """
    Implementation of CutMix Data Augmentation
    """
    IMG_SIZE = image_size

    def beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_one = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_two = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_one / (gamma_one + gamma_two)


    @tf.function
    def get_box(lambda_value):
        cut_r = tf.math.sqrt(1.0 - lambda_value)

        cut_width = IMG_SIZE * cut_r  # rw
        cut_width = tf.cast(cut_width, tf.int32)

        cut_height = IMG_SIZE * cut_r  # rh
        cut_height = tf.cast(cut_height, tf.int32)

        cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # rx
        cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # ry

        boundary_x1 = tf.clip_by_value(cut_x[0] - cut_width // 2, 0, IMG_SIZE)
        boundary_y1 = tf.clip_by_value(cut_y[0] - cut_height // 2, 0, IMG_SIZE)
        bb_x2 = tf.clip_by_value(cut_x[0] + cut_width // 2, 0, IMG_SIZE)
        bb_y2 = tf.clip_by_value(cut_y[0] + cut_height // 2, 0, IMG_SIZE)

        target_height = bb_y2 - boundary_y1
        if target_height == 0:
            target_height += 1

        target_width = bb_x2 - boundary_x1
        if target_width == 0:
            target_width += 1

        return boundary_x1, boundary_y1, target_height, target_width


    @tf.function
    def cutmix(train_ds_one, train_ds_two):
        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        alpha = [0.11]
        beta = [0.11]

        # Get a sample from the Beta distribution
        lambda_value = beta_distribution(1, alpha, beta)

        # Define Lambda
        lambda_value = lambda_value[0][0]

        # Get the bounding box offsets, heights and widths
        boundary_x1, boundary_y1, target_height, target_width = get_box(lambda_value)

        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundary_y1, boundary_x1, target_height, target_width
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundary_y1, boundary_x1, IMG_SIZE, IMG_SIZE
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundary_y1, boundary_x1, target_height, target_width
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundary_y1, boundary_x1, IMG_SIZE, IMG_SIZE
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_width * target_height) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # Combine the labels of both images
        label = lambda_value * tf.cast(label1, tf.float32) + (1 - lambda_value) * tf.cast(label2, tf.float32) 
        return image, label

    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = batch_size

    train_sub_dataset_one = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )

    train_sub_dataset_two = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )

    train_dataset = tf.data.Dataset.zip((train_sub_dataset_one, train_sub_dataset_two))

    train_dataset_mu = train_dataset.map(cutmix)
    return train_dataset_mu


def random_eraser_augmentation(X, y, batch_size):
    """
    Implementation of Random Eraser Data Augmentation
    """
    def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
        def eraser(input_image):
            if input_image.ndim == 3:
                img_height, img_width, img_channel = input_image.shape
            elif input_image.ndim == 2:
                img_height, img_width = input_image.shape

            p_1 = np.random.rand()

            if p_1 > p:
                return input_image

            while True:
                s = np.random.uniform(s_l, s_h) * img_height * img_width
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_width)
                top = np.random.randint(0, img_height)

                if left + w <= img_width and top + h <= img_height:
                    break

            if pixel_level:
                if input_image.ndim == 3:
                    c = np.random.uniform(v_l, v_h, (h, w, img_channel))
                if input_image.ndim == 2:
                    c = np.random.uniform(v_l, v_h, (h, w))
            else:
                c = np.random.uniform(v_l, v_h)

            input_image[top:top + h, left:left + w] = c

            return input_image

        return eraser
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=False)
    )

    generator = datagen.flow(
        X, y,
        batch_size=batch_size,
        shuffle=True,
    )

    return generator


def rotation_width_height_augmentation(X, y, batch_size):
    """
    Implementation of Data Augmentation width Random rotation, random width and height shifting
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.01,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.01,  # randomly shift images vertically (fraction of total height)
        rotation_range=50, 
    )

    generator = datagen.flow(
        X, y,
        batch_size=batch_size,
        shuffle=True,
    )

    return generator 
