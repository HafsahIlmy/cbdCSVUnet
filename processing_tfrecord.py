import os
import random
import time
import numpy as np
import tensorflow as tf
import rasterio
import gc

@tf.function
def random_transform(dataset):
    x = tf.random.uniform(())
    if x < 0.10:
        dataset = tf.image.flip_left_right(dataset)
    elif x < 0.20:
        dataset = tf.image.flip_up_down(dataset)
    elif x < 0.30:
        dataset = tf.image.flip_left_right(tf.image.flip_up_down(dataset))
    elif x < 0.40:
        dataset = tf.image.rot90(dataset, k=1)
    elif x < 0.50:
        dataset = tf.image.rot90(dataset, k=2)
    elif x < 0.60:
        dataset = tf.image.rot90(dataset, k=3)
    elif x < 0.70:
        dataset = tf.image.flip_left_right(tf.image.rot90(dataset, k=2))
    return dataset

@tf.function
def flip_inputs_up_down(inputs):
    return tf.image.flip_up_down(inputs)

@tf.function
def flip_inputs_left_right(inputs):
    return tf.image.flip_left_right(inputs)

@tf.function
def transpose_inputs(inputs):
    flip_up_down = tf.image.flip_up_down(inputs)
    transpose = tf.image.flip_left_right(flip_up_down)
    return transpose

@tf.function
def rotate_inputs_90(inputs):
    return tf.image.rot90(inputs, k=1)

@tf.function
def rotate_inputs_180(inputs):
    return tf.image.rot90(inputs, k=2)

@tf.function
def rotate_inputs_270(inputs):
    return tf.image.rot90(inputs, k=3)

def get_unique_filename(identifier, dirs, name):
    while True:
        randomnumber = str(random.randint(1, 12)).zfill(2)
        file = os.path.join(dirs, f"{name}_{identifier}_{randomnumber}.tif")
        if os.path.exists(file):
            return file

def get_unique_filename_s1(identifier, dirs, name):
    years = ["2022", "2023"]
    while True:
        m1 = str(random.randint(0, 12)).zfill(2)
        month = str(random.randint(0, 12)).zfill(2)
        year = random.choice(years)
        file = os.path.join(dirs, f"{name}_{identifier}_{m1}_{month}_{year}.tif")
        if os.path.exists(file):
            return file

def get_unique_filename_month(identifier, dirs, name):
    while True:
        randomnumber = str(random.randint(1, 12)).zfill(1)
        file = os.path.join(dirs, f"{name}_{identifier}_{randomnumber}.tif")
        if os.path.exists(file):
            return file

def image_pair_generatorv1(path, identifiers, nr, labelnr):
    
    def has_cashew(patch):
        return np.sum(patch == 1) > 256 * 256 * 0.05

    def clip_image(img, clip_pixels):
        return img[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels, :]

    def clip_image_class(img, clip_pixels):
        return img[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels]

    for identifier in identifiers:
        class_file = os.path.join(path, "label", f"class_{identifier}.tif")
        rgbn_file = os.path.join(path, "s2", f"rgbn_{identifier}.tif")
        planet_file = get_unique_filename_month(identifier, os.path.join(path, "planet"), "planet")
        other_file = get_unique_filename(identifier, os.path.join(path, "other"), "other")
        landsat_file = get_unique_filename(identifier, os.path.join(path, "l8"), "l8")
        s1_file = os.path.join(path, "s1", f"s1_{identifier}.tif")

        try:
            with rasterio.open(class_file) as src:
                class_image = src.read(1).astype(np.float32)
        except Exception as e:
            print(f"Error opening file {class_file}: {e}")
            continue

        # Membuat mask cashew dan memeriksa nilai unik
        cashew = np.where(class_image == labelnr, 1, 0)
        unique_values = np.unique(cashew)
        print(unique_values)

        # Jika hanya ada [0], lewati identifier ini
        if np.array_equal(unique_values, [0]):
            print(f"Identifier {identifier} hanya memiliki nilai 0 di cassava. Dilewati.")
            continue

        try:
            with rasterio.open(rgbn_file) as src:
                sat_image = src.read().astype(np.float32)
        except Exception as e:
            print(f"Error opening file {rgbn_file}: {e}")
            continue

        try:
            with rasterio.open(planet_file) as src:
                planet_image = src.read().astype(np.float32)
        except Exception as e:
            print(f"Error opening file {planet_file}: {e}")
            continue

        try:
            with rasterio.open(other_file) as src:
                other_image = src.read().astype(np.float32)
        except Exception as e:
            print(f"Error opening file {other_file}: {e}")
            continue

        try:
            with rasterio.open(landsat_file) as src:
                landsat_image = src.read().astype(np.float32)
        except Exception as e:
            print(f"Error opening file {landsat_file}: {e}")
            continue

        try:
            with rasterio.open(s1_file) as src:
                s1_image = src.read().astype(np.float32)
        except Exception as e:
            print(f"Error opening file {s1_file}: {e}")
            continue

        sat_image = np.transpose(sat_image, (1, 2, 0))
        planet_image = np.transpose(planet_image, (1, 2, 0))
        other_image = np.transpose(other_image, (1, 2, 0)) / 10000
        s1_image = np.transpose(s1_image, (1, 2, 0)) / 10000
        landsat_image = np.transpose(landsat_image, (1, 2, 0)) / 10000

        sat_image[:, :, :3] /= 1000.0
        sat_image[:, :, 3] /= 10000.0
        planet_image[:, :, :3] /= 1000.0
        planet_image[:, :, 3] /= 10000.0

        sat_image = clip_image(sat_image, 8)
        planet_image = clip_image(planet_image, 16)
        class_image = clip_image_class(class_image, 16)
        s1_image = clip_image(s1_image, 8)
        other_image = clip_image(other_image, 4)
        landsat_image = clip_image(landsat_image, 2)

        for _ in range(nr):
            sat_patch, planet_patch, other_patch, class_patch, s1_patch, landsat_patch = random_patch_pair(
                sat_image, cashew, planet_image, other_image, landsat_image, s1_image, _
            )
            yield sat_patch, planet_patch, other_patch, class_patch, s1_patch, landsat_patch

        del sat_image, planet_image, other_image, s1_patch, landsat_patch
        gc.collect()

def random_patch_pair(sat_image, class_image, planet_image, other_image, landsat_image, s1_image, iteration, patch_size=(256, 256)):
    patch_size_rgbn = (patch_size[0] // 2, patch_size[1] // 2)
    patch_size_other = (patch_size[0] // 4, patch_size[1] // 4)
    patch_size_landsat = (patch_size[0] // 8, patch_size[1] // 8)
    
    np.random.seed(int(time.time()) + iteration)

    start_i = np.random.randint(0, planet_image.shape[0] - patch_size[0] + 1)
    start_j = np.random.randint(0, planet_image.shape[1] - patch_size[1] + 1)

    planet_patch = planet_image[start_i : start_i + patch_size[0], start_j : start_j + patch_size[1], :]
    class_patch = class_image[start_i : start_i + patch_size[0], start_j : start_j + patch_size[1]]

    start_i = np.random.randint(0, sat_image.shape[0] - patch_size_rgbn[0] + 1)
    start_j = np.random.randint(0, sat_image.shape[1] - patch_size_rgbn[1] + 1)
    sat_patch = sat_image[start_i : start_i + patch_size_rgbn[0], start_j : start_j + patch_size_rgbn[1], :]

    start_i = np.random.randint(0, other_image.shape[0] - patch_size_other[0] + 1)
    start_j = np.random.randint(0, other_image.shape[1] - patch_size_other[1] + 1)
    other_patch = other_image[start_i : start_i + patch_size_other[0], start_j : start_j + patch_size_other[1], :]

    start_i = np.random.randint(0, landsat_image.shape[0] - patch_size_landsat[0] + 1)
    start_j = np.random.randint(0, landsat_image.shape[1] - patch_size_landsat[1] + 1)
    landsat_patch = landsat_image[start_i : start_i + patch_size_landsat[0], start_j : start_j + patch_size_landsat[1], :]

    start_i = np.random.randint(0, s1_image.shape[0] - patch_size_rgbn[0] + 1)
    start_j = np.random.randint(0, s1_image.shape[1] - patch_size_rgbn[1] + 1)
    s1_patch = s1_image[start_i : start_i + patch_size_rgbn[0], start_j : start_j + patch_size_rgbn[1], :]

    return sat_patch, planet_patch, other_patch, class_patch, s1_patch, landsat_patch

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_patches_to_tfrecord(patches, filename):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for sat_patch, planet_patch, other_patch, class_patch, s1_patch, landsat_patch in patches:
            planet_patch_bytes = planet_patch.astype(np.float32).tobytes()
            sat_patch_bytes = sat_patch.astype(np.float32).tobytes()
            other_patch_bytes = other_patch.astype(np.float32).tobytes()
            class_patch_bytes = class_patch.astype(np.float32).tobytes()
            s1_patch_bytes = s1_patch.astype(np.float32).tobytes()
            landsat_patch_bytes = landsat_patch.astype(np.float32).tobytes()

            feature = {
                'input_image1': _bytes_feature(planet_patch_bytes),
                'input_image2': _bytes_feature(sat_patch_bytes),
                'input_image3': _bytes_feature(other_patch_bytes),
                'input_image4': _bytes_feature(s1_patch_bytes),
                'input_image5': _bytes_feature(landsat_patch_bytes),
                'class_patch': _bytes_feature(class_patch_bytes),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())