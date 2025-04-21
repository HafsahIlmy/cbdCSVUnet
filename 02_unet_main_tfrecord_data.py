import os
import gc
import random
import time
import numpy as np
import tensorflow as tf
import rasterio
from initializer_data_2 import initial_dict, initialize_areas
from processing_tfrecord import (
    random_transform, flip_inputs_up_down, flip_inputs_left_right, transpose_inputs, rotate_inputs_90, rotate_inputs_180,
    rotate_inputs_270, get_unique_filename, get_unique_filename_month, image_pair_generatorv1, write_patches_to_tfrecord
)

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

patchsize = 256
year = 2023
month = 3
path = fr"C:\Users\Nyein\hafsah_playground\unet\data\train_{str(year)}_m{str(month)}"
outputpath = fr"C:\Users\Nyein\hafsah_playground\unet\data\train_{str(year)}_m{str(month)}\tfrecord_cassava_{str(year)}_m{str(month)}"
labelnr = 4

# Pastikan folder outputpath ada
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
    print(f"Created directory: {outputpath}")

filenames = initialize_areas()

for i in range(105, 201, 1):
    print(f'starting {i} from 200')
    for identifier in filenames:
        print(identifier)
        nr = str(i).zfill(3)
        filename = f'{outputpath}/patch_{identifier}{nr}.tfrecord.gz'
        if not os.path.exists(filename):
            patches = list(image_pair_generatorv1(path, [identifier], 9, labelnr))
            if not patches:  # Jika tidak ada patch yang dihasilkan
                print(f"No patches generated for identifier {identifier}. Skipping to next identifier.")
                continue  # Lewati ke identifier berikutnya
            print(f"Generated {len(patches)} patches for {identifier}. Saving to {filename}...")
            write_patches_to_tfrecord(patches, filename)
        else:
            print(f"File {filename} already exists. Skipping...")