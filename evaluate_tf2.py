# coding: utf-8
from __future__ import print_function

import os
import time

from skimage import filters

# CRITICAL: Set thread limits BEFORE importing TensorFlow
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils import *
from model_tf2 import *
from glob import glob
import argparse

# Configure TensorFlow threading - very conservative
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Enable GPU if available, limit memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU available: {len(physical_devices)} device(s)")
    except:
        pass
else:
    print("Running on CPU")

parser = argparse.ArgumentParser(description='KinD++ Testing Script')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results/',
                    help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./test_images/',
                    help='directory for testing inputs')
parser.add_argument('--adjustment', dest='adjustment', default=True, type=bool,
                    help='whether to adjust illumination')
parser.add_argument('--ratio', dest='ratio', default=5.0, type=float,
                    help='ratio for illumination adjustment')

args = parser.parse_args()

print("=" * 70)
print("Loading models...")
print("=" * 70)

# Build decomposition network
print("[1/3] Loading decomposition network...")
temp_model = keras.models.load_model(
    './checkpoint/decom_net_retrain/decom_model_final.keras',
    custom_objects={'lrelu': lrelu},
    compile=False
)

print("Decomposition model loaded")

# Extract single branch
# input_low = temp_model.get_layer('input_low').output
# R_low = temp_model.get_layer('DecomNet_low_sigmoid').output
# I_low = temp_model.get_layer('DecomNet_low_sigmoid1').output
decom_model = temp_model.get_layer("SharedDecomNet")
decom_model.summary()
print(f"Decomposition model ready - Params: {temp_model.count_params():,}")

# Build restoration network
print("[2/3] Loading restoration network...")
restoration_model = keras.models.load_model(
    './checkpoint/new_restoration_retrain/restoration_model_final.keras',
    custom_objects={'lrelu': lrelu, 'restoration_loss': restoration_loss},
    compile=False
)

# restoration_model.load_weights('restoration_model_converted.h5')
# input_r = keras.Input(shape=(None, None, 3), name='input_r')
# input_i = keras.Input(shape=(None, None, 1), name='input_i')
#
# # Build model (your function returns the output tensor, so we wrap it)
# output = Restoration_net(input_r, input_i, training=False)
# restoration_model = keras.Model(inputs=[input_r, input_i], outputs=output, name='RestorationNet')

# Load weights
# restoration_model.load_weights('restoration_model_converted.h5')

# Confirm successful loading
# print("✅ Weights loaded successfully!")
restoration_model.summary()

print(f"Restoration model ready - Params: {restoration_model.count_params():,}")

# Build illumination adjustment network
print("[3/3] Loading illumination adjustment network...")
adjust_model = keras.models.load_model(
    "./checkpoint/illumination_adjust_net_retrain/illumination_adjust_model_final.keras",
    custom_objects={'lrelu': lrelu},
    compile=False
)

# i_ratio = keras.Input(shape=(None, None, 1), name='i_ratio')
# output_adjust = Illumination_adjust_net(input_i, i_ratio)
#
# adjust_model = keras.Model(inputs=[input_i, i_ratio], outputs=[output_adjust], name='AdjustmentNet')
# adjust_model.load_weights('adjust_model_converted.h5')
print(f"Adjustment model ready - Params: {adjust_model.count_params():,}")


print("=" * 70)
print("All models loaded successfully!")
print("=" * 70)

# Warm-up pass to compile graphs (prevents hanging on first inference)
# print("\nWarming up models (compiling computation graphs)...")
# dummy_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
# dummy_r, dummy_i = decom_model(dummy_input, training=False)
# _ = restoration_model([dummy_r, dummy_i], training=False)
# dummy_ratio = np.ones((1, 64, 64, 1), dtype=np.float32)
# _ = adjust_model([dummy_i, dummy_ratio], training=False)
# print("Warm-up complete! Models ready for inference.")
# print("=" * 70)

# Load test data
print("\nLoading test images...")
eval_low_data = []
eval_img_name = []
eval_low_data_name = glob(args.test_dir + '*')
eval_low_data_name.sort()

for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    print(f"  Image {idx + 1}: {name} - {eval_low_im.shape}")
    h, w, c = eval_low_im.shape

    # The size of test image H and W need to be multiple of 4
    # If not multiple of 4, discard some border pixels
    h_tmp = h % 4
    w_tmp = w % 4
    eval_low_im_resize = eval_low_im[0:h - h_tmp, 0:w - w_tmp, :]
    if h_tmp > 0 or w_tmp > 0:
        print(f"    Resized to: {eval_low_im_resize.shape}")
    eval_low_data.append(eval_low_im_resize)

# Create output directory
sample_dir = args.save_dir
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("=" * 70)
print(f"Starting evaluation on {len(eval_low_data)} images...")
print(f"Adjustment: {args.adjustment}, Ratio: {args.ratio}")
print("=" * 70)

start_time = time.time()

for idx in range(len(eval_low_data)):
    img_start = time.time()
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    h, w, _ = input_low.shape

    print(f"\n[{idx + 1}/{len(eval_low_data)}] Processing: {name}")

    # Step 1: Decompose into reflectance and illumination
    print("  - Decomposing...")
    decom_r_low, decom_i_low = decom_model(input_low_eval, training=False)

    # Step 2: Restore reflectance
    print("  - Restoring reflectance... (this may take a moment)")
    restoration_r = restoration_model([decom_r_low, decom_i_low], training=False)
    print("    ✓ Restoration complete")

    # Step 3: Adjust illumination
    if args.adjustment:
        print("  - Adjusting illumination...")
        ratio = float(args.ratio)
        i_low_data_ratio = tf.ones([h, w], dtype=tf.float32) * ratio
        i_low_data_ratio_expand = tf.expand_dims(i_low_data_ratio, axis=2)
        i_low_data_ratio_expand2 = tf.expand_dims(i_low_data_ratio_expand, axis=0)
        adjust_i = adjust_model([decom_i_low, i_low_data_ratio_expand2], training=False)

    # Step 4: Post-processing
    print("  - Post-processing...")
    # The restoration result can find more details from very dark regions
    # However, it will restore very dark regions with gray colors
    # Use the following operator to alleviate this weakness
    decom_r_sq = tf.squeeze(decom_r_low)
    r_gray = tf.image.rgb_to_grayscale(decom_r_sq)
    r_gray_blur = filters.gaussian(r_gray.numpy(), sigma=1)
    # r_gray_gaussian = r_gray + tf.random.normal(tf.shape(r_gray), mean=0.0, stddev=0.03)
    low_i = tf.math.minimum((r_gray_blur * 2) ** 0.5, 1.0)
    low_i_sq = tf.squeeze(low_i)
    low_i_expand_0 = tf.expand_dims(low_i_sq, axis=0)
    low_i_expand_3 = tf.expand_dims(low_i_expand_0, axis=3)
    result_denoise = tf.multiply(restoration_r, low_i_expand_3)

    # Step 5: Final fusion
    if args.adjustment:
        fusion4 = tf.multiply(result_denoise, adjust_i)
        fusion = decom_i_low * input_low_eval + (1 - decom_i_low) * fusion4
    else:
        fusion = decom_i_low * input_low_eval + (1 - decom_i_low) * result_denoise

    # Step 6: Save result
    output_path = os.path.join(sample_dir, f'{name}_KinD_plus.png')
    save_images(output_path, fusion)

    img_time = time.time() - img_start
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Time: {img_time:.2f}s")

elapsed_time = time.time() - start_time
print("\n" + "=" * 70)
print(f"✓ Evaluation completed!")
print(f"Total time: {elapsed_time:.2f} seconds")
print(f"Average time per image: {elapsed_time / len(eval_low_data):.2f} seconds")
print(f"Results saved to: {sample_dir}")
print("=" * 70)