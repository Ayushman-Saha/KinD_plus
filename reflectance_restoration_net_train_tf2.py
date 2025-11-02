# coding: utf-8
from __future__ import print_function

import os
import time
import random
import numpy as np
from PIL import Image
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

from utils import *
from model_tf2 import DecomNet, Restoration_net, lrelu, restoration_loss

# Hyperparameters
batch_size = 10
patch_size = 48
initial_learning_rate = 0.0001
epochs = 2500
eval_every_epoch = 500

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

print("[*] Initialize model successfully...")

# ==================== Load Data ====================
print("[*] Loading evaluation data...")

eval_low_data = []
eval_high_data = []
eval_low_data_bmp = []

eval_low_data_name = (glob('./LOLdataset/our485/low/*.png') +
                      glob('./LOLdataset/add_sys/sys_low/*.png') +
                      glob('./LOLdataset/dark/low/*.png'))
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)

eval_low_data_name_bmp = glob('./LOLdataset/eval15/low/*.png')
eval_low_data_name_bmp.sort()
for idx in range(len(eval_low_data_name_bmp)):
    eval_low_im = load_images(eval_low_data_name_bmp[idx])
    eval_low_data_bmp.append(eval_low_im)
    print(eval_low_im.shape)

eval_high_data_name = (glob('./LOLdataset/our485/high/*.png') +
                       glob('./LOLdataset/add_sys/sys_high/*.png') +
                       glob('./LOLdataset/dark/high/*.png'))
eval_high_data_name.sort()
for idx in range(len(eval_high_data_name)):
    eval_high_im = load_images(eval_high_data_name[idx])
    eval_high_data.append(eval_high_im)
    print(eval_high_im.shape)

# ==================== Load DecomNet ====================
print("[*] Loading pretrained DecomNet...")

temp_model = keras.models.load_model('./checkpoint/decom_net_retrain/decom_model_final.keras',
                                     custom_objects={'lrelu': lrelu})
print("Loaded DecomNet")

# input_low = temp_model.get_layer('input_low').output
# R_low = temp_model.get_layer('DecomNet_low_sigmoid').output  # Reflectance output
# I_low = temp_model.get_layer('DecomNet_low_sigmoid1').output  # Illumination output
decomnet_model = temp_model.get_layer("SharedDecomNet")

# ==================== Extract R and I ====================
print("[*] Extracting R and I from images...")

train_restoration_low_r_data_480 = []
train_restoration_low_i_data_480 = []
train_restoration_high_r_data_480 = []

for idx in range(len(eval_high_data)):
    input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
    print(f"Processing high image {idx}")
    result_R, result_I = decomnet_model.predict(input_high_eval, verbose=0)
    result_R = (result_R * 0.99) ** 1.2
    result_R_sq = np.squeeze(result_R)
    train_restoration_high_r_data_480.append(result_R_sq)

for idx in range(len(eval_low_data)):
    input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
    print(f"Processing low image {idx}")
    result_R, result_I = decomnet_model.predict(input_low_eval, verbose=0)
    result_R_sq = np.squeeze(result_R)
    result_I_sq = np.squeeze(result_I)
    train_restoration_low_r_data_480.append(result_R_sq)
    train_restoration_low_i_data_480.append(result_I_sq)

eval_restoration_low_r_data_bmp = []
eval_restoration_low_i_data_bmp = []
for idx in range(len(eval_low_data_bmp)):
    input_low_eval = np.expand_dims(eval_low_data_bmp[idx], axis=0)
    print(f"Processing bmp image {idx}")
    result_R, result_I = decomnet_model.predict(input_low_eval, verbose=0)
    result_R_sq = np.squeeze(result_R)
    result_I_sq = np.squeeze(result_I)
    eval_restoration_low_r_data_bmp.append(result_R_sq)
    eval_restoration_low_i_data_bmp.append(result_I_sq)

# Split train/validation
eval_restoration_low_r_data = train_restoration_low_r_data_480[235:240]
eval_restoration_low_i_data = train_restoration_low_i_data_480[235:240]

train_restoration_low_r_data = (train_restoration_low_r_data_480[0:234] +
                                train_restoration_low_r_data_480[241:-1])
train_restoration_low_i_data = (train_restoration_low_i_data_480[0:234] +
                                train_restoration_low_i_data_480[241:-1])
train_restoration_high_r_data = (train_restoration_high_r_data_480[0:234] +
                                 train_restoration_high_r_data_480[241:-1])

print(f"Train: {len(train_restoration_high_r_data)}, Eval: {len(eval_restoration_low_r_data)}")

# ==================== Build Restoration Network ====================
print("[*] Building Restoration Network...")

input_low_r = keras.Input(shape=(None, None, 3), name='input_low_r')
input_low_i = keras.Input(shape=(None, None, 1), name='input_low_i')
output_r = Restoration_net(input_low_r, input_low_i)

restoration_model = Model(inputs=[input_low_r, input_low_i], outputs=output_r, name='RestorationNet')

# ==================== Optimizer ====================
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)


# ==================== Training Step ====================
@tf.function
def train_step(batch_low_r, batch_low_i, batch_high):
    with tf.GradientTape() as tape:
        pred = restoration_model([batch_low_r, batch_low_i], training=True)
        loss = restoration_loss(batch_high, pred)

    grads = tape.gradient(loss, restoration_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, restoration_model.trainable_variables))
    return loss


# ==================== Directories ====================
sample_dir = './new_restoration_train_results/'
checkpoint_dir = './checkpoint/new_restoration_retrain/'
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# ==================== Checkpoint Setup ====================
checkpoint = tf.train.Checkpoint(
    model=restoration_model,
    optimizer=optimizer,
    epoch=tf.Variable(0, dtype=tf.int64)
)

checkpoint_manager = tf.train.CheckpointManager(
    checkpoint,
    directory=checkpoint_dir,
    max_to_keep=3,
    checkpoint_name='model'
)

# Restore from latest checkpoint if available
start_epoch = 0
if checkpoint_manager.latest_checkpoint:
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
    start_epoch = int(checkpoint.epoch.numpy())
    print(f'[*] Restored from {checkpoint_manager.latest_checkpoint}')
    print(f'[*] Resuming from epoch {start_epoch}')
else:
    print('[*] No checkpoint found. Training from scratch.')

# ==================== Training Setup ====================
train_phase = 'restoration'
numBatch = len(train_restoration_low_r_data) // batch_size

print(f"[*] Start training for phase {train_phase}")
print(f"[*] Total epochs: {epochs}, Batches per epoch: {numBatch}")

# ==================== Training Loop ====================
start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epochs):
    epoch_start_time = time.time()
    epoch_losses = []

    # Learning rate schedule
    if epoch <= 300:
        lr = initial_learning_rate
    elif epoch <= 500:
        lr = initial_learning_rate / 2
    elif epoch <= 1500:
        lr = initial_learning_rate / 4
    else:
        lr = initial_learning_rate / 8

    optimizer.learning_rate.assign(lr)

    # Training batches
    for batch_id in range(numBatch):
        batch_input_low_r = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

        for patch_id in range(batch_size):
            h, w, _ = train_restoration_low_r_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            i_low_expand = np.expand_dims(train_restoration_low_i_data[image_id], axis=2)
            rand_mode = random.randint(0, 7)

            batch_input_low_r[patch_id, :, :, :] = data_augmentation(
                train_restoration_low_r_data[image_id][x:x + patch_size, y:y + patch_size, :], rand_mode)
            batch_input_low_i[patch_id, :, :, :] = data_augmentation(
                i_low_expand[x:x + patch_size, y:y + patch_size, :], rand_mode)
            batch_input_high[patch_id, :, :, :] = data_augmentation(
                train_restoration_high_r_data[image_id][x:x + patch_size, y:y + patch_size, :], rand_mode)

            image_id = (image_id + 1) % len(train_restoration_low_r_data)
            if image_id == 0:
                tmp = list(zip(train_restoration_low_r_data, train_restoration_low_i_data,
                               train_restoration_high_r_data))
                random.shuffle(tmp)
                train_restoration_low_r_data, train_restoration_low_i_data, train_restoration_high_r_data = zip(*tmp)

        loss = train_step(batch_input_low_r, batch_input_low_i, batch_input_high)
        epoch_losses.append(loss.numpy())

        # Print progress
        if (batch_id + 1) % 10 == 0 or batch_id == 0:
            elapsed = time.time() - start_time
            print(f"{train_phase} Epoch: [{epoch + 1:4d}/{epochs}] "
                  f"[{batch_id + 1:4d}/{numBatch}] "
                  f"time: {elapsed:7.2f}s, loss: {float(loss):.6f}")

    # Update epoch counter
    checkpoint.epoch.assign_add(1)

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    avg_loss = np.mean(epoch_losses)
    print(f"{'=' * 70}")
    print(f"Epoch {epoch + 1} complete - Time: {epoch_time:.2f}s, Avg Loss: {avg_loss:.6f}")
    print(f"{'=' * 70}")

    # Save checkpoint every epoch
    save_path = checkpoint_manager.save()
    print(f"[*] Saved checkpoint: {save_path}")

    # Evaluation
    if (epoch + 1) % eval_every_epoch == 0:
        print(f"[*] Evaluating for phase {train_phase} / epoch {epoch + 1}...")

        for idx in range(len(eval_restoration_low_r_data)):
            input_r = np.expand_dims(eval_restoration_low_r_data[idx], axis=0)
            input_i = np.expand_dims(np.expand_dims(eval_restoration_low_i_data[idx], axis=0), axis=3)
            result = restoration_model([input_r, input_i], training=False)
            save_images(os.path.join(sample_dir, f'eval_{idx + 101}_{epoch + 1}.png'), result.numpy())

        for idx in range(len(eval_restoration_low_r_data_bmp)):
            input_r = np.expand_dims(eval_restoration_low_r_data_bmp[idx], axis=0)
            input_i = np.expand_dims(np.expand_dims(eval_restoration_low_i_data_bmp[idx], axis=0), axis=3)
            result = restoration_model([input_r, input_i], training=False)
            save_images(os.path.join(sample_dir, f'eval_bmp_{idx + 101}_{epoch + 1}.png'), result.numpy())

        print(f"[*] Evaluation complete. Results saved to {sample_dir}")

# ==================== Final Model Export ====================
final_model_path = os.path.join(checkpoint_dir, 'restoration_model_final.keras')
restoration_model.save(final_model_path)
print(f"\n[*] Training complete! Final model saved to: {final_model_path}")
print(f"[*] Finish training for phase {train_phase}.")
print(f"[*] Total training time: {(time.time() - start_time) / 3600:.2f} hours")