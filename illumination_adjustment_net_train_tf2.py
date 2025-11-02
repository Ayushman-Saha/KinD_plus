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
from model_tf2 import DecomNet, Illumination_adjust_net, lrelu

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
print("[*] Loading training data...")

train_low_data = []
train_high_data = []

train_low_data_names = sorted(glob('./LOLdataset/our485/low/*.png'))
train_high_data_names = sorted(glob('./LOLdataset/our485/high/*.png'))
assert len(train_low_data_names) == len(train_high_data_names)

print(f'[*] Number of training data: {len(train_low_data_names)}')

for idx in range(len(train_low_data_names)):
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)

# ==================== Load DecomNet ====================
print("[*] Loading pretrained DecomNet...")

temp_model = keras.models.load_model('./checkpoint/decom_net_retrain/decom_model_final.keras',
                                     custom_objects={'lrelu': lrelu})
print("Loaded DecomNet")

decomnet_model = temp_model.get_layer('SharedDecomNet')

# ==================== Extract Illumination Maps ====================
print("[*] Extracting illumination maps from images...")

decomposed_low_i_data_480 = []
decomposed_high_i_data_480 = []

for idx in range(len(train_low_data)):
    input_low_eval = np.expand_dims(train_low_data[idx], axis=0)
    print(f"Processing low image {idx}")
    R, I = decomnet_model.predict(input_low_eval, verbose=0)
    I_sq = np.squeeze(I)
    decomposed_low_i_data_480.append(I_sq)

for idx in range(len(train_high_data)):
    input_high_eval = np.expand_dims(train_high_data[idx], axis=0)
    print(f"Processing high image {idx}")
    R, I = decomnet_model.predict(input_high_eval, verbose=0)
    I_sq = np.squeeze(I)
    decomposed_high_i_data_480.append(I_sq)

# Split train/validation
eval_adjust_low_i_data = decomposed_low_i_data_480[451:480]
eval_adjust_high_i_data = decomposed_high_i_data_480[451:480]

train_adjust_low_i_data = decomposed_low_i_data_480[0:450]
train_adjust_high_i_data = decomposed_high_i_data_480[0:450]

print(f'[*] Number of training data: {len(train_adjust_high_i_data)}')

# ==================== Build Illumination Adjustment Network ====================
print("[*] Building Illumination Adjustment Network...")

input_low_i = keras.Input(shape=(None, None, 1), name='input_low_i')
input_low_i_ratio = keras.Input(shape=(None, None, 1), name='input_low_i_ratio')
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

adjust_model = Model(inputs=[input_low_i, input_low_i_ratio], outputs=output_i,
                     name='IlluminationAdjustNet')


# ==================== Loss Function ====================
def adjustment_loss(y_true, y_pred):
    """Gradient-aware MSE loss for illumination adjustment"""
    # MSE loss
    loss_square = tf.reduce_mean(tf.square(y_pred - y_true))

    # Gradient loss
    gx_pred = gradient_no_abs(y_pred, 'x')
    gx_true = gradient_no_abs(y_true, 'x')
    gy_pred = gradient_no_abs(y_pred, 'y')
    gy_true = gradient_no_abs(y_true, 'y')

    x_loss = tf.square(gx_pred - gx_true)
    y_loss = tf.square(gy_pred - gy_true)
    loss_grad = tf.reduce_mean(x_loss + y_loss)

    return loss_square + loss_grad


# ==================== Optimizer ====================
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)


# ==================== Training Step ====================
@tf.function
def train_step(batch_low_i, batch_low_i_ratio, batch_high_i):
    with tf.GradientTape() as tape:
        pred = adjust_model([batch_low_i, batch_low_i_ratio], training=True)
        loss = adjustment_loss(batch_high_i, pred)

    grads = tape.gradient(loss, adjust_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, adjust_model.trainable_variables))
    return loss


# ==================== Directories ====================
sample_dir = './illumination_adjust_net_train/'
checkpoint_dir = './checkpoint/illumination_adjust_net_retrain/'
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# ==================== Checkpoint Setup ====================
checkpoint = tf.train.Checkpoint(
    model=adjust_model,
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
train_phase = 'adjustment'
numBatch = len(train_adjust_low_i_data) // batch_size

print(f"[*] Start training for phase {train_phase}")
print(f"[*] Total epochs: {epochs}, Batches per epoch: {numBatch}")

# ==================== Training Loop ====================
start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epochs):
    epoch_start_time = time.time()
    epoch_losses = []

    # Training batches
    for batch_id in range(numBatch):
        batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_low_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

        input_low_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_high_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_low_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

        for patch_id in range(batch_size):
            i_low_data = train_adjust_low_i_data[image_id]
            i_low_expand = np.expand_dims(i_low_data, axis=2)
            i_high_data = train_adjust_high_i_data[image_id]
            i_high_expand = np.expand_dims(i_high_data, axis=2)

            h, w = train_adjust_low_i_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)

            i_low_data_crop = i_low_expand[x:x + patch_size, y:y + patch_size, :]
            i_high_data_crop = i_high_expand[x:x + patch_size, y:y + patch_size, :]

            rand_mode = np.random.randint(0, 7)
            batch_input_low_i[patch_id, :, :, :] = data_augmentation(i_low_data_crop, rand_mode)
            batch_input_high_i[patch_id, :, :, :] = data_augmentation(i_high_data_crop, rand_mode)

            # Calculate ratio
            ratio = np.mean(i_low_data_crop / (i_high_data_crop + 0.0001))
            i_low_data_ratio = np.ones([patch_size, patch_size]) * (1 / ratio + 0.0001)
            i_low_ratio_expand = np.expand_dims(i_low_data_ratio, axis=2)
            i_high_data_ratio = np.ones([patch_size, patch_size]) * ratio
            i_high_ratio_expand = np.expand_dims(i_high_data_ratio, axis=2)

            batch_input_low_i_ratio[patch_id, :, :, :] = i_low_ratio_expand

            # Random swap (augmentation)
            rand_mode = np.random.randint(0, 2)
            if rand_mode == 1:
                input_low_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                input_high_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]
            else:
                input_low_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                input_high_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                i_high_ratio_expand_inv = np.expand_dims(i_high_data_ratio, axis=2)
                input_low_i_rand_ratio[patch_id, :, :, :] = i_high_ratio_expand

            image_id = (image_id + 1) % len(train_adjust_low_i_data)

        loss = train_step(input_low_i_rand, input_low_i_rand_ratio, input_high_i_rand)
        epoch_losses.append(loss.numpy())

        # Print progress
        if (batch_id + 1) % 10 == 0 or batch_id == 0:
            elapsed = time.time() - start_time
            print(f"{train_phase} Epoch: [{epoch + 1:4d}/{epochs}] "
                  f"[{batch_id + 1:4d}/{numBatch}] "
                  f"time: {elapsed:7.2f}s, loss: {loss:.6f}")

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

        for idx in range(10):
            rand_idx = idx
            input_uu_i = eval_adjust_low_i_data[rand_idx]
            input_low_eval_i = np.expand_dims(input_uu_i, axis=0)
            input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=3)

            h, w = eval_adjust_low_i_data[idx].shape
            rand_ratio = np.random.random(1) * 5
            input_uu_i_ratio = np.ones([h, w]) * rand_ratio
            input_low_eval_i_ratio = np.expand_dims(input_uu_i_ratio, axis=0)
            input_low_eval_ii_ratio = np.expand_dims(input_low_eval_i_ratio, axis=3)

            result = adjust_model([input_low_eval_ii, input_low_eval_ii_ratio], training=False)
            save_images(os.path.join(sample_dir, f'h_eval_{epoch + 1}_{rand_idx + 1}_{rand_ratio[0]:.5f}.png'),
                        input_uu_i, result.numpy())

        print(f"[*] Evaluation complete. Results saved to {sample_dir}")

# ==================== Final Model Export ====================
final_model_path = os.path.join(checkpoint_dir, 'illumination_adjust_model_final.keras')
adjust_model.save(final_model_path)
print(f"\n[*] Training complete! Final model saved to: {final_model_path}")
print(f"[*] Finish training for phase {train_phase}.")
print(f"[*] Total training time: {(time.time() - start_time) / 3600:.2f} hours")