import os, time, random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
from PIL import Image
from utils import load_images, data_augmentation, gradient, save_images
from model_tf2 import DecomNet

# --- Hyperparameters ---
batch_size = 10
patch_size = 48
epochs = 2500
learning_rate = 1e-4
train_data_dir = './LOLdataset/our485'
train_result_dir = './decom_net_train_result'
checkpoint_dir = './checkpoint/decom_net_retrain/'
eval_every_epoch = 500

os.makedirs(train_result_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Data loading ---
# Training data
train_low_names = sorted(glob(f'{train_data_dir}/low/*.png'))
train_high_names = sorted(glob(f'{train_data_dir}/high/*.png'))
assert len(train_low_names) == len(train_high_names)
print(f'[*] Number of training data: {len(train_low_names)}')

train_low_data = [load_images(p) for p in train_low_names]
train_high_data = [load_images(p) for p in train_high_names]

# Evaluation data
eval_low_names = sorted(glob('./LOLdataset/eval15/low/*.png'))
eval_high_names = sorted(glob('./LOLdataset/eval15/high/*.png'))

eval_low_data = [load_images(p) for p in eval_low_names]
eval_high_data = [load_images(p) for p in eval_high_names]

# --- Define model with SHARED DecomNet ---
# Define inputs
input_low = keras.Input(shape=(None, None, 3), name='input_low')
input_high = keras.Input(shape=(None, None, 3), name='input_high')

# Create a shared DecomNet by wrapping it in a functional model
# This ensures weight sharing when applied to different inputs
decomnet_input = keras.Input(shape=(None, None, 3))
R_out, I_out = DecomNet(decomnet_input, name='DecomNet')
decomnet_model = keras.Model(inputs=decomnet_input, outputs=[R_out, I_out], name='SharedDecomNet')

# Apply the SAME model to both inputs (weight sharing!)
R_low, I_low = decomnet_model(input_low)
R_high, I_high = decomnet_model(input_high)

I_low_3 = keras.layers.Concatenate(axis=-1, name='I_low_3')([I_low, I_low, I_low])
I_high_3 = keras.layers.Concatenate(axis=-1, name='I_high_3')([I_high, I_high, I_high])

# Main training model
decom_model = keras.Model(
    inputs=[input_low, input_high],
    outputs=[R_low, I_low, R_high, I_high],
    name='DecomNet'
)


# --- Loss functions ---
def mutual_i_loss(I_low, I_high):
    """Mutual consistency loss for illumination maps"""
    gx_l, gx_h = gradient(I_low, "x"), gradient(I_high, "x")
    gy_l, gy_h = gradient(I_low, "y"), gradient(I_high, "y")
    x_loss = (gx_l + gx_h) * tf.exp(-10 * (gx_l + gx_h))
    y_loss = (gy_l + gy_h) * tf.exp(-10 * (gy_l + gy_h))
    return tf.reduce_mean(x_loss + y_loss)


def mutual_i_input_loss(I_low, input_im):
    """Illumination smoothness loss constrained by input structure"""
    gray = tf.image.rgb_to_grayscale(input_im)
    gx_l, gy_l = gradient(I_low, "x"), gradient(I_low, "y")
    gx_g, gy_g = gradient(gray, "x"), gradient(gray, "y")
    x_loss = tf.abs(gx_l / tf.maximum(gx_g, 0.01))
    y_loss = tf.abs(gy_l / tf.maximum(gy_g, 0.01))
    return tf.reduce_mean(x_loss + y_loss)


# --- Optimizer ---
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)


# --- Training step ---
@tf.function
def train_step(low_patch, high_patch):
    """Single training step with automatic differentiation"""
    with tf.GradientTape() as tape:
        R_low, I_low, R_high, I_high = decom_model([low_patch, high_patch], training=True)
        I_low_3 = tf.concat([I_low, I_low, I_low], axis=-1)
        I_high_3 = tf.concat([I_high, I_high, I_high], axis=-1)

        # Reconstruction losses
        recon_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - low_patch))
        recon_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - high_patch))

        # Reflectance consistency loss
        equal_R = tf.reduce_mean(tf.abs(R_low - R_high))

        # Illumination losses
        i_mutual = mutual_i_loss(I_low, I_high)
        i_input_h = mutual_i_input_loss(I_high, high_patch)
        i_input_l = mutual_i_input_loss(I_low, low_patch)

        # Total loss
        loss = (recon_low + recon_high
                + 0.009 * equal_R
                + 0.2 * i_mutual
                + 0.15 * (i_input_h + i_input_l))

    # Compute and apply gradients
    grads = tape.gradient(loss, decom_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, decom_model.trainable_variables))
    return loss


# --- Checkpoint setup ---
checkpoint = tf.train.Checkpoint(
    model=decom_model,
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

# --- Training setup ---
train_phase = 'decomposition'
num_batches = len(train_low_data) // batch_size

print(f"[*] Start training for phase {train_phase}")
print(f"[*] Total epochs: {epochs}, Batches per epoch: {num_batches}")

# --- Training loop ---
start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epochs):
    epoch_start_time = time.time()
    epoch_losses = []

    for batch_id in range(num_batches):
        # Prepare batch
        batch_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32)
        batch_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32)

        for i in range(batch_size):
            h, w, _ = train_low_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            mode = random.randint(0, 7)

            batch_low[i] = data_augmentation(
                train_low_data[image_id][x:x + patch_size, y:y + patch_size, :], mode
            )
            batch_high[i] = data_augmentation(
                train_high_data[image_id][x:x + patch_size, y:y + patch_size, :], mode
            )

            image_id = (image_id + 1) % len(train_low_data)

            # Shuffle data when we've gone through all images
            if image_id == 0:
                tmp = list(zip(train_low_data, train_high_data))
                random.shuffle(tmp)
                train_low_data, train_high_data = zip(*tmp)

        # Training step
        loss = train_step(batch_low, batch_high)
        epoch_losses.append(float(loss))

        # Print progress
        if (batch_id + 1) % 10 == 0 or batch_id == 0:
            elapsed = time.time() - start_time
            print(f"{train_phase} Epoch: [{epoch + 1:4d}/{epochs}] "
                  f"[{batch_id + 1:4d}/{num_batches}] "
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

    # Evaluation - Use the trained model directly!
    if (epoch + 1) % eval_every_epoch == 0:
        print(f"[*] Evaluating for phase {train_phase} / epoch {epoch + 1}...")

        # Evaluate on low-light images
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            # Create dummy high input (not used in computation for low output)
            input_high_dummy = np.zeros_like(input_low_eval)

            # Get outputs from trained model
            R_low, I_low, _, _ = decom_model([input_low_eval, input_high_dummy], training=False)

            # Convert single channel illumination to 3-channel for saving
            I_low_3 = tf.concat([I_low, I_low, I_low], axis=-1)

            save_path = os.path.join(train_result_dir, f'low_{idx + 1}_{epoch + 1}.png')
            save_images(save_path, R_low.numpy(), I_low_3.numpy())

        # Evaluate on high-light images
        for idx in range(len(eval_high_data)):
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            # Create dummy low input (not used in computation for high output)
            input_low_dummy = np.zeros_like(input_high_eval)

            # Get outputs from trained model
            _, _, R_high, I_high = decom_model([input_low_dummy, input_high_eval], training=False)

            # Convert single channel illumination to 3-channel for saving
            I_high_3 = tf.concat([I_high, I_high, I_high], axis=-1)

            save_path = os.path.join(train_result_dir, f'high_{idx + 1}_{epoch + 1}.png')
            save_images(save_path, R_high.numpy(), I_high_3.numpy())

        print(f"[*] Evaluation complete. Results saved to {train_result_dir}")

# --- Final model export ---
final_model_path = os.path.join(checkpoint_dir, 'decom_model_final.keras')
decom_model.save(final_model_path)
print(f"\n[*] Training complete! Final model saved to: {final_model_path}")

print(f"[*] Finish training for phase {train_phase}.")
print(f"[*] Total training time: {(time.time() - start_time) / 3600:.2f} hours")