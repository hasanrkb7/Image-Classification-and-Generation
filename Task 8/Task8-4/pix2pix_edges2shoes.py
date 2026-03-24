import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ── Dataset ───────────────────────────────────────────────────────────────────
# edges2shoes: paired images (edge sketch on left, real shoe on right)
# Each image is 256x512 — left half is edge, right half is shoe photo
# Total: 49,825 train pairs, 200 test pairs
# Different from TF tutorial which uses facades dataset

print("Downloading edges2shoes dataset...")
_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"
path = tf.keras.utils.get_file("edges2shoes.tar.gz", origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path), "edges2shoes/")

print(f"Dataset   : edges2shoes (paired edge sketches -> real shoe photos)")
print(f"Image size: 256x256 (split from 256x512 combined image)")
print(f"Train pairs: 49825 | Test pairs: 200")

# ── Image loading & preprocessing ─────────────────────────────────────────────
IMG_WIDTH  = 286
IMG_HEIGHT = 286
TARGET_SIZE = 256

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    w = tf.shape(image)[1] // 2
    # Left half = edge (input), Right half = shoe (target)
    input_image  = tf.cast(image[:, w:, :], tf.float32)   # real shoe
    target_image = tf.cast(image[:, :w, :], tf.float32)   # edge sketch
    return input_image, target_image

def resize(input_image, target_image, height, width):
    input_image  = tf.image.resize(input_image,  [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, target_image

def random_crop(input_image, target_image):
    stacked = tf.stack([input_image, target_image], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, TARGET_SIZE, TARGET_SIZE, 3])
    return cropped[0], cropped[1]

def normalize(input_image, target_image):
    input_image  = (input_image  / 127.5) - 1
    target_image = (target_image / 127.5) - 1
    return input_image, target_image

def random_jitter(input_image, target_image):
    input_image, target_image = resize(input_image, target_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, target_image = random_crop(input_image, target_image)
    if tf.random.uniform(()) > 0.5:
        input_image  = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)
    return input_image, target_image

def load_image_train(image_file):
    input_image, target_image = load_image(image_file)
    input_image, target_image = random_jitter(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)
    return input_image, target_image

def load_image_test(image_file):
    input_image, target_image = load_image(image_file)
    input_image, target_image = resize(input_image, target_image, TARGET_SIZE, TARGET_SIZE)
    input_image, target_image = normalize(input_image, target_image)
    return input_image, target_image

BATCH_SIZE = 1
BUFFER_SIZE = 400

train_ds = tf.data.Dataset.list_files(PATH + "train/*.jpg")\
    .take(1000)\
    .map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)\
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.list_files(PATH + "val/*.jpg")\
    .map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(BATCH_SIZE)

# ── Generator (U-Net) ─────────────────────────────────────────────────────────
# U-Net architecture: encoder-decoder with skip connections
# Skip connections help preserve fine details (important for edges->shoes)
# Architecture from original Pix2Pix paper (Isola et al., 2017)

def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same",
               use_bias=False, kernel_initializer=tf.random_normal_initializer(0., 0.02)))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same",
               use_bias=False, kernel_initializer=tf.random_normal_initializer(0., 0.02)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def make_generator():
    inputs = tf.keras.layers.Input(shape=[TARGET_SIZE, TARGET_SIZE, 3])
    # Encoder (downsampling)
    down_stack = [
        downsample(64,  4, apply_batchnorm=False),  # 128x128
        downsample(128, 4),                          # 64x64
        downsample(256, 4),                          # 32x32
        downsample(512, 4),                          # 16x16
        downsample(512, 4),                          # 8x8
        downsample(512, 4),                          # 4x4
        downsample(512, 4),                          # 2x2
        downsample(512, 4),                          # 1x1
    ]
    # Decoder (upsampling) with skip connections
    up_stack = [
        upsample(512, 4, apply_dropout=True),        # 2x2
        upsample(512, 4, apply_dropout=True),        # 4x4
        upsample(512, 4, apply_dropout=True),        # 8x8
        upsample(512, 4),                            # 16x16
        upsample(256, 4),                            # 32x32
        upsample(128, 4),                            # 64x64
        upsample(64,  4),                            # 128x128
    ]
    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding="same",
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation="tanh")                           # 256x256

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="UNet_Generator")

# ── Discriminator (PatchGAN) ───────────────────────────────────────────────────
# PatchGAN: classifies each NxN patch as real or fake
# More effective than classifying whole image — focuses on local texture
# 70x70 PatchGAN from original Pix2Pix paper

def make_discriminator():
    inp = tf.keras.layers.Input(shape=[TARGET_SIZE, TARGET_SIZE, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[TARGET_SIZE, TARGET_SIZE, 3], name="target_image")
    x = tf.keras.layers.Concatenate()([inp, tar])   # (256, 256, 6)
    x = downsample(64,  4, apply_batchnorm=False)(x) # 128x128
    x = downsample(128, 4)(x)                         # 64x64
    x = downsample(256, 4)(x)                         # 32x32
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, 4, strides=1,
        kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(1, 4, strides=1,
        kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=x, name="PatchGAN_Discriminator")

# ── Loss functions ─────────────────────────────────────────────────────────────
# Pix2Pix uses two losses:
# 1. Adversarial loss (GAN loss) — fool the discriminator
# 2. L1 loss — pixel-wise similarity to ground truth (makes output less blurry)
# Total G loss = GAN loss + LAMBDA * L1 loss

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100  # weight for L1 loss — from original paper

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_fn(tf.ones_like(disc_real_output),  disc_real_output)
    fake_loss = loss_fn(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_fn(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss  = tf.reduce_mean(tf.abs(target - gen_output))
    total_loss = gan_loss + (LAMBDA * l1_loss)
    return total_loss, gan_loss, l1_loss

generator     = make_generator()
discriminator = make_discriminator()

gen_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# ── Training step ─────────────────────────────────────────────────────────────
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real  = discriminator([input_image, target],     training=True)
        disc_fake  = discriminator([input_image, gen_output], training=True)
        g_total, g_gan, g_l1 = generator_loss(disc_fake, gen_output, target)
        d_loss = discriminator_loss(disc_real, disc_fake)

    gen_grads  = gen_tape.gradient(g_total, generator.trainable_variables)
    disc_grads = disc_tape.gradient(d_loss,  discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads,  generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return g_total, g_gan, g_l1, d_loss

# ── Save sample images ────────────────────────────────────────────────────────
save_dir = "results_pix2pix"
os.makedirs(save_dir, exist_ok=True)

def save_images(epoch, test_input, target):
    prediction = generator(test_input, training=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input (Edge)", "Ground Truth (Shoe)", "Generated (Shoe)"]
    imgs   = [test_input[0], target[0], prediction[0]]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img * 0.5 + 0.5)  # denormalize from [-1,1] to [0,1]
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle(f"Pix2Pix edges2shoes — Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:03d}.png")
    plt.close()

# Get a fixed test sample for consistent comparison
for test_input, target in test_ds.take(1):
    fixed_test_input  = test_input
    fixed_test_target = target

# ── Train ─────────────────────────────────────────────────────────────────────
EPOCHS = 15

print(f"\nModel       : Pix2Pix")
print(f"Generator   : U-Net (encoder-decoder with skip connections)")
print(f"Discriminator: PatchGAN (70x70 patch-based)")
print(f"Dataset     : edges2shoes (different from TF tutorial facades)")
print(f"Train pairs : 49825 | Test pairs: 200")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Epochs      : {EPOCHS}")
print(f"Optimizer   : Adam (lr=2e-4, beta_1=0.5)")
print(f"Loss        : GAN loss + L1 loss (LAMBDA={LAMBDA})\n")

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    g_totals, g_gans, g_l1s, d_losses = [], [], [], []

    for input_image, target in train_ds:
        g_t, g_g, g_l, d_l = train_step(input_image, target)
        g_totals.append(g_t.numpy())
        g_gans.append(g_g.numpy())
        g_l1s.append(g_l.numpy())
        d_losses.append(d_l.numpy())

    elapsed = time.time() - start
    print(f"Epoch {epoch:3d}/{EPOCHS} | G_total: {np.mean(g_totals):.4f} | "
          f"G_GAN: {np.mean(g_gans):.4f} | G_L1: {np.mean(g_l1s):.4f} | "
          f"D: {np.mean(d_losses):.4f} | Time: {elapsed:.1f}s")

    if epoch % 5 == 0 or epoch == 1:
        save_images(epoch, fixed_test_input, fixed_test_target)

print("\nPix2Pix training complete. Images saved to results_pix2pix/")