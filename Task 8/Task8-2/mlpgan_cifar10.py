import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5
x_train = x_train.reshape(-1, 32 * 32 * 3)

print(f"Dataset   : CIFAR-10")
print(f"Train size: {x_train.shape[0]} images, flattened shape {x_train.shape[1:]}")

BATCH_SIZE = 128
NOISE_DIM  = 100
EPOCHS     = 50
IMG_DIM    = 32 * 32 * 3

train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(50000).batch(BATCH_SIZE)

def make_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(NOISE_DIM,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(IMG_DIM, activation="tanh"),
        tf.keras.layers.Reshape((32, 32, 3)),
    ], name="MLP_Generator")
    return model

def make_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, input_shape=(IMG_DIM,)),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ], name="MLP_Discriminator")
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out),  real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss

def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)

gen_opt  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator     = make_generator()
discriminator = make_discriminator()

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        fake_flat   = tf.reshape(fake_images, [-1, IMG_DIM])
        real_out    = discriminator(real_images, training=True)
        fake_out    = discriminator(fake_flat,   training=True)
        g_loss      = generator_loss(fake_out)
        d_loss      = discriminator_loss(real_out, fake_out)
    gen_grads  = gen_tape.gradient(g_loss,  generator.trainable_variables)
    disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads,  generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return g_loss, d_loss

save_dir = "results_mlpgan"
os.makedirs(save_dir, exist_ok=True)
fixed_noise = tf.random.normal([16, NOISE_DIM])

def save_images(epoch):
    generated = generator(fixed_noise, training=False).numpy()
    generated = (generated + 1) / 2.0
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(generated[i], 0, 1))
        ax.axis("off")
    plt.suptitle(f"MLP-GAN CIFAR-10 - Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:03d}.png")
    plt.close()

print(f"\nModel     : MLP-GAN (Fully Connected GAN - no convolutions)")
print(f"Noise dim : {NOISE_DIM}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs    : {EPOCHS}")
print(f"Optimizer : Adam (lr=2e-4, beta_1=0.5)")
print(f"Loss      : Binary Crossentropy (from logits)\n")

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    g_losses, d_losses = [], []
    for batch in train_ds:
        g_l, d_l = train_step(batch)
        g_losses.append(g_l.numpy())
        d_losses.append(d_l.numpy())
    elapsed = time.time() - start
    print(f"Epoch {epoch:3d}/{EPOCHS} | G Loss: {np.mean(g_losses):.4f} | D Loss: {np.mean(d_losses):.4f} | Time: {elapsed:.1f}s")
    if epoch % 10 == 0 or epoch == 1:
        save_images(epoch)

print("\nMLP-GAN training complete. Images saved to results_mlpgan/")