import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ── Dataset ──────────────────────────────────────────────────────────────────
# CIFAR-10: 60,000 color images (32x32 RGB), 10 classes
# Train: 50,000 images | Test: 10,000 images
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

print(f"Dataset   : CIFAR-10")
print(f"Train size: {x_train.shape[0]} images, shape {x_train.shape[1:]}")
print(f"Test size : {x_test.shape[0]} images, shape {x_test.shape[1:]}")

BATCH_SIZE  = 128
LATENT_DIM  = 256
EPOCHS      = 30
IMG_SHAPE   = (32, 32, 3)

train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(BATCH_SIZE)
test_ds  = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)

# ── Encoder ──────────────────────────────────────────────────────────────────
class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32,  3, strides=2, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(64,  3, strides=2, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
        self.flat  = tf.keras.layers.Flatten()
        self.dense_mu     = tf.keras.layers.Dense(latent_dim)
        self.dense_logvar = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        mu     = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar

# ── CVAE ─────────────────────────────────────────────────────────────────────
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ── Decoder ──────────────────────────────────────────────────────────────────
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense  = tf.keras.layers.Dense(4 * 4 * 128, activation="relu")
        self.reshape = tf.keras.layers.Reshape((4, 4, 128))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64,  3, strides=2, padding="same", activation="relu")
        self.deconv3 = tf.keras.layers.Conv2DTranspose(32,  3, strides=2, padding="same", activation="relu")
        self.out     = tf.keras.layers.Conv2DTranspose(3,   3, padding="same", activation="sigmoid")

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.out(x)
    
# ── Loss ─────────────────────────────────────────────────────────────────────
def compute_loss(model, x):
    x_recon, mu, logvar = model(x)
    # Reconstruction loss (pixel-wise)
    recon_loss = tf.reduce_mean(tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(x, x_recon), axis=[1, 2]))
    # KL divergence
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
        1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1))
    return recon_loss + kl_loss

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ── Training ─────────────────────────────────────────────────────────────────
model     = CVAE(LATENT_DIM)
optimizer = tf.keras.optimizers.Adam(1e-3)

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# Fixed noise for consistent image generation across epochs
fixed_z = tf.random.normal(shape=(16, LATENT_DIM))

def save_generated_images(epoch):
    generated = model.decoder(fixed_z)
    generated = generated.numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(generated[i], 0, 1))
        ax.axis("off")
    plt.suptitle(f"CVAE CIFAR-10 — Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:03d}.png")
    plt.close()

print(f"\nModel     : CVAE (Convolutional VAE)")
print(f"Latent dim: {LATENT_DIM}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs    : {EPOCHS}")
print(f"Optimizer : Adam (lr=1e-3)")
print(f"Loss      : Reconstruction (BCE) + KL Divergence\n")

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_loss = []
    for batch in train_ds:
        loss = train_step(model, batch, optimizer)
        train_loss.append(loss.numpy())

    test_loss = []
    for batch in test_ds:
        test_loss.append(compute_loss(model, batch).numpy())

    avg_train = np.mean(train_loss)
    avg_test  = np.mean(test_loss)
    elapsed   = time.time() - start

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train:.2f} | Test Loss: {avg_test:.2f} | Time: {elapsed:.1f}s")

    if epoch % 5 == 0 or epoch == 1:
        save_generated_images(epoch)

print("\nTraining complete. Images saved to results/")