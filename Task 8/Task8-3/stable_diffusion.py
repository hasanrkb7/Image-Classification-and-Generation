import tensorflow as tf
import keras_cv
import matplotlib.pyplot as plt
import os

os.makedirs("results_sd", exist_ok=True)

print("Loading Stable Diffusion model...")
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
print("Model loaded.")

# Prompts
prompts = [
    "a futuristic city at night with neon lights, cyberpunk style",
    "a beautiful sunset over mountains, oil painting",
    "a cute robot playing guitar in a garden",
    "an underwater kingdom with colorful coral and fish",
]

print(f"\nModel     : Stable Diffusion")
print(f"Image size: 512x512")
print(f"Prompts   : {len(prompts)}")
print(f"Steps     : 50 (default)\n")

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}: '{prompt}'")
    images = model.text_to_image(prompt, batch_size=1)
    img = images[0]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(prompt, fontsize=9, wrap=True)
    plt.tight_layout()
    fname = f"results_sd/image_{i+1:02d}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Saved to {fname}")

print("\nAll images generated and saved to results_sd/")