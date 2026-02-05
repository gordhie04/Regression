import os
import json
import matplotlib.pyplot as plt
from PIL import Image

# Create .kaggle directory
!mkdir -p ~/.kaggle

# Write kaggle.json directly
kaggle_credentials = {
    "username": "elizabethneek",
    "key": "bcd0fa1881fa26f964d4d086de8fe315"
}

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    json.dump(kaggle_credentials, f)

# Set correct permissions
!chmod 600 ~/.kaggle/kaggle.json

print("Kaggle credentials configured!")
#%%
!kaggle datasets download -d soumikrakshit/div2k-high-resolution-images
!unzip div2k-high-resolution-images.zip -d ./data
#%%
# Checking if data folder exists
if os.path.exists('./data'):
    print("✓ Data folder exists!")

    # Print out content
    contents = os.listdir('./data')
    print(f"\nContents: {contents}")

else:
    print("✗ Data folder not found")
#%%
# Path to your images
image_dir = './data/DIV2K_valid_HR/DIV2K_valid_HR/'

# Get list of image files
image_files = sorted(os.listdir(image_dir))[:30]  # Get first 30 images

# Display images in a grid
fig, axes = plt.subplots(10, 3, figsize=(20, 40))
axes = axes.ravel()

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(image_dir, img_file)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(img_file)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
