import shutil
import random
import os

# Function to copy images to a target directory
def copy_images(image_paths, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for image_path in image_paths:
        # Check if the file already exists in the target directory to avoid SameFileError
        destination = os.path.join(target_dir, os.path.basename(image_path))
        if not os.path.exists(destination):
            shutil.copy(image_path, target_dir)  # Copy each image to the target directory

# Paths for your dataset
base_path = "dataset"  # Update this to the path of your dataset
train_path = os.path.join(base_path, "train")
validation_path = os.path.join(base_path, "validation")
test_path = os.path.join(base_path, "test")

# List all images in each folder
train_normal = [os.path.join(train_path, "normal", f) for f in os.listdir(os.path.join(train_path, "normal"))]
train_pneumonia = [os.path.join(train_path, "pneumonia", f) for f in os.listdir(os.path.join(train_path, "pneumonia"))]
validation_normal = [os.path.join(validation_path, "normal", f) for f in os.listdir(os.path.join(validation_path, "normal"))]
validation_pneumonia = [os.path.join(validation_path, "pneumonia", f) for f in os.listdir(os.path.join(validation_path, "pneumonia"))]
test_normal = [os.path.join(test_path, "normal", f) for f in os.listdir(os.path.join(test_path, "normal"))]
test_pneumonia = [os.path.join(test_path, "pneumonia", f) for f in os.listdir(os.path.join(test_path, "pneumonia"))]

# Combine all images
all_normal = train_normal + validation_normal + test_normal
all_pneumonia = train_pneumonia + validation_pneumonia + test_pneumonia

# Shuffle the images
random.shuffle(all_normal)
random.shuffle(all_pneumonia)

# Calculate the new split sizes
total_normal = len(all_normal)
total_pneumonia = len(all_pneumonia)

train_size_normal = int(0.7 * total_normal)
validation_size_normal = int(0.19 * total_normal)
test_size_normal = total_normal - train_size_normal - validation_size_normal

train_size_pneumonia = int(0.7 * total_pneumonia)
validation_size_pneumonia = int(0.19 * total_pneumonia)
test_size_pneumonia = total_pneumonia - train_size_pneumonia - validation_size_pneumonia

# Create the new directories if they don't exist
for folder in ["train", "validation", "test"]:
    for subfolder in ["normal", "pneumonia"]:
        os.makedirs(os.path.join(base_path, folder, subfolder), exist_ok=True)

# Split and move images into new folders
copy_images(all_normal[:train_size_normal], os.path.join(base_path, "train", "normal"))
copy_images(all_pneumonia[:train_size_pneumonia], os.path.join(base_path, "train", "pneumonia"))

copy_images(all_normal[train_size_normal:train_size_normal+validation_size_normal], os.path.join(base_path, "validation", "normal"))
copy_images(all_pneumonia[train_size_pneumonia:train_size_pneumonia+validation_size_pneumonia], os.path.join(base_path, "validation", "pneumonia"))

copy_images(all_normal[train_size_normal+validation_size_normal:], os.path.join(base_path, "test", "normal"))
copy_images(all_pneumonia[train_size_pneumonia+validation_size_pneumonia:], os.path.join(base_path, "test", "pneumonia"))

print(f"Data split complete:")
print(f"Train - Normal: {train_size_normal}, Pneumonia: {train_size_pneumonia}")
print(f"Validation - Normal: {validation_size_normal}, Pneumonia: {validation_size_pneumonia}")
print(f"Test - Normal: {test_size_normal}, Pneumonia: {test_size_pneumonia}")
