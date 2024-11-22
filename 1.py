import os

def count_images(folder):
    return sum([len(files) for r, d, files in os.walk(folder)])

# Set paths
train_path = "dataset/train"
validation_path = "dataset/validation"
test_path = "dataset/test"

# Count images
train_normal_count = count_images(os.path.join(train_path, 'normal'))
train_pneumonia_count = count_images(os.path.join(train_path, 'pneumonia'))
validation_normal_count = count_images(os.path.join(validation_path, 'normal'))
validation_pneumonia_count = count_images(os.path.join(validation_path, 'pneumonia'))
test_normal_count = count_images(os.path.join(test_path, 'normal'))
test_pneumonia_count = count_images(os.path.join(test_path, 'pneumonia'))

# Print out counts
print(f"Train - Normal: {train_normal_count}, Pneumonia: {train_pneumonia_count}")
print(f"Validation - Normal: {validation_normal_count}, Pneumonia: {validation_pneumonia_count}")
print(f"Test - Normal: {test_normal_count}, Pneumonia: {test_pneumonia_count}")
