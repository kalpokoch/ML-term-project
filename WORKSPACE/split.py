import os
import shutil
import random

# Define the paths to your dataset folders
dataset_path = r"D:\ML term project\Dataset\sign"
train_path = r"D:\ML term project\Dataset\train"
test_path = r"D:\ML term project\Dataset\test"

# Create directories for training and test sets
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Define the ratio for splitting the dataset (e.g., 70% training, 30% test)
train_ratio = 0.8

# Iterate over each class folder
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        # Create subdirectories for the current class in training and test sets
        train_class_path = os.path.join(train_path, class_folder)
        test_class_path = os.path.join(test_path, class_folder)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        # Gather the filenames of the data samples in the current class folder
        data_files = os.listdir(class_path)
        
        # Shuffle the filenames to randomize the data order
        random.shuffle(data_files)
        
        # Calculate the number of samples for each subset
        num_samples = len(data_files)
        num_train_samples = int(train_ratio * num_samples)
        
        # Split the data into training and test sets
        train_files = data_files[:num_train_samples]
        test_files = data_files[num_train_samples:]
        
        # Copy the files to the corresponding directories
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(train_class_path, file))
        for file in test_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(test_class_path, file))

print("Dataset split into training and test sets and saved successfully.")
