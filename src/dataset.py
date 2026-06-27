import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_info(data_dir):
    """
    Scans the dataset directory and returns a dictionary of information
    useful for EDA (Class distribution, paths, labels).
    """
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    file_paths = []
    labels = []
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        # Search for common image formats
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        cls_files = []
        for p in patterns:
            cls_files.extend(glob.glob(os.path.join(cls_dir, p)))
            
        file_paths.extend(cls_files)
        labels.extend([cls] * len(cls_files))
        
    return {
        "classes": classes,
        "file_paths": file_paths,
        "labels": labels,
        "total_images": len(file_paths)
    }

def process_path(file_path, label, img_size):
    """Loads, decodes, resizes, and normalizes an image."""
    # Read file
    img = tf.io.read_file(file_path)
    # Decode jpeg
    img = tf.image.decode_jpeg(img, channels=3)
    # Convert to float and normalize to [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize
    img = tf.image.resize(img, img_size)
    return img, label

def get_data_augmenter():
    """Returns a sequential layer for data augmentation in TensorFlow."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
    ])

def create_tf_dataset(file_paths, labels, num_classes, img_size=(224, 224), batch_size=32, is_training=False):
    """Creates a high-performance tf.data.Dataset pipeline."""
    # Convert labels to one-hot encoding
    labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_one_hot))
    
    # Shuffle for training
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_paths), reshuffle_each_iteration=True)
        
    # Map preprocessing function (parallelize with AUTOTUNE)
    dataset = dataset.map(
        lambda path, lbl: (process_path(path, lbl, img_size)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply data augmentation if training
    if is_training:
        augmenter = get_data_augmenter()
        dataset = dataset.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def prepare_datasets(data_dir, img_size=(224, 224), val_split=0.15, test_split=0.15, batch_size=32):
    """
    Loads all images, performs stratified train-test-val splits,
    and returns tf.data.Dataset objects for training, validation, and testing.
    Also returns label mapping and testing paths for evaluation.
    """
    info = get_image_info(data_dir)
    
    if info["total_images"] == 0:
        raise ValueError(f"No images found in dataset directory: {data_dir}")
        
    classes = info["classes"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    
    # Map textual labels to numeric indices
    labels_idx = [class_to_idx[lbl] for lbl in info["labels"]]
    
    # Initial split: train + val and test
    train_val_paths, test_paths, train_val_lbls, test_lbls = train_test_split(
        info["file_paths"],
        labels_idx,
        test_size=test_split,
        random_state=42,
        stratify=labels_idx
    )
    
    # Second split: train and val
    val_size_adjusted = val_split / (1.0 - test_split)
    train_paths, val_paths, train_lbls, val_lbls = train_test_split(
        train_val_paths,
        train_val_lbls,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=train_val_lbls
    )
    
    num_classes = len(classes)
    
    # Create tf.data.Dataset objects
    train_ds = create_tf_dataset(train_paths, train_lbls, num_classes, img_size, batch_size, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_lbls, num_classes, img_size, batch_size, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_lbls, num_classes, img_size, batch_size, is_training=False)
    
    print(f"Dataset preparation complete:")
    print(f" - Train set size: {len(train_paths)}")
    print(f" - Validation set size: {len(val_paths)}")
    print(f" - Test set size: {len(test_paths)}")
    
    return train_ds, val_ds, test_ds, {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "classes": classes,
        "test_paths": test_paths,
        "test_labels": test_lbls
    }
