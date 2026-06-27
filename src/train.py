import os
import argparse
import time
import numpy as np
import tensorflow as tf
from dataset import prepare_datasets
from models import build_custom_cnn, build_mobilenetv2_transfer, build_resnet50_transfer, compile_model
from utils import plot_history, plot_confusion_matrix, evaluate_predictions, save_model_comparison

# Ensure TensorFlow doesn't hog GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease classification models.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--quick_run", action="store_true", help="Run 1 epoch with minimal steps for rapid testing")
    return parser.parse_args()

def train_and_evaluate_model(model_name, model_fn, train_ds, val_ds, test_ds, classes, test_labels, args, models_dir, outputs_dir):
    print(f"\n" + "#"*60)
    print(f" TRAINING MODEL: {model_name} ")
    print("#"*60 + "\n")
    
    num_classes = len(classes)
    
    # Instantiate the model architecture
    try:
        model = model_fn(num_classes=num_classes)
    except Exception as e:
        print(f"Warning: Failed to load pre-trained weights for {model_name} (likely offline). Initializing with random weights. Error: {e}")
        # Workaround for offline runs: load models with random weights (weights=None)
        if "MobileNet" in model_name:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models
            base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(128, activation='relu'), layers.Dropout(0.3), layers.Dense(num_classes, activation='softmax')])
        elif "ResNet" in model_name:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras import layers, models
            base = ResNet50(input_shape=(224,224,3), include_top=False, weights=None)
            model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(128, activation='relu'), layers.Dropout(0.4), layers.Dense(num_classes, activation='softmax')])
        else:
            raise e
            
    model = compile_model(model)
    
    # Callbacks
    model_file = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_file, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    
    # Configure epochs/steps for quick_run validation
    epochs = 1 if args.quick_run else args.epochs
    
    # Train
    start_time = time.time()
    if args.quick_run:
        history = model.fit(
            train_ds.take(2),
            validation_data=val_ds.take(2),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    # Load best weights
    if os.path.exists(model_file):
        try:
            model = tf.keras.models.load_model(model_file)
            print("Loaded best saved model weights for evaluation.")
        except Exception as e:
            print(f"Could not load saved model h5 file (likely due to custom layers format). Using active weights. Error: {e}")
            
    # Evaluation on Test Set
    print("Evaluating model on test set...")
    # Predict
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Plot history and confusion matrix
    plot_history(history, model_name, outputs_dir)
    plot_confusion_matrix(test_labels, y_pred, classes, model_name, outputs_dir)
    
    # Get metrics
    metrics = evaluate_predictions(test_labels, y_pred, classes, model_name)
    metrics["TrainingTime_Sec"] = elapsed_time
    
    return metrics

def main():
    args = parse_args()
    
    # Setup base directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_dir, args.data_dir)
    models_dir = os.path.join(project_dir, "models")
    outputs_dir = os.path.join(project_dir, "outputs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Check if data directory exists and has subdirectories
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: Data directory '{data_dir}' is empty or does not exist.")
        print("Please run 'python generate_dataset.py' to generate a synthetic dataset, or place a PlantVillage dataset in the data folder.")
        return
        
    print(f"Loading and splitting dataset from: {data_dir}")
    train_ds, val_ds, test_ds, info = prepare_datasets(
        data_dir=data_dir,
        img_size=(224, 224),
        batch_size=args.batch_size
    )
    
    classes = info["classes"]
    test_labels = info["test_labels"]
    
    # Save classes list to outputs directory for the app to access later
    with open(os.path.join(outputs_dir, "classes.txt"), "w") as f:
        f.write("\n".join(classes))
        
    all_metrics = []
    
    # Define models to train
    models_to_train = [
        ("Custom CNN", lambda num_classes: build_custom_cnn(num_classes=num_classes)),
        ("MobileNetV2", lambda num_classes: build_mobilenetv2_transfer(num_classes=num_classes)),
        ("ResNet50", lambda num_classes: build_resnet50_transfer(num_classes=num_classes))
    ]
    
    for model_name, model_fn in models_to_train:
        try:
            metrics = train_and_evaluate_model(
                model_name=model_name,
                model_fn=model_fn,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                classes=classes,
                test_labels=test_labels,
                args=args,
                models_dir=models_dir,
                outputs_dir=outputs_dir
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save comparison dataframe
    if all_metrics:
        save_model_comparison(all_metrics, outputs_dir)
        print("\n=== Model Comparison Summary ===")
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        print(df.to_string(index=False))
        
    print("\nTraining workflow completed successfully!")

if __name__ == "__main__":
    main()
