import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=5):
    """
    Builds a custom CNN model.
    A block of convolutional, pooling, batch normalization, and dropout layers
    designed for agricultural image feature extraction.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_mobilenetv2_transfer(input_shape=(224, 224, 3), num_classes=5, fine_tune_at=100):
    """
    Builds a MobileNetV2 transfer learning model.
    By default, freezes base layers except the top fine_tune_at layers if specified.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = True
    
    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model = models.Sequential([
        # MobileNetV2 base
        base_model,
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_resnet50_transfer(input_shape=(224, 224, 3), num_classes=5, fine_tune_at=140):
    """
    Builds a ResNet50 transfer learning model.
    """
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    
    # Freeze initial layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=1e-4):
    """Compiles the Keras model with Adam optimizer and cross entropy loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
