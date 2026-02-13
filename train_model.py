"""
Deep Learning Model Training for Diabetic Retinopathy Detection
Uses Transfer Learning with Inception V3, ResNet50, and Xception V3
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3, ResNet50, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

def create_model(base_model_name='xception'):
    """
    Create a transfer learning model using pre-trained base model
    
    Args:
        base_model_name: 'inception', 'resnet50', or 'xception'
    
    Returns:
        Compiled Keras model
    """
    # Select base model
    if base_model_name == 'inception':
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    elif base_model_name == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    else:  # xception
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def create_data_generators():
    """Create data generators with augmentation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # No augmentation for validation/test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def train_model(model_name='xception'):
    """Train the model"""
    print(f"Creating {model_name} model...")
    model = create_model(model_name)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    print("Training model...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(f'models/{model_name}_final.h5')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train Xception model (as mentioned in the project)
    print("=" * 50)
    print("Training Xception Model")
    print("=" * 50)
    model_xception, history_xception = train_model('xception')
    
    # Optionally train other models
    # print("\n" + "=" * 50)
    # print("Training ResNet50 Model")
    # print("=" * 50)
    # model_resnet, history_resnet = train_model('resnet50')
    
    # print("\n" + "=" * 50)
    # print("Training Inception V3 Model")
    # print("=" * 50)
    # model_inception, history_inception = train_model('inception')
    
    print("\nTraining completed!")
