"""
Deep Learning Model Training for Diabetic Retinopathy Detection
Uses Transfer Learning with Inception V3, ResNet50, and Xception V3
Supports Multi-GPU Distributed Training for Faster Training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3, ResNet50, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision training for faster performance (uses less memory, trains faster)
# This uses float16 for computations while keeping float32 for stability
try:
    set_global_policy('mixed_float16')
    print("Mixed precision training enabled (float16)")
except:
    print("Mixed precision not available, using float32")

# Detect available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Found {len(gpus)} GPU(s)")

# Configure distributed training strategy
if len(gpus) > 1:
    # Multi-GPU training using MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy for {len(gpus)} GPUs")
    print(f"Replicas in sync: {strategy.num_replicas_in_sync}")
elif len(gpus) == 1:
    # Single GPU - use default strategy but still benefit from GPU
    strategy = tf.distribute.get_strategy()
    print("Using single GPU")
else:
    # CPU only
    strategy = tf.distribute.get_strategy()
    print("No GPU found, using CPU (training will be slow)")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'

def create_model(base_model_name='xception', strategy=None):
    """
    Create a transfer learning model using pre-trained base model
    Supports distributed training strategy
    
    Args:
        base_model_name: 'inception', 'resnet50', or 'xception'
        strategy: TensorFlow distribution strategy for multi-GPU
    
    Returns:
        Compiled Keras model
    """
    if strategy is None:
        strategy = tf.distribute.get_strategy()
    
    with strategy.scope():
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
        # Use float32 for final layer when using mixed precision
        outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs, outputs)
        
        # Compile with mixed precision compatible settings
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_data_generators(strategy=None):
    """Create data generators with augmentation
    Supports distributed training with proper batch size scaling
    """
    if strategy is None:
        strategy = tf.distribute.get_strategy()
    
    # Scale batch size based on number of GPUs
    num_replicas = strategy.num_replicas_in_sync
    effective_batch_size = BATCH_SIZE * num_replicas
    print(f"Effective batch size: {effective_batch_size} (base: {BATCH_SIZE} x {num_replicas} replicas)")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        shear_range=0.2
    )
    
    # No augmentation for validation/test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=effective_batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=effective_batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=effective_batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator, test_generator

def train_model(model_name='xception', strategy=None):
    """Train the model with distributed training support"""
    if strategy is None:
        strategy = tf.distribute.get_strategy()
    
    print(f"Creating {model_name} model with distributed strategy...")
    model = create_model(model_name, strategy)
    
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(strategy)
    
    # Callbacks with distributed training support
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False  # Save full model for distributed training
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Calculate steps per epoch for distributed training
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    validation_steps = val_gen.samples // val_gen.batch_size
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print("Training model with distributed strategy...")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        workers=4,  # Parallel data loading
        use_multiprocessing=True  # Speed up data loading
    )
    
    # Save final model
    model.save(f'models/{model_name}_final.h5')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

if __name__ == '__main__':
    # Create models and logs directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Print training configuration
    print("=" * 70)
    print("Diabetic Retinopathy Detection - Distributed Training")
    print("=" * 70)
    print(f"Strategy: {strategy}")
    print(f"Number of GPUs: {len(gpus)}")
    print(f"Mixed Precision: Enabled")
    print(f"Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * strategy.num_replicas_in_sync})")
    print("=" * 70)
    
    # Train Xception model (as mentioned in the project)
    print("\n" + "=" * 70)
    print("Training Xception Model with Distributed Computing")
    print("=" * 70)
    model_xception, history_xception = train_model('xception', strategy)
    
    # Optionally train other models
    # print("\n" + "=" * 70)
    # print("Training ResNet50 Model with Distributed Computing")
    # print("=" * 70)
    # model_resnet, history_resnet = train_model('resnet50', strategy)
    
    # print("\n" + "=" * 70)
    # print("Training Inception V3 Model with Distributed Computing")
    # print("=" * 70)
    # model_inception, history_inception = train_model('inception', strategy)
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print("\nTo view training progress, run: tensorboard --logdir=logs/")
