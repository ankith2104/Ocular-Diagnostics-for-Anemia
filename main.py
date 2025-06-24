import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications, callbacks
import tensorflowjs as tfjs

# Constants
IMG_SIZE = (224, 224)   # EfficientNetB0 default input size
BATCH_SIZE = 16         
EPOCHS = 50           

class AnemiaDetector:
    def __init__(self, train_dir, val_dir, test_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.model = None
        self.preprocess_input = None
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        
    def create_model(self):
        """Create EfficientNet-based model with Adam optimizer"""
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMG_SIZE, 3)
        )
        
        # Freeze base layers (transfer learning)
        base_model.trainable = False
        
        inputs = layers.Input(shape=(*IMG_SIZE, 3))
        x = base_model(inputs)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x) 
        
        model = models.Model(inputs, outputs)
        
        # Using Adam 
        optimizer = optimizers.Adam(learning_rate=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        self.preprocess_input = applications.efficientnet.preprocess_input
        return model
    
    def create_data_pipelines(self):
        """Create data generators with augmentation"""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='reflect'
        )
        
        val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_input
        )
        
        self.train_gen = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )
        
        self.val_gen = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_gen = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        return self.train_gen, self.val_gen, self.test_gen
    
    def train(self):
        """Train model with callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        ]
        
        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def evaluate(self):
        """Evaluate model on test set"""
        results = self.model.evaluate(self.test_gen, verbose=1)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3],
            'auc': results[4]
        }
        print("\nTest Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return metrics
    
    def save_model(self, model, filename):
        """Save model in multiple formats"""
        
        # 1. Save model weights
        weights_path = f"{filename}_weights.h5"
        model.save_weights(weights_path)
        print(f"Model weights saved as {weights_path}")
        
        # 2. Save complete model (HDF5 format)
        model_path = f"{filename}.h5"
        model.save(model_path)
        print(f"Model weights saved as {model_path}")
        
        # 3. Convert to TensorFlow.js
        tfjs_path = f"{filename}_tfjs"
        tfjs.converters.save_keras_model(model, tfjs_path)
        print(f"TensorFlow.js model saved in {tfjs_path} directory")
        
        # 4. Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = {tf.lite.Optimize.DEFAULT}
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        tflite_path = f"{filename}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved as {tflite_path}")
        
        print(f"All model formats saved successfully for: {filename}")
    
    def plot_history(self, history):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)  
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 2)  
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot precision
        plt.subplot(2, 2, 3)  
        plt.plot(history.history['precision'], label='Train Precision')
        plt.plot(history.history['val_precision'], label='Val Precision')
        plt.title('Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()

        # Plot recall
        plt.subplot(2, 2, 4)  
        plt.plot(history.history['recall'], label='Train Recall')
        plt.plot(history.history['val_recall'], label='Val Recall')
        plt.title('Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        plt.close()
    
    def run_pipeline(self):
        """End-to-end training and export"""
        print("Building model...")
        self.create_model()
        print("Setting up data...")
        self.create_data_pipelines()
        self.model.summary()
        print("Training...")
        history = self.train()
        print("Evaluating...")
        self.evaluate()
        print("Converting...")
        self.save_model(self.model, 'anemia_detector')
        print("Plotting history...")
        self.plot_history(history)
        print("Done!")

if __name__ == "__main__":
    detector = AnemiaDetector(
        train_dir='Conjuctiva/Training',
        val_dir='Conjuctiva/Validation',
        test_dir='Conjuctiva/Testing'
    )
    detector.run_pipeline() 