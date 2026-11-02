import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input


# ==========================================
# MODEL 1: SIMPLE CNN (Baseline)
# ==========================================
def build_simple_cnn(input_shape=(224, 224, 3), num_classes=17):
    """
    A lightweight model to test if the pipeline works.
    Structure: 3 Conv Blocks -> Flatten -> Dense
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="Simple_CNN")
    return model

# ==========================================
# MODEL 2: DEEP CNN (VGG-Style Custom)
# ==========================================
def build_deep_cnn(input_shape=(224, 224, 3), num_classes=17):
    """
    Imitates VGG architecture but defined manually.
    Uses blocks of 2 Conv layers followed by pooling.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x) # Better than Flatten for deep nets
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="Deep_CNN_VGG_Style")
    return model

# ==========================================
# MODEL 3: RESNET-STYLE (Custom Residuals)
# ==========================================
def residual_block(x, filters, kernel_size=3, stride=1):
    """Helper function to create a Residual Block (Skip Connection)"""
    shortcut = x
    
    # First Conv
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Second Conv
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def build_resnet_custom(input_shape=(224, 224, 3), num_classes=17):
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Stack Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2) # Downsample
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2) # Downsample
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="ResNet_Custom")
    return model

# ==========================================
# MODEL 4: VISION TRANSFORMER (ViT) - MANDATORY
# ==========================================
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    # --- THIS IS THE MISSING PART THAT CAUSED THE CRASH ---
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    # --- THIS IS ALSO NEEDED FOR SAVING ---
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units, # We access units from the Dense layer
        })
        return config

# ==========================================
# MODEL 4: COMPACT CONVOLUTIONAL TRANSFORMER (CCT)
# ==========================================
def build_vision_transformer(input_shape=(224, 224, 3), num_classes=17):
    inputs = Input(shape=input_shape)
    
    # 1. Normalization
    x = layers.Rescaling(1./255)(inputs)

    # 2. CCT Tokenizer
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x) 
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x) 
    
    x = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x) 
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x) 

    # 3. Reshape
    target_shape = (-1, 128) 
    x = layers.Reshape(target_shape)(x)
    
    # 4. Positional Embeddings
    seq_len = 196 
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=128)(positions)
    x = x + pos_embedding

    # 5. Transformer Encoder
    for _ in range(2):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(128, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(128)(x3)
        x = layers.Add()([x3, x2])

    # 6. Sequence Pooling
    attention_weights = layers.Dense(1, activation='softmax')(x)
    weighted_representation = tf.matmul(attention_weights, x, transpose_a=True)
    weighted_representation = tf.squeeze(weighted_representation, -2)

    # 7. Classification Head
    outputs = layers.Dense(num_classes, activation='softmax')(weighted_representation)
    
    # IMPORTANT: THIS NAME DETERMINES THE FILENAME
    model = models.Model(inputs, outputs, name="CCT_Transformer") 
    return model