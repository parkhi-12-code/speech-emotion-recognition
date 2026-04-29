import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# ========================
# REPRODUCIBILITY
# ========================
np.random.seed(42)
tf.random.set_seed(42)

# ========================
# CONFIG
# ========================
DATA_PATH   = "data/Ravdess/"
SAMPLE_RATE = 22050
MFCCS       = 40
MAX_LEN     = 174

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ========================
# FEATURE EXTRACTION
# No augmentation here — training only (after split)
# ========================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCCS)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc   = np.concatenate([mfcc, delta, delta2], axis=0)  # shape: (120, T)

    mfcc = librosa.util.fix_length(mfcc, size=MAX_LEN)

    return mfcc  # shape: (120, 174)


# ========================
# AUGMENTATION — training data only
# ========================
def augment_train(X_train):
    noise = np.random.randn(*X_train.shape)
    return X_train + 0.003 * noise


# ========================
# LOAD DATA
# ========================
X, y = [], []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            label = emotion_map.get(emotion_code)

            if label:
                path     = os.path.join(root, file)
                features = extract_features(path)
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

# reshape for CNN — shape: (N, 120, 174, 1)
X = X[..., np.newaxis]

# label encoding
le        = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========================
# TRAIN / TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ========================
# NORMALIZATION — train stats only, applied to both
# ========================
mean    = X_train.mean()
std     = X_train.std() + 1e-6
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# ========================
# AUGMENTATION — training set only
# ========================
X_train = augment_train(X_train)

# ========================
# CLASS WEIGHTS — computed from data
# ========================
weights       = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# ========================
# MODEL
# KEY FIX: GlobalAveragePooling2D replaces Flatten
# This reduces the dense input from 75,264 → 128
# preventing the 9.6M parameter explosion that caused collapse
# ========================
model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]),

    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3 — added for the larger (120×174) input
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # THE CRITICAL FIX: GlobalAveragePooling instead of Flatten
    # Flatten produced 75,264 values → 9.6M params → collapse
    # GlobalAveragePooling produces 128 values → ~16K params → learnable
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),

    layers.Dense(len(le.classes_), activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# EARLY STOPPING
# ========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

# ========================
# TRAIN
# ========================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights
)

# ========================
# EVALUATION
# ========================
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0          # suppresses the UndefinedMetricWarning
))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — SER (RAVDESS)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],    label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],    label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ========================
# SAMPLE PREDICTION
# ========================
sample           = np.expand_dims(X_test[0], axis=0)
prediction       = model.predict(sample)
predicted_label  = le.inverse_transform([np.argmax(prediction)])

print("\nSample Prediction:", predicted_label[0])
print("Actual Label     :", le.inverse_transform([y_test[0]])[0])