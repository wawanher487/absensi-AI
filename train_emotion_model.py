import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def create_emotion_model(num_classes=4):
    """
    Membuat arsitektur model CNN untuk deteksi emosi.
    - Input: Gambar grayscale 48x48 pixel.
    - Output: 4 kelas emosi (Marah, Senang, Netral, Sedih).
    """
    model = Sequential(name="Emotion_Detection_Model")
    
    # Blok Konvolusi 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Blok Konvolusi 2
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Lapisan Fully Connected
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    
    # Lapisan Output (4 kelas sesuai permintaan Anda)
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_emotion_detection():
    """
    Melatih model deteksi emosi dari dataset FER2013.
    """
    print("[INFO] Memulai training deteksi emosi...")
    # PASTIKAN PATH INI BENAR menunjuk ke folder 'train' dari dataset FER2013
    dataset_path = "train"
    model_path = "emotion_model.h5"
    
    # Kelas yang akan digunakan sesuai permintaan Anda
    emotion_labels = ['angry', 'happy', 'neutral', 'sad']
    # emotion_labels = ['Marah', 'Senang', 'Netral', 'Sedih']
    
    # Generator data untuk augmentasi dan preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Menggunakan 20% data untuk validasi
    )

    # Siapkan generator untuk data training
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        classes=emotion_labels,
        subset='training'
    )

    # Siapkan generator untuk data validasi
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        classes=emotion_labels,
        subset='validation'
    )

    # Buat model
    model = create_emotion_model(num_classes=len(emotion_labels))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callback untuk menyimpan HANYA model terbaik selama training
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Mulai training
    print("[INFO] Melatih model...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 64,
        epochs=50, # Anda bisa menambah jumlah epoch untuk akurasi yang lebih baik
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 64,
        callbacks=[checkpoint]
    )
    print(f"[SUCCESS] Training deteksi emosi selesai. Model terbaik disimpan di {model_path}")

if __name__ == "__main__":
    train_emotion_detection()