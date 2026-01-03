import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

img_size = 224
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "archive/Training",
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = test_datagen.flow_from_directory(
    "archive/Testing",
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dense(4,activation='softmax')
])

model.compile(optimizer=Adam(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=12)

model.save("brain_tumor_model_multi.h5")
print("Model Trained & Saved Successfully 🎯")
