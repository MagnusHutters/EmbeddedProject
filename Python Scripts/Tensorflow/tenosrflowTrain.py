import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths for your datasets
train_dir = 'datasets/Training'
validation_dir = 'datasets/Validation'


# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255)  # Set to grayscale

validation_datagen = ImageDataGenerator(
    rescale=1./255)  # Set to grayscale

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 32),  # Change to 120x160
    batch_size=20,
    class_mode='categorical',
    color_mode='grayscale',  # Set to grayscale
    shuffle = True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(48, 32),  # Change to 120x160
    batch_size=20,
    class_mode='categorical',
    color_mode='grayscale',  # Set to grayscale
    shuffle=True
)


# Model Creation
model = Sequential([
    Conv2D(6, (3, 3), activation='relu',use_bias=False, input_shape=(48, 32, 1)),  # Update for grayscale
    MaxPooling2D(2, 2),
    Conv2D(12, (3, 3), use_bias=False, activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(24, (3, 3), use_bias=False, activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(24, use_bias=False, activation='relu'),
    Dense(3, use_bias=False, activation='softmax')  # 3 for three categories
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


print(model.layers[0].input_shape)
#exit()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=300,  # depends on your dataset
    epochs=4,
    validation_data=validation_generator,
    validation_steps=30  # depends on your dataset
)

# Save the model
model.save('object_classification_model_no_bias.h5')
