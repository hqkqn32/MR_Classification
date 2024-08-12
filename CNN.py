import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
#%%
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'your_path\\Training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'Your_path\\Training',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    'Your_Path\\Testing',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

#%%
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')  # çıkış katmanında 4 sınıf olduğu için
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#%%
# epoch sayısı değişebilir
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print('Test accuracy:', test_acc)

model.save('tumor_classification_model.h5')

#%%

import matplotlib.pyplot as plt

# Eğitim ve doğrulama kayıplarını plot etme
import matplotlib.pyplot as plt

# Eğitim ve doğrulama kayıplarını plot etme (yüzde olarak)
plt.plot([x * 100 for x in history.history['loss']], label='Training Loss')
plt.plot([x * 100 for x in history.history['val_loss']], label='Validation Loss')
plt.title('Training and Validation Loss (Percentage)')
plt.xlabel('Epochs')
plt.ylabel('Loss (%)')
plt.legend()
plt.show()



# Eğitim ve doğrulama başarı oranlarını plot etme (yüzde olarak)
plt.plot([x * 100 for x in history.history['accuracy']], label='Training Accuracy')
plt.plot([x * 100 for x in history.history['val_accuracy']], label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Percentage)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


