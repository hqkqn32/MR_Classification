import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Modeli yükleme
model = tf.keras.models.load_model('tumor_classification_model.h5')

# Görüntüyü yükleme ve işleme
img_path = 'test.png'
img = image.load_img(img_path, target_size=(150, 150))  # Modelin beklediği boyut
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalizasyon (eğer model eğitilirken kullanıldıysa)

# Modelin özetini görme
model.summary()

# Tahmin yapma
prediction = model.predict(img_array)
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Sıralama: 0, 1, 2, 3
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

# Tahmin sonucunu terminale yazdırma
print(f"Tahmin edilen sınıf: {predicted_class}")
print(f"Güven yüzdesi: {confidence:.2f}%")
