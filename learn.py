from tensorflow.keras import layers, models
import numpy as np
import cv2

train_images = []
test_images = []

def img(idx, list):
    for i, f in enumerate(idx):
        img = cv2.imread(f)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        list.append(img)

train_images_idx, test_images_idx, train_labels, test_labels = np.load('data/all_pack.npy', allow_pickle=True)#내가 만든 파일을 불러온다

img(train_images_idx, train_images)
img(test_images_idx, test_images)

test_images = np.array(test_images)
train_images = np.array(train_images)

# train_images = train_images_idx.reshape((len(train_images_idx), 512, 512, 3))
# test_images = test_images_idx.reshape((len(test_images_idx), 512, 512, 3))
# train_images = train_images_idx.reshape(12, 512, 512, 3)
# test_images = test_images_idx.reshape(4, 512, 512, 3)

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=40)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

image_class = ["jung","rain"]
predictions = model.predict(train_images)
predictions_test = model.predict(test_images)

for i in range(len(test_images)):#결과 예측값 저장
    a = cv2.imread(test_images_idx[i])
    cv2.imwrite('result/{}_{:2.0F}%.jpg'.format(image_class[np.argmax(predictions_test[i])],100 * np.max(predictions_test[i])), a)
    print(image_class[np.argmax(predictions_test[i])], image_class[test_labels[i]], 100 * np.max(predictions_test[i]))
model.save('data/all_pack_40.h5')