import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense

# Set the root directory for the dataset
root_dir = 'path/to/your/directory'

# Create a list to store the image metadata
image_metadata = []

# Iterate through each subdirectory in the root directory
for dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, dir)):
        # Get the directory name (e.g., 'healthy', 'diseased')
        label = dir
        
        # Iterate through each image file in the directory
        for file in os.listdir(os.path.join(root_dir, dir)):
            if file.endswith('.jpg'):
                # Get the image path and name
                img_path = os.path.join(root_dir, dir, file)
                img_name = os.path.basename(img_path)
                
                # Load the image using OpenCV
                img = cv2.imread(img_path)
                
                # Convert the image to grayscale and scale pixel values to [0, 1]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = gray / 255.0
                
                # Add the image metadata to the list
                image_metadata.append({
                    'image_name': img_name,
                    'label': label,
                    'image': gray,
                    'angle': 0,  # assume 0° angle for now
                    'lighting': 'natural'  # assume natural light for now
                })

# Convert the image metadata list to a Pandas DataFrame
df = pd.DataFrame(image_metadata)

# Save the DataFrame to a CSV file
df.to_csv('image_metadata.csv', index=False)

# Load the saved CSV file
df = pd.read_csv('image_metadata.csv')

# Split the dataset into training and testing sets (e.g., 80% for training and 20% for testing)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess the images using OpenCV (e.g., resizing, normalizing)
scaler = MinMaxScaler()
train_images = []
for i, row in train_df.iterrows():
    img = row['image']
    img = cv2.resize(img.astype(np.float32), (224, 224))  
    img = np.reshape(img, (1, 224, 224, 1))
    train_images.append(img)
train_images = np.array(train_images)

test_images = []
for i, row in test_df.iterrows():
    img = row['image']
    img = cv2.resize(img.astype(np.float32), (224, 224))  
    img = np.reshape(img, (1, 224, 224, 1))
    test_images.append(img)
test_images = np.array(test_images)

# One-hot encode the labels
labels_train = pd.get_dummies(train_df['label']).values
labels_test = pd.get_dummies(test_df['label']).values

# Scale the training images using MinMaxScaler
train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

# Train a convolutional neural network (CNN) model using Keras or TensorFlow
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(len(set(df['label'])), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the scaled images and one-hot encoded labels
model.fit(train_images_scaled, labels_train, epochs=10)

# Evaluate the model on the testing set
loss_test, accuracy_test = model.evaluate(test_images_scaled)
print(f'Test accuracy: {accuracy_test:.2f}')

# Use the trained model to make predictions on new images
new_img_path = 'path/to/new/image.jpg'
new_img = cv2.imread(new_img_path)
new_img_scaled = scaler.transform(np.array([new_img / 255.0]).reshape(1, 224, 224, 1))
new_label_encoded = pd.get_dummies([os.path.basename(new_img_path)]).values[0]

prediction = model.predict(new_img_scaled)
print(f'Prediction: {np.argmax(prediction)}')