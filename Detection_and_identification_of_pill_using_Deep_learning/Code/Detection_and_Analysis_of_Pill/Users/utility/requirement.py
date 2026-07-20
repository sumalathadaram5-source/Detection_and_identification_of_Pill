import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dropout, Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

def load_data(train_csv_path, test_csv_path):
    """Load the training and testing datasets from CSV files."""
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Ensure labels are integers
    train_df['label'] = pd.Categorical(train_df['label']).codes
    test_df['label'] = pd.Categorical(test_df['label']).codes
    
    return train_df, test_df

def check_data_balance(train_df):
    """Visualize the distribution of training labels."""
    sns.histplot(train_df['label'], kde=False).set(title="Distribution of Training Labels")
    plt.show()

def preprocess_images(file_paths, base_path, target_size=(64, 64)):
    """Preprocess images by loading and resizing them."""
    images = []
    for file in file_paths:
        img = load_img(base_path / file, color_mode='grayscale')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img = np.array(img)
        images.append(img)
    return np.array(images)

def build_model(input_shape, num_classes):
    """Build and compile the CNN model."""
    inputs = Input(input_shape)
    X = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((3, 3))(X)

    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Flatten()(X)

    dense_1 = Dense(256, activation='relu')(X)
    dropout_1 = Dropout(0.4)(dense_1)

    dense_2 = Dense(128, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.4)(dense_2)

    output = Dense(num_classes, activation='softmax', name='pill_output')(dropout_2)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def plot_history(history):
    """Plot training history for accuracy and loss."""
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def evaluate_model(model, x_test, y_test):
    """Evaluate the model on test data and print classification report."""
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    pill_report = classification_report(y_test, y_pred)
    pill_accuracy = accuracy_score(y_test, y_pred)
    #precision, recall = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    
    print("Pill Identification Classification Report:\n", pill_report)
    print("Pill Identification Accuracy: {:.2f}%".format(pill_accuracy * 100))
    
    return pill_accuracy

def main():
    # Paths to datasets
    train_csv_path = Path(r"C:\Users\UPENDRA\Downloads\pilldata\Training_set.csv")
    test_csv_path = Path(r"C:\Users\UPENDRA\Downloads\pilldata\Testing_set.csv")
    train_path = Path(r"C:\Users\UPENDRA\Downloads\pilldata\train")
    
    # Load data
    train_df, test_df = load_data(train_csv_path, test_csv_path)
    
    # Check data balance
    check_data_balance(train_df)

    # Preprocess images
    x_train = preprocess_images(train_df['filename'], train_path)
    x_train = x_train.reshape(len(x_train), 64, 64, 1) / 255.0
    y_train = np.array(train_df['label'], dtype=np.int32)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=39, stratify=y_train)

    # Build the model
    input_size = (64, 64, 1)
    model = build_model(input_size, len(train_df['label'].unique()))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=1, validation_split=0.1)

    # Save the model
    model.save(r'C:\Users\UPENDRA\Downloads\pilldata\model.h5')
    
    # Plot training history
    plot_history(history)

    # Evaluate the model
    pill_accuracy= evaluate_model(model, x_test, y_test)
    
    return pill_accuracy

#########################################
#prediction code

def load_trained_model(model_path):
    return load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale')
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img.reshape(64, 64, 1)  # Reshape for model input
    img = img / 255.0  # Normalize the image
    return img

# Function to predict the pill
def predict_pill(pred):
    
    pred_class = np.argmax(pred, axis=-1)
    print(pred_class)
    # Label mapping
    label_map = {
        0: 'Alaxan',
        1: 'Medicol',
        2: 'Bioflu',
        3: 'Kremil S',
        4: 'DayZinc',
        5: 'Biogesic',
        6: 'Fish Oil',
        7: 'Neozep',
        8: 'Decolgen',
        9: 'Bactidol'
    }
    
    # Get the predicted label
    predicted_label = label_map[pred_class[0]]
    print(predicted_label)

    return predicted_label

# Function to display the image
def display_image(image_path):
    img = load_img(image_path, color_mode='grayscale')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# Main function to execute the prediction
def predictions(image_path):
    #print(image_path)
    processed_image = preprocess_image(image_path)

    # Make predictions using the model
    #input_size = (64, 64, 1)

    # Build the model
    #model = build_model(input_size)
    model_path=r'D:\data-point\Detection_and_Analysis_of_Pill\media\pilldata\model.h5'
    model=load_model(model_path)
    model.save(model_path)

    #sample = processed_image.reshape(( 64, 64, 1)) 
    val = model.predict( np.array([ processed_image ]) )  
    
    try:
        predicted_label = predict_pill(val)
        print(f'The predicted pill is: {predicted_label}')
        #display_image(image_path)
        return predicted_label
    except Exception as e:
        print(e)
        return e


