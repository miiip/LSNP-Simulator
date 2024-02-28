import time
import Model
import Generator
import Classifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from Crypto.Util import number
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import ElGamal
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256
import random

def display_image(index_to_display):
    digits = load_digits()
    image = digits.images[index_to_display]
    image = image / 255.0
    print(list(image[0]))
    image = image * 0.00000000001
    print(list(image[0]))

    #image = np.clip(image, 0.0, 1.0)
    label = digits.target[index_to_display]
    image = image * 255
    print(list(image[0]))
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Digit: {label}")
    plt.axis('off')  # Turn off axis labels
    plt.show()

def load_dataset(dataset_name):
    if dataset_name == 'fashion':
        mnist_fashion = fetch_openml(name="Fashion-MNIST", version=2)
        x, target = np.array(mnist_fashion.data), np.array(mnist_fashion.target, dtype=int)
    if dataset_name == 'iris':
        iris = load_iris()
        x = iris.data
        target = iris.target
    elif dataset_name == 'breast':
        breast = load_breast_cancer()
        x = breast.data
        target = breast.target
    elif dataset_name == 'wine':
        wine = load_wine()
        x = wine.data
        target = wine.target
    else:
        digits = load_digits()
        x = digits.data
        target = digits.target

    #x = PolynomialFeatures(degree=4).fit_transform(x)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    target = np.array(target).reshape(-1, 1)
    encoder = OneHotEncoder()
    y = encoder.fit_transform(target).toarray()
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y


def generate_keys():
    private_key = random.randint(2, 2 ** 2048)

    # Generate a public key based on the private key (for simplicity)
    public_key = pow(2, private_key, 2 ** 2048)
    return private_key, public_key

def float_to_int(f):
    return int(f * 1e6)

def int_to_float(i):
    return i / 1e6

def encrypt(public_key, plaintext):
    plaintext_int = float_to_int(plaintext)

    # Generate a random ephemeral key
    k = random.randint(2, 2**2048)

    # Compute c1 and c2
    c1 = pow(2, k, 2**2048)
    c2 = (plaintext_int * pow(public_key, k, 2**2048)) % 2**2048

    return c1, c2

def decrypt(private_key, ciphertext):
    c1, c2 = ciphertext

    # Compute the shared secret
    shared_secret = pow(c1, private_key, 2**2048)

    # Compute the modular inverse of the shared secret
    inverse_shared_secret = pow(shared_secret, -1, 2**2048)

    # Decrypt the ciphertext
    plaintext_int = (c2 * inverse_shared_secret) % 2**2048

    return int_to_float(plaintext_int)
def experiment(dataset):
    train_x, test_x, train_y, test_y = load_dataset(dataset)
    maxgen = 80

    classifier = Classifier.Classifier(train_x, train_y)
    model = Model.Model()

    model_w, model_l = classifier.train_classification_softmax(train_x, train_y)
    predict = model.classification(model_w, model_l, train_x, train_y)
    acc_train_last = model.accuracy_rate(predict, train_y)
    loss = []
    it = []
    for i in range(maxgen):
        it.append(i+1)
        model_w, model_l = classifier.train_classification_softmax(train_x, train_y)
        predict = model.classification(model_w, model_l, train_x, train_y)
        loss.append(Model.Model().loss(predict, train_y))
        acc_train = model.accuracy_rate(predict, train_y)
        if np.abs(acc_train - acc_train_last) < 0.00001:
            print("Max iter: ", i + 2)
            break
        acc_train_last = acc_train
    print(it)
    plt.scatter(it, loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    predict = model.classification(model_w, model_l, test_x, test_y)
    print('Samples', len(test_x))
    accuracy = accuracy_score(test_y, predict)
    precision = precision_score(test_y, predict, average='weighted')
    recall = recall_score(test_y, predict, average='weighted')
    f1 = f1_score(test_y, predict, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")



experiment('fashion')
