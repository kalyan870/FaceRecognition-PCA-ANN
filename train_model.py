import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = "dataset/faces"

X = []
y = []
label_names = []

print("Loading Dataset...")

for idx, person_name in enumerate(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person_name)

    if os.path.isdir(person_path):
        label_names.append(person_name)

        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (100, 100))
                X.append(img.flatten())
                y.append(idx)

X = np.array(X)
y = np.array(y)

print(f"Total Images Loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Applying PCA...")

n_components = min(40, X_train.shape[0] - 1)
pca = PCA(n_components=n_components, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Training ANN Model...")

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=42
)

model.fit(X_train_pca, y_train)

predictions = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "model.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(label_names, "labels.pkl")

print("Model Saved Successfully!")

plt.figure(figsize=(6,4))
plt.bar(["Accuracy"], [accuracy * 100])
plt.ylim(0,100)
plt.ylabel("Accuracy %")
plt.title("Face Recognition Model Accuracy")

os.makedirs("outputs/graphs", exist_ok=True)
plt.savefig("outputs/graphs/accuracy_graph.png")

print("Accuracy Graph Saved!")

os.makedirs("outputs/eigenfaces", exist_ok=True)

eigenfaces = pca.components_.reshape((-1, 100, 100))

for i in range(min(5, eigenfaces.shape[0])):
    plt.figure()
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.axis('off')
    plt.savefig(f"outputs/eigenfaces/eigenface_{i}.png")
    plt.close()

print("Eigenfaces Saved!")

print("Training Completed Successfully!")
