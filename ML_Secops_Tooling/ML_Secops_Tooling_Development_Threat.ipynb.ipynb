import logging
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from art.attacks.evasion import FastGradientMethod
from art.attacks.poisoning import PoisoningAttackSVM
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import SklearnClassifier
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.extraction import CopycatCNN
from tensorflow.keras import layers, models

# Set up logging
logging.basicConfig(filename='ml_model_threats_model_development.log', level=logging.INFO)

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_train, y_val, y_train = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train an SVM model
model = SVC(probability=True)
model.fit(X_train, y_train.argmax(axis=1))

# Set clip values based on the feature range of the Iris dataset (min and max values)
clip_values = (X.min(), X.max())

# Wrap in ART classifier with clip_values
classifier = SklearnClassifier(model=model, clip_values=clip_values)

# Define parameters for the poisoning attack
step = 0.1
eps = 0.1
max_iter = 20

# Create a poisoning attack
poisoning_attack = PoisoningAttackSVM(
    classifier=classifier,
    x_train=X_train,
    y_train=y_train,
    x_val=X_val,
    y_val=y_val,
    step=step,
    eps=eps,
    max_iter=max_iter
)

# Generate poisoned training data
X_train_poisoned, y_train_poisoned = poisoning_attack.poison(X_train, y_train)

# Re-train model with poisoned data
model_poisoned = SVC(probability=True)
model_poisoned.fit(X_train_poisoned, y_train_poisoned.argmax(axis=1))

# 2. Adversarial Evasion (FGSM Attack)
attack_evasion = FastGradientMethod(estimator=classifier, eps=0.1)
X_test_adv = attack_evasion.generate(x=X_test)

# 3. Membership Inference Attack (Inference Leakage)
membership_inference = MembershipInferenceBlackBox(estimator=classifier)
membership_inference.fit(X_train, y_train, X_test, y_test)
inferred_train = membership_inference.infer(X_train, y_train)
inferred_test = membership_inference.infer(X_test, y_test)

# 4. Model Stealing (Copycat Attack)
def create_thieved_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Get input shape and number of classes
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

# Create and compile the thieved model
thieved_model = create_thieved_model(input_shape, num_classes)
thieved_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create optimizer for ART classifier
optimizer = tf.keras.optimizers.Adam()

# Wrap the thieved model in ART's TensorFlowV2Classifier with optimizer
thieved_classifier = TensorFlowV2Classifier(
    model=thieved_model, 
    nb_classes=num_classes, 
    input_shape=input_shape, 
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=optimizer
)
victim_model = classifier

# Create the CopycatCNN attack instance
stealing_attack = CopycatCNN(classifier=victim_model, nb_epochs=10)

# Perform the model stealing attack
stolen_model = stealing_attack.extract(x=X_train, y=y_train, thieved_classifier=thieved_classifier)

# Log the threats
accuracy_original = np.sum(np.argmax(classifier.predict(X_test), axis=1) == y_test.argmax(axis=1)) / len(y_test)
accuracy_poisoned = np.sum(model_poisoned.predict(X_test) == y_test) / len(y_test)
accuracy_adv = np.sum(np.argmax(classifier.predict(X_test_adv), axis=1) == y_test.argmax(axis=1)) / len(y_test)

logging.info("=== ML Threat Detection on Iris Dataset ===")
if accuracy_poisoned < accuracy_original:
    logging.info(f"Data Poisoning detected! Original accuracy: {accuracy_original*100:.2f}%, Poisoned accuracy: {accuracy_poisoned*100:.2f}%")
if accuracy_adv < accuracy_original:
    logging.info(f"Model Evasion detected! Original accuracy: {accuracy_original*100:.2f}%, Adversarial accuracy: {accuracy_adv*100:.2f}%")
logging.info(f"Membership inference detected on training set: {np.mean(inferred_train)}")
logging.info(f"Membership inference detected on test set: {np.mean(inferred_test)}")
