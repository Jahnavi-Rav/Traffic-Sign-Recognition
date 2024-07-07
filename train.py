import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from model import create_model

# Load preprocessed data
data = np.load('data/train_data.npz')
X_train = data['X_train']
y_train = data['y_train']

data = np.load('data/test_data.npz')
X_test = data['X_test']
y_test = data['y_test']

# Set constants
N_CLASSES = 43
RESIZED_IMAGE = (32, 32, 1)
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10

# Create model
model = create_model(RESIZED_IMAGE, N_CLASSES, LEARNING_RATE)

# Train model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

# Predictions
y_test_pred = model.predict(X_test)
y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)

# Classification report and confusion matrix
print(classification_report(y_test_true_classified, y_test_pred_classified))
cm = confusion_matrix(y_test_true_classified, y_test_pred_classified)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.tight_layout()
plt.show()

# Log2 version of confusion matrix
plt.imshow(np.log2(cm + 1), interpolation='nearest', cmap=plt.get_cmap("tab20"))
plt.colorbar()
plt.tight_layout()
plt.show()
