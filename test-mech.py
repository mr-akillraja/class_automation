import cv2, glob, json, os, numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

plt.ion()  # interactive mode

# ================= GPU SETUP =================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {[g.name for g in gpus]}")
else:
    print("⚠️ No GPU — running on CPU")

DEVICE = "/GPU:0" if gpus else "/CPU:0"

# ================= PATHS =================
BASE_DIR     = "/home/akill-sud/Documents/projects/mechtronics"
WEIGHTS_PATH = os.path.join(BASE_DIR, "final_model.weights.h5")
BEST_PATH    = os.path.join(BASE_DIR, "best_model.weights.h5")
CLASS_PATH   = os.path.join(BASE_DIR, "class_indices.json")

# ================= BUILD MODEL =================
print("🔧 Building model...")
inputs = Input(shape=(224, 224, 3))
base   = EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)

x = GlobalAveragePooling2D()(base.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
out = Dense(3, activation="softmax")(x)

model = Model(inputs=inputs, outputs=out)

# ================= LOAD WEIGHTS =================
if os.path.exists(WEIGHTS_PATH):
    wpath = WEIGHTS_PATH
elif os.path.exists(BEST_PATH):
    wpath = BEST_PATH
    print("⚠️ Using best_model.weights.h5")
else:
    raise FileNotFoundError(f"❌ No weights found in {BASE_DIR}")

model.load_weights(wpath)
print(f"✅ Weights loaded: {wpath}")

# ================= LOAD CLASSES =================
if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH) as f:
        idx = json.load(f)
    class_names = [k for k, v in sorted(idx.items(), key=lambda x: x[1])]
else:
    class_names = ["accept", "discon", "porosity"]

print(f"✅ Classes: {class_names}")

# ================= SETTINGS =================
IMG_SIZE = 224

CLASS_BGR = {
    class_names[0]: (0, 200, 0),
    class_names[1]: (0, 165, 255),
    class_names[2]: (0, 0, 255),
}

# ================= CAMERA =================
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera")

# ================= PREDICTION =================
def predict(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    with tf.device(DEVICE):
        preds = model.predict(img, verbose=0)

    cid = int(np.argmax(preds[0]))
    conf = float(preds[0][cid])

    scores = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    return class_names[cid], conf, scores

# ================= DISPLAY =================
fig, ax = plt.subplots()

print("\n🎥 Live feed started — CTRL+C to quit\n")

last_label, last_color, last_scores = None, (255, 255, 255), {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed")
        break

    display = frame.copy()

    if last_label:
        cv2.rectangle(display, (0, 0), (640, 70), (0, 0, 0), -1)
        cv2.putText(display,
                    f"{last_label.upper()}  {last_scores.get(last_label, 0):.1%}",
                    (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    last_color,
                    3,
                    cv2.LINE_AA)

    # Convert to RGB for matplotlib
    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    ax.clear()
    ax.imshow(rgb)
    ax.set_title("Weld Inspector")
    ax.axis("off")

    plt.pause(0.001)

    # Auto predict every few frames (since no keyboard input)
    label, conf, scores = predict(frame)
    last_label, last_color, last_scores = label, CLASS_BGR[label], scores

    print(f"➤ {label.upper()} ({conf:.1%})")

# ================= CLEANUP =================
cap.release()
plt.close()
print("✅ Done")