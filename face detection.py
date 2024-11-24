import cv2
import numpy as np
import threading
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Initialize the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the face recognizer if available
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    raise RuntimeError("LBPHFaceRecognizer is not available. Ensure you have installed opencv-contrib-python.")

# Function to train the recognizer with the uploaded images
def train_model(image_paths):
    faces = []
    labels = []

    for image_path in image_paths:
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Detect faces in the image
        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)

        # Check if any faces were detected
        if len(detected_faces) == 0:
            messagebox.showerror("Error", f"No face detected in the uploaded image: {image_path}")
            continue

        # Loop through the detected faces
        for (x, y, w, h) in detected_faces:
            face = img[y:y+h, x:x+w]
            faces.append(face)
            labels.append(0)  # Assuming a single user, use appropriate labels for multiple users

    if len(faces) > 0:
        # Train the recognizer with the faces
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save("trained_model.yml")
        messagebox.showinfo("Success", "Faces trained successfully!")

# Function to open the camera and detect faces
def open_camera():
    cap = cv2.VideoCapture(0)

    # Load the trained model
    face_recognizer.read("trained_model.yml")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            # Predict the label and confidence of the detected face
            label, confidence = face_recognizer.predict(face)

            # Adjust threshold for matching
            if confidence < 80:  # Adjust this threshold based on your testing
                cv2.putText(frame, "Face Matched", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face Not Matched", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("Face Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to upload an image
def upload_image():
    file_paths = filedialog.askopenfilenames()  # Allow multiple file selection
    if file_paths:
        for file_path in file_paths:
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img
        train_model(file_paths)

# Create the GUI
root = Tk()
root.title("Facial Recognition System")
root.geometry("500x400")

# Label for the uploaded image
image_label = Label(root)
image_label.pack(pady=10)

# Button to upload an image
upload_button = Button(root, text="Upload Images", command=upload_image)
upload_button.pack(pady=10)

# Button to open the camera for detection
camera_button = Button(root, text="Open Camera for Detection", command=lambda: threading.Thread(target=open_camera).start())
camera_button.pack(pady=10)

root.mainloop()
