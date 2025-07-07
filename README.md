# Real-Time Face Landmark Detection in the Browser

This project enables real-time face landmark detection directly in your web browser using the device's camera. It's built withpython, and Google's [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html), allowing users to view facial feature tracking live — no installations or downloads required.

## 🌟 Features

- 🔍 468-point facial landmark detection
- 💻 Runs entirely in the browser (client-side only)
- 📱 Mobile and desktop compatible
- 🚀 Fast and privacy-friendly (no data leaves your device)
- 🎯 Ideal for AR/VR, face filters, and styling assistants


> ⚠️ Please allow camera access when prompted.

## 🛠️ How It Works

1. JavaScript uses `getUserMedia()` to access the device camera.
2. The video feed is rendered to a `<canvas>` element.
3. MediaPipe Face Mesh processes each frame in real-time.
4. Facial landmarks are drawn directly over the video.

## 🚀 Getting Started
- Clone the project.
- Intall the packages and run main.py.
- The web browser will run and you will immediately see the face landmark detection working.
