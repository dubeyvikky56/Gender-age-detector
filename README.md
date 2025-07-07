# 🧠 Gender and Age Detection (Command Line Tool)

A Python-based command-line tool for detecting **gender** and **age** from input images using **DeepFace**, **OpenCV**, and **pre-trained Caffe models**. No web interface or Flask required. Just run the script and get accurate predictions directly in your terminal.

---

## 🔧 Technologies Used

- Python 3.x
- DeepFace
- TensorFlow (compatible with tf-keras)
- OpenCV
- Pre-trained Caffe Models (Age & Gender)
- RetinaFace (for face detection)

---

## 📁 Project Structure

```
gad/
├── detect.py
├── age_net.caffemodel
├── gender_net.caffemodel
├── age_deploy.prototxt
├── gender_deploy.prototxt
├── opencv_face_detector_uint8.pb
├── opencv_face_detector.pbtxt
├── your_image.jpg
```

---

## ▶️ How to Run

1. Clone this repository or download the files.
2. Place the image you want to analyze in the folder.
3. Open CMD and navigate to the project directory:

```bash
cd path\to\gad
python detect.py
```

You’ll see predicted age and gender printed in the terminal.

---

## 📌 Example Output

```
Predicted Gender: Male
Predicted Age: 24
```

---

## 💡 Features

- Predicts **age** (0–100) and **gender** (Male/Female)
- No training required — uses pre-trained models
- Command-line based, lightweight, fast
- Perfect for beginners in AI/ML and image analysis

---

## 🧑‍💻 Author

**Niket Kumar**  
[LinkedIn](https://www.linkedin.com/in/niket-kumar-7353652b3)

---

## 📜 License

This project is for educational and demonstration purposes only.
