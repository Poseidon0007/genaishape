# Genaishape 🧠📐
A GenAI-powered web app that detects, completes, and analyzes geometric shapes in uploaded images using **YOLOv5**, **OpenCV**, and **custom heuristics** — all within a **Streamlit** interface.

## 🌐 Live App
👉 [Try it Now on Streamlit Cloud](https://genaishape-czhv2uxrsrj3ztubs2y5mp.streamlit.app/)

---

## 💡 Features
- 📷 Upload image and detect shapes using:
  - 🔍 YOLOv5
  - 🧠 Custom geometric logic
  - 🌀 Hough Circle Transform
  - 🕸️ Canny Edge Detection
- 📊 Upload CSV file to visualize shape analysis
- 🔁 Automatic shape classification and symmetry line drawing
- ⚙️ Fast and responsive UI with Streamlit
- ☁️ Deployed on Streamlit Cloud

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV, YOLOv5 (PyTorch)
- **ML Backend**: PyTorch, torchvision
- **Deployment**: Streamlit Cloud
- **Data Analysis**: Pandas

---

## 📂 Example Use
1. Upload an image with partial or complete shapes.
2. Select a detection mode (e.g., YOLOv5 or Custom Detection).
3. View results with visual overlays and analysis.
4. Optionally, upload a `.csv` file with shape metadata to get plots.

---

## 📸 Screenshots
![YOLO Detection](screenshots/yolo.png)
![Custom Shape Detection](screenshots/custom.png)
![CSV Plot](screenshots/csv.png)

---

