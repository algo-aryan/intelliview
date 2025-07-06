
# 🧠 IntelliView – AI-Powered Interview Prep & Resume Analyzer

**IntelliView** is a backend-heavy, AI-powered job preparation platform that enables users to simulate mock interviews, get real-time feedback, and evaluate resumes through ATS scoring. Built with a robust Python backend using Flask and integrated with Gemini AI (Google Generative AI), IntelliView empowers users to improve their interview performance and optimize resumes for applicant tracking systems.

---

## 🚀 Live Deployment

Hosted on **AWS EC2** with persistent server processes managed via `tmux`.  
🔗 `https://3d50-16-16-183-86.ngrok-free.app`

---

## 🎯 Core Features

- 🎤 **Mock Interview Simulator**  
  Users can take AI-driven interviews with real-time guidance and feedback.

- 📄 **ATS Resume Analyzer**  
  Upload a resume in PDF format to receive an ATS compatibility score with actionable suggestions.

- 📊 **Performance History**  
  Track interview performance history with detailed breakdowns.

- 🌐 **Responsive UI**  
  Fully responsive, minimalistic frontend that works across all devices.

---

## 🧠 AI & NLP Capabilities

- **Gemini AI (Google Generative AI)** is used to:
  - Generate mock interview questions
  - Evaluate user answers
  - Suggest improvements

- **PDF Parsing & Resume Scoring**:
  - Extracts skills and content using `PyMuPDF`
  - Checks format, keyword match, and ATS compliance

---

## 📁 Project Structure

```
IntelliView/
├── web_app/
│   ├── static/
│   │   ├── assets/                # Images and icons
│   │   └── css/                   # Custom stylesheets
│   ├── templates/                # Rendered HTML templates via Flask
│   │   ├── ats_score.html
│   │   ├── history.html
│   │   ├── history_list.html
│   │   ├── index.html
│   │   ├── interview.html
│   │   ├── profile.html
│   │   ├── settings.html
│   │   └── take-interview.html
│   ├── .env                      # Environment configuration
│   ├── .dockerignore
│   ├── .gitignore
│   ├── main.py                   # Flask backend server (entry point)
│   ├── requirements.txt          # Python dependencies
│   ├── run.sh                    # Shell script to start the app
│   ├── setup.sh                  # Setup script for deployment
│   ├── runtime.txt
```

---

## ⚙️ Technologies Used

### 🔧 Backend
- **Python 3**
- **Flask** – RESTful web framework
- **Gemini AI (Google Generative AI)** – for generating interview feedback and questions
- **PyMuPDF** – to extract text and structure from uploaded PDF resumes
- **dotenv** – to manage secret keys and environment variables
- **Werkzeug** – for request handling and file upload security

### 🌐 Frontend
- **HTML5**, **CSS3**, **Vanilla JavaScript**
- Dynamic rendering via **Flask templating** with `.html` files inside `/templates`
- Clean, responsive layout with mobile support

### 🖥️ Deployment & DevOps
- **AWS EC2** – app hosting
- **tmux** – background session management for running persistent backend
- **Shell scripting (`run.sh`, `setup.sh`)** – automation for setup and server start
- **Git + GitHub** – version control

---

## 🛠️ Local Development Setup

### ✅ Prerequisites
- Python 3.8+
- Git
- pip
- Virtualenv (recommended)

### 📦 Installation

```bash
# Clone the repo
git clone https://github.com/algo-aryan/intelliview.git
cd intelliview/project/web_app

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
./setup.sh

# Set up environment variables
cp .env.example .env   # Then add Gemini API key and other secrets

# Run the Flask app
python3 web_app/main.py
```

---

## 🖥️ Deployment Guide (AWS EC2 + tmux)

```bash
# SSH into EC2
ssh ubuntu@<your-ec2-ip>

# Navigate to project directory
cd /home/ubuntu/intelliview/project/web_app

# Pull latest changes
git checkout fresh-start
git pull origin fresh-start

# Activate virtualenv
source .venv/bin/activate

# Install/update dependencies
./setup.sh

# Start app inside tmux
tmux new -s intelliview
python3 web_app/main.py

# Detach safely (CTRL + B, then D)
```

---

## 🧪 Sample Environment Variables (`.env`)

```dotenv
GOOGLE_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_flask_secret
UPLOAD_FOLDER=uploads/
```

---

## 🧼 Shell Scripts

- `setup.sh`: Initial setup, installs Python packages, sets up venv, and prepares environment.
- `run.sh`: Executes `main.py` inside activated venv, used for starting the server post-deployment.

---

## 📬 Contact

For queries or contributions, reach out via:

📧 aryan1509bansal@gmail.com 
🔗 [LinkedIn](https://linkedin.com/in/aryanbansal1509)

---

## 📃 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with credit.
