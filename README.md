---

# 🧠 H.A.AI.E — Help Against Anxiety (Experimental) / WAICY project

### Local Emotional Companion — Private • Lightweight • Humanized

---

## 🌱 Overview

**H.A.AI.E** is an experimental AI companion designed to help people overcome social anxiety and loneliness — locally, without internet access, and without monetizing personal data.

It’s *not* meant to replace human interaction, but to help users **relearn communication and emotional expression** in a safe, private, and non-commercial way.

---

## 🎯 Goals

* 🗣️ **Conversational companion**: Interact naturally with emotion-based feedback.
* 🔒 **Privacy first**: Everything runs **locally** (no cloud, no data collection).
* 🧍‍♀️ **Humanized interface**: Live2D-based Vtuber for visual expression.
* 🧩 **Lightweight and accessible**: Optimized for "low-resource" systems. (16gb vram recommanded)
* 🧘‍♂️ **Emotional support**: Non-judgmental presence to help practice communication/chat bot with a personnality

---

## 🧩 Model Dependencies

To run **H.A.AI.E**, you’ll need to download or clone the following models:

### 🗣️ Text-to-Speech

```
OuteTTS-0.2-500M
```

### 💬 Emotion Detection

```
ModernBERT-large-go-emotions
multilingual_go_emotions_V1.2
```

### 😏 Sarcasm & Irony Detection

```
sarcasm-detection-RoBERTa-base-CR
twitter-roberta-base-irony
```

### 🧍‍♀️ Speech Model (French Example)

```
fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx
```

### 🧍‍♀️ LLM persona

```
Gemma4 26B A4B
```

### Detector of emotionnal dependancy

```
Phi-4-mini-QAT_4b 
```

---

## 🧰 Installation & Run / main script

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/HAAIE.git
cd HAAIE
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the test script

```bash
python main.py
```

### 5. Download models 

<!-- [link of the drive](https://drive.google.com/drive/folders/1xmPeIi9dzqERD8u6cl4wBQzBhvR0C0Oh?usp=sharing) --->

## 🧰 Installation & Run / emmotionnal dashboard


### Prerequisites

* Node.js (version 16 or higher recommended)
* npm (included with Node.js)

1. **Install the dependencies**

   ```bash
   npm install
   ```

### Launch the Dashboard

1. **Start the development server**

   ```bash
   npm run dev
   ```

2. **Access the Dashboard**
   Open your browser at the indicated address.


---

## 💡 Philosophy

> “Your loneliness is not a product.”

H.A.AI.E is open-source and built for **mental health awareness**, **privacy**, and **social reconnection**, not profit.

---

## 🧑‍💻 Author

Independent developer — France
Contact: perso[aroba]archibarbu[dot]com

---

#### Please check license before use it in commercial project.

#### Attention : 


Ce projet n'a pas été validé par une organisation médicale.
Il n'a aucune prétention thérapeutique et ne pose aucun 
diagnostic ni traitement. 

Données stockées localement uniquement. (Les emotions peuvent etre stocker, pas les conversations)


#### Disclaimer :

This project has not been validated by any medical organization.
It makes no therapeutic claims and does not provide diagnosis 
or treatment.

Data stored locally only. (Onlu emotions can be stored, no any conversations)

IA can be not accurate, use this caution. 
18 under-age, please don't use.