# 🎶 Raga Detection using TensorFlow (Enhanced Version)

This repository extends the original **E2ERaga** framework with advanced modules and expert-knowledge refinements for **robust raga and tonic identification** from audio samples.

The base pipeline remains the same (*CREPE-based pitch extraction + KNN models*), but this enhanced version integrates:

- ✅ **Feature-rich classification** (spectral, temporal, harmonic, pitch-based)  
- ✅ **Ensemble ML models** (XGBoost, Random Forest, Gradient Boosting, SVM, MLP)  
- ✅ **Musicological rules** to resolve common raga confusions  
- ✅ **Improved accuracy & stability** compared to the original  

---

## 🚀 What’s New

- 🎵 Advanced **feature extraction** (spectral, temporal, harmonic, pitch-based)  
- 🤖 **Ensemble ML models** for stronger classification  
- 📚 **Knowledge-driven refinements** to reduce misclassifications  
- 🎼 Improved accuracy in **distinguishing similar ragas**  
- 🔒 Overall: higher **stability & reliability** in predictions  

---

## 📦 Installation

**Requirements:**
- `python==3.7.16`  
- [Anaconda](https://www.anaconda.com/download) *(recommended for package management)*  

**Install dependencies:**
```bash
pip install -r requirements.txt


## ⚙️ Model Setup

1. Create an empty folder called **`model/`** inside the `SPD_KNN` folder.  
2. Download and place the following files:  
   - **Pitch model:** [Download here](https://drive.google.com/file/d/1On0sbDARW6uVvfVQ6IJkhWtUaaH1fBw8/view?usp=sharing) → put in `model/`  
   - **Tonic models (Hindustani & Carnatic):** [Download here](https://drive.google.com/drive/folders/1h7dois2zZMLBcx7gl-_0phlILzOUvL8q) → put in `model/`  
3. Download the **Raga KNN models**:  
   - Carnatic → [Download here](https://drive.google.com/drive/folders/1OXGknLkShVFQSCZkcIfdIl5eYeCN9T9E) → place in `data/RagaDataset/Carnatic/model/` *(create folders if needed)*  
   - Hindustani → [Download here](https://drive.google.com/drive/folders/14OMUyhbA2sw2rD6y1-cMINreo-S-GaiE) → place in `data/RagaDataset/Hindustani/model/` *(create folders if needed)*  

---

## 🎼 Data

The datasets cannot be uploaded here due to licensing restrictions.  
They can be requested directly from **CompMusic**:  
👉 [Request Dataset Here](https://compmusic.upf.edu/node/328)


▶️ Running the Model
🔹 Runtime Input (record live audio)
python main.py --runtime=True --tradition=h --duration=60


tradition: h (Hindustani) / c (Carnatic)

duration: recording length in seconds

🔹 File Input (pre-recorded audio)
python main.py --runtime_file=data/sample_data/Ahira_bhairav_27.wav --tradition=h


Supports .wav and .mp3 (mp3 is auto-converted to wav).

🔹 Simple Output Mode

For concise logs:

python main.py --runtime_file=<audio_file> --tradition=<h/c> --simple_output


Or via environment flags (PowerShell example):

$env:TF_CPP_MIN_LOG_LEVEL='3'; $env:RAGA_QUIET='1'; python main.py --runtime_file=<audio_file> --tradition=<h/c>

🎶 Supported Ragas

Carnatic: 40 ragas

Hindustani: 30 ragas

🔬 Advanced Enhancements

The advanced/ folder introduces richer models & refinements:

🎛 feature_extraction.py → Extracts spectral, temporal, harmonic & pitch features

🧠 ensemble_models.py → XGBoost, Random Forest, Gradient Boosting, SVM, MLP

🎼 expert_knowledge.py → Musicological rules & common-confusion resolvers

🙏 Acknowledgments

🎵 CREPE for pitch extraction

📚 CompMusic & Sankalp Gulati for datasets

👨‍💻 Vishwaas Hegde for the original E2ERaga & SPD_KNN
