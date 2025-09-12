# Raga-detection using Tf (Enhanced Version)  

This repository builds upon the original [E2ERaga](https://github.com/VishwaasHegde/E2ERaga) by adding **advanced modules and expert-knowledge refinements** for more robust raga and tonic identification from audio samples.  

The base pipeline remains the same (CREPE-based pitch extraction + KNN models), but the enhanced version integrates **feature-rich classification, ensemble models, and musicological rules** to improve accuracy and reduce common misclassifications.  
 

---

## 🚀 What’s New in This Version
- Added **advanced feature extraction** (spectral, temporal, harmonic, pitch-based).  
- Integrated **ensemble ML models** (XGBoost, Random Forest, Gradient Boosting, SVM, MLP).  
- Introduced **expert-knowledge rules** to resolve common raga confusions.    
- Improved accuracy in distinguishing similar ragas   
- Overall: **Higher stability and more reliable raga prediction** compared to the original.  

---

## Getting Started  
Requires `python==3.7.16`  

Download and install [Anaconda](https://www.anaconda.com/products/individual) for easier package management.  

Install the requirements:  
```bash
pip install -r requirements.txt
Model Setup
Create an empty folder called model inside E2ERaga.

Download the pitch model (model-full.h5) from here and place it in the model folder.

Download the tonic models (Hindustani and Carnatic) from here and place them in the model folder.

Download the Carnatic raga KNN models from here and place them in data\RagaDataset\Carnatic\model (create folders if needed).

Download the Hindustani raga KNN models from here and place them in data\RagaDataset\Hindustani\model.

Data
Datasets cannot be uploaded here due to licensing restrictions.
They can be requested from CompMusic.

Running the Model
Runtime Input (recording audio)
bash
Copy code
python main.py --runtime=True --tradition=h --duration=60
tradition = h (Hindustani) / c (Carnatic)

duration = length of recording in seconds

File Input (pre-recorded audio)
bash
Copy code
python main.py --runtime_file=<audio_file_path> --tradition=<h/c>
Example:

bash
Copy code
python main.py --runtime_file=data/sample_data/Ahira_bhairav_27.wav --tradition=h
Supports .wav and .mp3 (mp3 will be internally converted to wav).

Simple Output Mode (concise logs)
bash
Copy code
python main.py --runtime_file=<audio_file_path> --tradition=<h/c> --simple_output
Or set environment flags (PowerShell example):

bash
Copy code
$env:TF_CPP_MIN_LOG_LEVEL='3'; $env:RAGA_QUIET='1'; python main.py --runtime_file=<audio_file_path> --tradition=<h/c>
Supported Ragas
Carnatic: 40 ragas (see data/carnatic_targets.txt)

Hindustani: 30 ragas (see data/hindustani_targets.txt)

🔹 Advanced Enhancements (New in This Version)
The advanced/ folder introduces richer models and knowledge-driven refinements:

feature_extraction.py → Extracts spectral, temporal, harmonic, and pitch-based features.

ensemble_models.py → Classifies ragas using XGBoost, Random Forest, Gradient Boosting, SVM, and MLP ensembles.

expert_knowledge.py → Adds musicological rules and common-confusion resolvers.


Acknowledgments
CREPE for pitch extraction.

CompMusic and Sankalp Gulati for datasets.

Vishwaas Hegde for the original E2ERaga and SPD_KNN repositories.