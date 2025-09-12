## Raga Detection (Carnatic + Hindustani)

Production-ready, config-driven, end-to-end raga detection using PyTorch.

### Features
- Multi-view features: Log-Mel, CQT, Chroma, Tonnetz, F0 (pyin/CREPE)
- Augmentations: noise, pitch-shift, time-stretch, bandpass, SpecAugment, tonic shift, mixup
- Model: CRNN with attention pooling (optional transformer), class-imbalance handling
- AMP training, robust sliding-window inference with probability aggregation

### Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Data layout
```
data/
  audio/
    RagaName1/
      file1.wav
      file2.wav
    RagaName2/
      file3.wav
```

### Build manifest
```bash
python src/dataset.py --build-manifest --data-root data/audio --output data/manifest.csv --val-size 0.15 --test-size 0.15 --seed 42
```

### Train
```bash
python src/train.py --config config.yaml
```

### Inference
```bash
python src/infer.py --audio path/to/file.wav --checkpoint runs/best.ckpt --config config.yaml
```

### Docker
```bash
docker build -t raga-detect .
docker run --gpus all -v $PWD:/workspace raga-detect python src/train.py --config config.yaml
```

### Notes
- F0 backend defaults to `pyin`. To use CREPE, ensure GPU/CPU performance is acceptable.
- Feature normalization stats are computed on the training split and cached.

