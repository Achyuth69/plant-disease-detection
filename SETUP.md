# Teammate Setup Guide

**The dataset is NOT included in this project folder** (it's ~800 MB).  
Each teammate needs to download it once on their own machine.

---

## One-time setup (do this first)

### 1. Install dependencies
```cmd
pip install -r requirements.txt
```

### 2. Download the dataset manually

1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Sign in to Kaggle (free account) and click **Download**
3. Extract the downloaded `archive.zip`
4. You will get a folder called `PlantVillage` with 38 sub-folders

### 3. Place the data here
```
plant_disease_detection\
    data\
        raw\
            PlantVillage\        ← paste the extracted folder here
                Apple___Apple_scab\
                Apple___Black_rot\
                ... (38 folders)
```

### 4. Split the data (80% train / 10% val / 10% test)
```cmd
python src/kaggle_download.py --skip_download
```

---

## Run the project

```cmd
# Train the model
python src/train.py

# Generate result plots
python src/visualize.py

# Launch the web app  →  http://127.0.0.1:7860
python app.py
```

---

## What each teammate shares

| What | How |
|------|-----|
| Source code | GitHub / USB / Google Drive (this folder, ~50 KB) |
| Dataset | Each person downloads from Kaggle independently |
| Trained model | Share only `outputs/checkpoints/best_model.pth` (~50 MB) via Drive |

> The trained model `.pth` file can be shared on Google Drive so teammates  
> can run the app without training from scratch.
