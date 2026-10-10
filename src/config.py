"""
config.py
---------
Central configuration for the entire Plant Disease Detection project.
All scripts import from here — change a value once, affects everything.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (all relative to the PROJECT ROOT, one level above src/)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")          # downloaded zip lands here
TRAIN_DIR     = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR       = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DIR      = os.path.join(PROJECT_ROOT, "data", "test")

OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs")
CKPT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "plots")
LOG_CSV       = os.path.join(PROJECT_ROOT, "outputs", "training_log.csv")
HISTORY_JSON  = os.path.join(PROJECT_ROOT, "outputs", "training_history.json")
BEST_CKPT     = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_model.pth")
LAST_CKPT     = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "last_model.pth")
REPORT_TXT    = os.path.join(PROJECT_ROOT, "outputs", "classification_report.txt")
CM_NPY        = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrix.npy")
PRED_CSV      = os.path.join(PROJECT_ROOT, "outputs", "predictions.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle dataset
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_DATASET  = "emmarex/plantdisease"       # kaggle datasets download -d <this>
KAGGLE_ZIP_NAME = "plantdisease.zip"           # expected zip filename after download
# The folder inside the zip that contains per-class sub-folders:
KAGGLE_INNER_FOLDER = "PlantVillage"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset split ratios   (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42                               # reproducible splits

# ─────────────────────────────────────────────────────────────────────────────
# Image / model settings
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224                             # pixels (H = W)
NUM_CHANNELS = 3                               # RGB

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
NUM_EPOCHS    = 15          # 15 epochs → ~85-90% accuracy, trains in 1-2 hrs on CPU
BATCH_SIZE    = 64          # larger batch = fewer iterations per epoch
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.4
NUM_WORKERS   = 0           # MUST be 0 on Windows — multiprocessing causes hangs
PATIENCE      = 5           # stop early if no improvement for 5 epochs
LABEL_SMOOTH  = 0.1
GRAD_CLIP     = 5.0

# Cosine-annealing warm restarts
LR_T0    = 5                # restart every 5 epochs (suits 15-epoch run)
LR_TMULT = 1
LR_MIN   = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# Disease remedy / info database
# Each key must match a PlantVillage class folder name EXACTLY.
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    # ── Apple ────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "display": "Apple — Apple Scab",
        "severity": "Moderate",
        "cause": "Fungus: Venturia inaequalis",
        "symptoms": "Olive-green to brown scab-like lesions on leaves and fruit.",
        "remedy": (
            "• Apply fungicides (captan, myclobutanil) at bud break and repeat every 7–10 days.\n"
            "• Remove and destroy fallen infected leaves in autumn.\n"
            "• Choose scab-resistant apple varieties for future planting."
        ),
    },
    "Apple___Black_rot": {
        "display": "Apple — Black Rot",
        "severity": "Severe",
        "cause": "Fungus: Botryosphaeria obtusa",
        "symptoms": "Circular brown lesions with purple borders on leaves; mummified fruit.",
        "remedy": (
            "• Prune and destroy infected wood and mummified fruit.\n"
            "• Apply captan or thiophanate-methyl fungicides.\n"
            "• Maintain orchard sanitation to limit spread."
        ),
    },
    "Apple___Cedar_apple_rust": {
        "display": "Apple — Cedar Apple Rust",
        "severity": "Moderate",
        "cause": "Fungus: Gymnosporangium juniperi-virginianae",
        "symptoms": "Bright orange-yellow spots on upper leaf surface in spring.",
        "remedy": (
            "• Remove nearby cedar/juniper trees (alternate host) if feasible.\n"
            "• Apply myclobutanil or mancozeb from pink stage through petal fall.\n"
            "• Plant rust-resistant apple cultivars."
        ),
    },
    "Apple___healthy": {
        "display": "Apple — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy with no visible disease symptoms.",
        "remedy": (
            "• Continue regular monitoring and preventive fungicide schedule.\n"
            "• Ensure adequate irrigation, nutrition, and pruning.\n"
            "• Keep orchard floor clean of debris."
        ),
    },
    # ── Blueberry ────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "display": "Blueberry — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Maintain proper pH (4.5–5.5), irrigation, and pest monitoring.",
    },
    # ── Cherry ───────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "display": "Cherry — Powdery Mildew",
        "severity": "Moderate",
        "cause": "Fungus: Podosphaera clandestina",
        "symptoms": "White powdery coating on young leaves, shoots, and fruit.",
        "remedy": (
            "• Apply sulfur-based or trifloxystrobin fungicides at first sign.\n"
            "• Improve air circulation by pruning dense canopy.\n"
            "• Avoid excess nitrogen fertilization."
        ),
    },
    "Cherry_(including_sour)___healthy": {
        "display": "Cherry — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Continue routine care and preventive spraying.",
    },
    # ── Corn / Maize ─────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": "Corn — Gray Leaf Spot",
        "severity": "Severe",
        "cause": "Fungus: Cercospora zeae-maydis",
        "symptoms": "Rectangular gray-tan lesions running parallel to leaf veins.",
        "remedy": (
            "• Use resistant hybrid varieties.\n"
            "• Apply strobilurin or triazole fungicides at VT–R1 stage.\n"
            "• Practice crop rotation and residue management."
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn — Common Rust",
        "severity": "Moderate",
        "cause": "Fungus: Puccinia sorghi",
        "symptoms": "Cinnamon-brown pustules (uredinia) on both leaf surfaces.",
        "remedy": (
            "• Plant rust-resistant hybrids.\n"
            "• Apply propiconazole or azoxystrobin if infection is heavy before tasseling.\n"
            "• Early-season infections rarely need treatment in resistant varieties."
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn — Northern Leaf Blight",
        "severity": "Severe",
        "cause": "Fungus: Exserohilum turcicum",
        "symptoms": "Large, cigar-shaped, gray-green lesions 2.5–15 cm long.",
        "remedy": (
            "• Grow resistant hybrids (Ht gene).\n"
            "• Apply fungicides (azoxystrobin, propiconazole) at first sign above ear leaf.\n"
            "• Rotate crops and till residue."
        ),
    },
    "Corn_(maize)___healthy": {
        "display": "Corn — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Monitor regularly; maintain balanced fertilization.",
    },
    # ── Grape ────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "display": "Grape — Black Rot",
        "severity": "Severe",
        "cause": "Fungus: Guignardia bidwellii",
        "symptoms": "Tan circular lesions with dark borders on leaves; black shriveled berries.",
        "remedy": (
            "• Apply mancozeb or myclobutanil starting at budbreak.\n"
            "• Remove mummified berries and infected canes.\n"
            "• Ensure good canopy air circulation."
        ),
    },
    "Grape___Esca_(Black_Measles)": {
        "display": "Grape — Esca (Black Measles)",
        "severity": "Severe",
        "cause": "Fungal complex: Phaeomoniella chlamydospora and others",
        "symptoms": "Tiger-stripe leaf discoloration, dark streaking in wood, sudden vine collapse.",
        "remedy": (
            "• No effective chemical cure exists — prevent trunk wounds.\n"
            "• Remove and destroy severely infected vines.\n"
            "• Protect pruning wounds with wound sealants."
        ),
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": "Grape — Leaf Blight",
        "severity": "Moderate",
        "cause": "Fungus: Pseudocercospora vitis",
        "symptoms": "Angular dark brown spots on upper leaf surface, premature defoliation.",
        "remedy": (
            "• Apply copper-based fungicides or mancozeb.\n"
            "• Remove and destroy fallen infected leaves.\n"
            "• Improve vine spacing for better airflow."
        ),
    },
    "Grape___healthy": {
        "display": "Grape — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Continue standard vineyard management practices.",
    },
    # ── Orange ───────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": "Orange — Citrus Greening (HLB)",
        "severity": "Critical",
        "cause": "Bacterium: Candidatus Liberibacter asiaticus (spread by psyllid)",
        "symptoms": "Asymmetric yellowing (blotchy mottle) of leaves; lopsided, bitter fruit.",
        "remedy": (
            "• No cure — remove and destroy infected trees immediately.\n"
            "• Control Asian citrus psyllid vector with insecticides (imidacloprid).\n"
            "• Plant certified disease-free nursery stock."
        ),
    },
    # ── Peach ────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "display": "Peach — Bacterial Spot",
        "severity": "Moderate",
        "cause": "Bacterium: Xanthomonas arboricola pv. pruni",
        "symptoms": "Small water-soaked lesions turning brown/purple on leaves; fruit spots.",
        "remedy": (
            "• Apply copper-based bactericides from shuck split through harvest.\n"
            "• Use resistant cultivars.\n"
            "• Avoid overhead irrigation."
        ),
    },
    "Peach___healthy": {
        "display": "Peach — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Maintain regular pruning and copper spray program.",
    },
    # ── Pepper ───────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "severity": "Moderate",
        "cause": "Bacterium: Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small, water-soaked, circular leaf spots with yellow halos; fruit lesions.",
        "remedy": (
            "• Use disease-free, certified seed.\n"
            "• Apply copper + mancozeb sprays on a 5–7 day schedule.\n"
            "• Avoid overhead watering; practice crop rotation."
        ),
    },
    "Pepper,_bell___healthy": {
        "display": "Bell Pepper — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Ensure adequate nutrition and pest monitoring.",
    },
    # ── Potato ───────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "severity": "Moderate",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Dark brown concentric ring lesions (target-board pattern) on older leaves.",
        "remedy": (
            "• Apply chlorothalonil, mancozeb, or azoxystrobin every 7–14 days.\n"
            "• Remove and destroy infected plant debris.\n"
            "• Practice 2–3 year crop rotation."
        ),
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight",
        "severity": "Critical",
        "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Water-soaked greenish lesions turning brown/black; white sporulation on leaf underside.",
        "remedy": (
            "• Apply preventive fungicides (mancozeb, cymoxanil) before symptoms appear.\n"
            "• Destroy all infected plant material immediately.\n"
            "• Use certified blight-resistant potato varieties."
        ),
    },
    "Potato___healthy": {
        "display": "Potato — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Continue regular scouting and preventive fungicide schedule.",
    },
    # ── Raspberry ────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "display": "Raspberry — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Monitor for aphids and fungal issues; prune old canes annually.",
    },
    # ── Soybean ──────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "display": "Soybean — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Maintain proper row spacing and scout for soybean rust.",
    },
    # ── Squash ───────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "display": "Squash — Powdery Mildew",
        "severity": "Moderate",
        "cause": "Fungi: Podosphaera xanthii / Erysiphe cichoracearum",
        "symptoms": "White powdery patches on upper leaf surface; leaves curl and die.",
        "remedy": (
            "• Apply potassium bicarbonate, neem oil, or sulfur sprays.\n"
            "• Improve air circulation.\n"
            "• Remove heavily infected leaves."
        ),
    },
    # ── Strawberry ───────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "display": "Strawberry — Leaf Scorch",
        "severity": "Moderate",
        "cause": "Fungus: Diplocarpon earlianum",
        "symptoms": "Irregular dark purple spots; affected tissue turns brown and 'scorched'.",
        "remedy": (
            "• Apply captan or myclobutanil fungicides.\n"
            "• Renovate beds after harvest; remove old foliage.\n"
            "• Avoid wetting foliage; use drip irrigation."
        ),
    },
    "Strawberry___healthy": {
        "display": "Strawberry — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.",
        "remedy": "• Keep beds clean and well-ventilated; monitor for spider mites.",
    },
    # ── Tomato ───────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "display": "Tomato — Bacterial Spot",
        "severity": "Moderate",
        "cause": "Bacterium: Xanthomonas spp.",
        "symptoms": "Small, water-soaked, scab-like spots on leaves and fruit.",
        "remedy": (
            "• Use disease-free seed; apply copper bactericides weekly.\n"
            "• Avoid overhead irrigation.\n"
            "• Practice 2-year crop rotation."
        ),
    },
    "Tomato___Early_blight": {
        "display": "Tomato — Early Blight",
        "severity": "Moderate",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Concentric ring (target) lesions on older leaves, stem collar rot.",
        "remedy": (
            "• Apply chlorothalonil or azoxystrobin on a 7-day schedule.\n"
            "• Stake plants for better airflow.\n"
            "• Remove lower infected leaves."
        ),
    },
    "Tomato___Late_blight": {
        "display": "Tomato — Late Blight",
        "severity": "Critical",
        "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Greasy gray-green lesions on leaves; white mold on undersides.",
        "remedy": (
            "• Apply mancozeb or chlorothalonil preventively.\n"
            "• Destroy infected plants immediately to prevent spread.\n"
            "• Avoid wetting foliage."
        ),
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "severity": "Moderate",
        "cause": "Fungus: Passalora fulva",
        "symptoms": "Pale green/yellow spots on upper leaf surface; olive-brown mold below.",
        "remedy": (
            "• Reduce humidity — improve ventilation in greenhouses.\n"
            "• Apply copper, chlorothalonil, or azoxystrobin.\n"
            "• Use mold-resistant varieties."
        ),
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot",
        "severity": "Moderate",
        "cause": "Fungus: Septoria lycopersici",
        "symptoms": "Small circular spots with dark borders and light centers; heavy defoliation.",
        "remedy": (
            "• Apply chlorothalonil, mancozeb, or copper fungicides.\n"
            "• Remove and dispose of infected lower leaves.\n"
            "• Avoid working with plants when wet."
        ),
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": "Tomato — Spider Mites",
        "severity": "Moderate",
        "cause": "Pest: Tetranychus urticae",
        "symptoms": "Stippled, bronzed leaves; fine webbing on leaf undersides.",
        "remedy": (
            "• Apply miticides (abamectin, bifenazate) or insecticidal soap.\n"
            "• Increase humidity — mites thrive in dry conditions.\n"
            "• Introduce predatory mites (Phytoseiidae) for biological control."
        ),
    },
    "Tomato___Target_Spot": {
        "display": "Tomato — Target Spot",
        "severity": "Moderate",
        "cause": "Fungus: Corynespora cassiicola",
        "symptoms": "Brown spots with concentric rings and yellow halos on leaves.",
        "remedy": (
            "• Apply azoxystrobin or chlorothalonil fungicides.\n"
            "• Remove infected leaves and debris.\n"
            "• Improve air circulation around plants."
        ),
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus",
        "severity": "Critical",
        "cause": "Virus: TYLCV (spread by whitefly Bemisia tabaci)",
        "symptoms": "Upward leaf curling, yellowing, stunted growth, flower drop.",
        "remedy": (
            "• No cure — remove and destroy infected plants.\n"
            "• Control whitefly vector with yellow sticky traps and imidacloprid.\n"
            "• Use TYLCV-resistant tomato varieties."
        ),
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato — Mosaic Virus",
        "severity": "Severe",
        "cause": "Virus: Tomato mosaic virus (ToMV)",
        "symptoms": "Mosaic light/dark green leaf pattern; leaf deformation; stunting.",
        "remedy": (
            "• No cure — remove infected plants.\n"
            "• Disinfect tools with 10% bleach solution between plants.\n"
            "• Plant virus-resistant (Tm-2 gene) varieties."
        ),
    },
    "Tomato___healthy": {
        "display": "Tomato — Healthy",
        "severity": "None",
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy with no visible disease symptoms.",
        "remedy": "• Continue regular scouting, pruning, and preventive spray schedule.",
    },
}
