"""
app.py
------
Responsive Gradio web application for PlantGuard AI.
- Desktop  : full navigation bar + two-column layout
- Mobile   : compact top bar + single-column stacked layout

Run:  python app.py
Open: http://127.0.0.1:7860
"""

import io, os, sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model  import BNPlantDiseaseNet
from config import BEST_CKPT, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Disease info  (keys = exact class folder names from the trained model)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot", "plant": "Bell Pepper",
        "emoji": "🌶️", "severity": "Moderate",
        "cause": "Bacterium: Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small water-soaked spots with yellow halos on leaves; dark lesions on fruit.",
        "remedy": "• Use certified disease-free seed.\n• Apply copper + mancozeb sprays every 5–7 days.\n• Avoid overhead watering; practice crop rotation.",
    },
    "Pepper__bell___healthy": {
        "display": "Bell Pepper — Healthy", "plant": "Bell Pepper",
        "emoji": "🌶️", "severity": "None", "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.", "remedy": "• Maintain nutrition, irrigation and pest monitoring.",
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight", "plant": "Potato",
        "emoji": "🥔", "severity": "Moderate", "cause": "Fungus: Alternaria solani",
        "symptoms": "Dark brown concentric ring lesions on older lower leaves.",
        "remedy": "• Apply chlorothalonil, mancozeb, or azoxystrobin every 7–14 days.\n• Remove infected debris.\n• Practice 2–3 year crop rotation.",
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight", "plant": "Potato",
        "emoji": "🥔", "severity": "Critical", "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Water-soaked greenish lesions turning brown/black; white sporulation on underside.",
        "remedy": "• Apply preventive fungicides (mancozeb, cymoxanil) BEFORE symptoms.\n• Destroy all infected material immediately.\n• Use blight-resistant varieties.",
    },
    "Potato___healthy": {
        "display": "Potato — Healthy", "plant": "Potato",
        "emoji": "🥔", "severity": "None", "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.", "remedy": "• Continue regular scouting and preventive fungicide schedule.",
    },
    "Tomato_Bacterial_spot": {
        "display": "Tomato — Bacterial Spot", "plant": "Tomato",
        "emoji": "🍅", "severity": "Moderate", "cause": "Bacterium: Xanthomonas spp.",
        "symptoms": "Small water-soaked scab-like spots on leaves, stems and fruit.",
        "remedy": "• Use disease-free seed; apply copper bactericides weekly.\n• Avoid overhead irrigation.\n• Practice 2-year crop rotation.",
    },
    "Tomato_Early_blight": {
        "display": "Tomato — Early Blight", "plant": "Tomato",
        "emoji": "🍅", "severity": "Moderate", "cause": "Fungus: Alternaria solani",
        "symptoms": "Concentric ring (bullseye) lesions starting on older lower leaves.",
        "remedy": "• Apply chlorothalonil or azoxystrobin on a 7-day schedule.\n• Stake plants for better airflow.\n• Remove lower infected leaves.",
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight", "plant": "Tomato",
        "emoji": "🍅", "severity": "Critical", "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Greasy gray-green lesions on leaves; white mold on undersides.",
        "remedy": "• Apply mancozeb or chlorothalonil preventively.\n• Destroy infected plants immediately.\n• Avoid wetting foliage.",
    },
    "Tomato_Leaf_Mold": {
        "display": "Tomato — Leaf Mold", "plant": "Tomato",
        "emoji": "🍅", "severity": "Moderate", "cause": "Fungus: Passalora fulva",
        "symptoms": "Yellow spots on upper surface; olive-brown mold on underside.",
        "remedy": "• Reduce humidity — improve ventilation.\n• Apply copper, chlorothalonil, or azoxystrobin.\n• Use mold-resistant varieties.",
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot", "plant": "Tomato",
        "emoji": "🍅", "severity": "Moderate", "cause": "Fungus: Septoria lycopersici",
        "symptoms": "Small circular spots with dark borders and light gray centers.",
        "remedy": "• Apply chlorothalonil, mancozeb, or copper fungicides.\n• Remove infected lower leaves.\n• Avoid working with wet plants.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Tomato — Spider Mites", "plant": "Tomato",
        "emoji": "🍅", "severity": "Moderate", "cause": "Pest: Tetranychus urticae",
        "symptoms": "Stippled bronzed/silvery leaves; fine webbing on undersides.",
        "remedy": "• Apply miticides (abamectin, bifenazate) or insecticidal soap.\n• Increase humidity.\n• Introduce predatory mites for bio-control.",
    },
    "Tomato__Target_Spot": {
        "display": "Tomato — Target Spot", "plant": "Tomato",
        "emoji": "🍅", "severity": "Moderate", "cause": "Fungus: Corynespora cassiicola",
        "symptoms": "Brown spots with concentric rings and yellow halos.",
        "remedy": "• Apply azoxystrobin or chlorothalonil.\n• Remove infected leaves.\n• Improve air circulation.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus", "plant": "Tomato",
        "emoji": "🍅", "severity": "Critical", "cause": "Virus: TYLCV (spread by whitefly)",
        "symptoms": "Upward curling, yellowing leaves; stunted growth; flower/fruit drop.",
        "remedy": "• No cure — remove and destroy infected plants.\n• Control whitefly with traps and imidacloprid.\n• Plant TYLCV-resistant varieties.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Tomato — Mosaic Virus", "plant": "Tomato",
        "emoji": "🍅", "severity": "Severe", "cause": "Virus: ToMV (spread by contact/tools)",
        "symptoms": "Mosaic mottled leaf pattern; leaf deformation; plant stunting.",
        "remedy": "• No cure — remove infected plants.\n• Disinfect tools with 10% bleach.\n• Plant Tm-2 resistant varieties.",
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy", "plant": "Tomato",
        "emoji": "🍅", "severity": "None", "cause": "No disease detected",
        "symptoms": "Leaf appears healthy.", "remedy": "• Continue regular scouting and preventive spray schedule.",
    },
}

CONFIDENCE_THRESHOLD = 0.40

SEV_CFG = {
    "None":     {"color": "#16a34a", "bg": "#dcfce7", "icon": "✅", "label": "Healthy"},
    "Moderate": {"color": "#d97706", "bg": "#fef3c7", "icon": "⚠️",  "label": "Moderate Risk"},
    "Severe":   {"color": "#ea580c", "bg": "#ffedd5", "icon": "🔶", "label": "Severe"},
    "Critical": {"color": "#dc2626", "bg": "#fee2e2", "icon": "🚨", "label": "Critical"},
    "Unknown":  {"color": "#6b7280", "bg": "#f3f4f6", "icon": "❓", "label": "Unknown"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model, _class_names = None, None

def _load_model():
    global _model, _class_names
    if _model is not None:
        return _model, _class_names
    if not os.path.isfile(BEST_CKPT):
        raise FileNotFoundError(f"Model not found at: {BEST_CKPT}\nRun: python src/train.py")
    ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
    _class_names = ckpt["class_names"]
    _model = BNPlantDiseaseNet(num_classes=ckpt["num_classes"])
    _model.load_state_dict(ckpt["model_state"])
    _model.to(DEVICE).eval()
    print(f"[PlantGuard] Loaded | {ckpt['num_classes']} classes | "
          f"epoch={ckpt.get('epoch','?')} | val_acc={ckpt.get('val_acc',0)*100:.1f}%")
    return _model, _class_names

_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def _predict(img: Image.Image, topk: int = 5):
    m, cls = _load_model()
    if img.mode != "RGB":
        img = img.convert("RGB")
    t = _TF(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(m(t), dim=1).squeeze(0).cpu()
    k = min(topk, len(cls))
    top_p, top_i = torch.topk(probs, k=k)
    return [(cls[i.item()], top_p[j].item()) for j, i in enumerate(top_i)]

# ─────────────────────────────────────────────────────────────────────────────
# HTML result builders
# ─────────────────────────────────────────────────────────────────────────────

def _placeholder_html():
    return """
<div class="pg-placeholder">
  <div style="font-size:56px;margin-bottom:16px;">🌿</div>
  <p style="font-size:17px;font-weight:600;color:#475569;margin:0 0 6px;">Ready to Scan</p>
  <p style="font-size:13px;color:#94a3b8;margin:0;">Upload a leaf photo and click Analyze</p>
</div>"""

def _low_conf_html(conf: float):
    return f"""
<div class="pg-card" style="border-color:#fb923c;background:#fff7ed;text-align:center;padding:32px 24px;">
  <div style="font-size:52px;margin-bottom:12px;">🤔</div>
  <h2 style="margin:0 0 10px;color:#c2410c;font-size:20px;">Not a Recognizable Leaf</h2>
  <p style="color:#9a3412;font-size:14px;line-height:1.6;margin:0 0 16px;">
    Confidence is only <strong>{conf*100:.1f}%</strong>
    (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%).<br>
    The image may not be a supported plant leaf.
  </p>
  <div style="background:#fed7aa;border-radius:10px;padding:12px;font-size:13px;
              color:#7c2d12;text-align:left;">
    <strong>Supported crops:</strong> Bell Pepper &nbsp;·&nbsp; Potato &nbsp;·&nbsp; Tomato
  </div>
</div>"""

def _result_html(cls: str, conf: float):
    info  = DISEASE_INFO.get(cls, {
        "display": cls.replace("___"," — ").replace("_"," "),
        "plant": "Unknown", "emoji": "🌿", "severity": "Unknown",
        "cause": "No information available.", "symptoms": "No information available.",
        "remedy": "Consult a plant pathologist.",
    })
    sev     = info.get("severity", "Unknown")
    sc      = SEV_CFG.get(sev, SEV_CFG["Unknown"])
    remedy  = info.get("remedy", "").replace("\n", "<br>")
    healthy = sev == "None"
    bar_clr = "#16a34a" if conf >= 0.80 else "#d97706" if conf >= 0.60 else "#dc2626"

    disease_detail = "" if healthy else f"""
<div class="pg-grid-2" style="margin-bottom:12px;">
  <div class="pg-mini-card" style="background:#f0fdf4;border-color:#86efac;">
    <div class="pg-mini-label" style="color:#15803d;">🦠 Cause</div>
    <div class="pg-mini-body" style="color:#166534;">{info.get("cause","—")}</div>
  </div>
  <div class="pg-mini-card" style="background:#fefce8;border-color:#fde047;">
    <div class="pg-mini-label" style="color:#a16207;">🔍 Symptoms</div>
    <div class="pg-mini-body" style="color:#713f12;">{info.get("symptoms","—")}</div>
  </div>
</div>
<div class="pg-card" style="background:#eff6ff;border-color:#93c5fd;margin-bottom:12px;">
  <div class="pg-mini-label" style="color:#1d4ed8;margin-bottom:8px;">💊 Recommended Treatment</div>
  <div style="font-size:13px;color:#1e3a8a;line-height:1.8;">{remedy}</div>
</div>"""

    healthy_block = "" if not healthy else f"""
<div class="pg-card" style="background:#f0fdf4;border-color:#86efac;text-align:center;margin-bottom:12px;">
  <div style="font-size:36px;margin-bottom:8px;">🎉</div>
  <div style="font-size:16px;font-weight:700;color:#15803d;margin-bottom:6px;">Your plant looks healthy!</div>
  <div style="font-size:13px;color:#166534;line-height:1.7;">{remedy}</div>
</div>"""

    return f"""
<div class="pg-result-wrap">
  <!-- Hero card -->
  <div class="pg-hero-card" style="background:linear-gradient(135deg,{sc['bg']} 0%,#fff 100%);
       border-color:{sc['color']}30;box-shadow:0 4px 20px {sc['color']}18;">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
      <span style="font-size:44px;line-height:1;">{info.get('emoji','🌿')}</span>
      <div style="flex:1;min-width:0;">
        <div style="font-size:18px;font-weight:700;color:#1e293b;line-height:1.2;">
          {info.get('display', cls)}
        </div>
        <div style="display:inline-flex;align-items:center;gap:5px;
                    background:{sc['color']};color:white;border-radius:20px;
                    padding:4px 12px;margin-top:6px;font-size:12px;font-weight:700;">
          {sc['icon']} {sc['label']}
        </div>
      </div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;
                  font-size:12px;color:#64748b;margin-bottom:5px;">
        <span>Model Confidence</span>
        <span style="font-weight:700;color:{bar_clr};">{conf*100:.1f}%</span>
      </div>
      <div style="background:#e2e8f0;border-radius:999px;height:9px;overflow:hidden;">
        <div style="background:{bar_clr};width:{conf*100:.1f}%;height:100%;border-radius:999px;"></div>
      </div>
    </div>
  </div>

  {healthy_block}
  {disease_detail}

  <div class="pg-disclaimer">
    ⚕️ AI-assisted tool only. Always consult a certified agronomist before applying treatments.
  </div>
</div>"""

def _build_chart(results):
    labels = [DISEASE_INFO.get(r[0],{}).get("display",
              r[0].replace("___"," / ").replace("_"," ")) for r in reversed(results)]
    labels = [l[:36]+"…" if len(l)>38 else l for l in labels]
    values = [r[1]*100 for r in reversed(results)]
    max_v  = max(values)
    colors = ["#16a34a" if v==max_v else "#93c5fd" for v in values]
    fig, ax = plt.subplots(figsize=(7, max(3, len(results)*0.7)))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=10, color="#64748b")
    ax.set_title("Top Predictions", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.tick_params(axis="y", labelsize=9, colors="#334155")
    ax.tick_params(axis="x", labelsize=8, colors="#94a3b8")
    ax.axvline(x=CONFIDENCE_THRESHOLD*100, color="#f87171", linestyle="--",
               alpha=0.7, linewidth=1.2, label=f"{CONFIDENCE_THRESHOLD*100:.0f}% threshold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(val+1.2, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, color="#1e293b", fontweight="600")
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor="#f8fafc")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

def _build_table(results):
    rows = []
    for rank, (cls, prob) in enumerate(results, 1):
        d = DISEASE_INFO.get(cls,{}).get("display", cls.replace("___"," — ").replace("_"," "))
        s = DISEASE_INFO.get(cls,{}).get("severity","—")
        rows.append([str(rank), d, s, f"{prob*100:.1f}%"])
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────
def predict_disease(image):
    if image is None:
        return _placeholder_html(), None, []
    try:
        results = _predict(image, topk=5)
    except Exception as e:
        return f"<p style='color:red;padding:16px;'><b>Error:</b> {e}</p>", None, []
    top_cls, top_conf = results[0]
    chart = _build_chart(results)
    table = _build_table(results)
    if top_conf < CONFIDENCE_THRESHOLD:
        return _low_conf_html(top_conf), chart, table
    return _result_html(top_cls, top_conf), chart, table

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — responsive layout via media queries
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ═══ Base reset ═══════════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
body { background: #f1f5f9 !important; margin: 0; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important;
                    padding: 0 !important; background: transparent !important; }
footer { display: none !important; }

/* ═══ Shared component styles ══════════════════════════════════════════════ */
.pg-card {
  background: white; border: 1px solid #e2e8f0;
  border-radius: 16px; padding: 18px;
}
.pg-hero-card {
  border: 2px solid #e2e8f0;
  border-radius: 20px; padding: 20px; margin-bottom: 12px;
}
.pg-mini-card {
  border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px;
}
.pg-mini-label {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px; margin-bottom: 6px;
}
.pg-mini-body { font-size: 12px; line-height: 1.5; }
.pg-disclaimer {
  background: #f8fafc; border: 1px solid #e2e8f0;
  border-radius: 10px; padding: 10px 12px;
  font-size: 11px; color: #94a3b8; line-height: 1.5;
}
.pg-result-wrap { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
.pg-placeholder {
  font-family: -apple-system,sans-serif;
  text-align: center; padding: 64px 24px; color: #94a3b8;
}
.pg-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

/* ═══ DESKTOP layout (≥ 768 px) ════════════════════════════════════════════ */

/* Top navigation bar */
#pg-navbar {
  background: linear-gradient(90deg, #14532d 0%, #166534 40%, #15803d 100%);
  padding: 0 32px;
  display: flex; align-items: center; justify-content: space-between;
  height: 64px;
  box-shadow: 0 2px 12px #14532d40;
  position: sticky; top: 0; z-index: 100;
}
#pg-navbar .nav-brand {
  display: flex; align-items: center; gap: 10px;
}
#pg-navbar .nav-brand-icon { font-size: 26px; }
#pg-navbar .nav-brand-title {
  font-size: 20px; font-weight: 800; color: white; letter-spacing: -.5px;
}
#pg-navbar .nav-brand-sub {
  font-size: 11px; color: #86efac; margin-top: 1px;
}
#pg-navbar .nav-badges {
  display: flex; gap: 8px;
}
#pg-navbar .nav-badge {
  background: #ffffff18; color: #dcfce7;
  border: 1px solid #ffffff25; border-radius: 20px;
  padding: 4px 12px; font-size: 12px; font-weight: 600;
}

/* Desktop hero banner under navbar */
#pg-hero {
  background: linear-gradient(135deg,#166534 0%,#15803d 50%,#16a34a 100%);
  padding: 36px 40px 40px;
  display: flex; align-items: center; justify-content: space-between; gap: 32px;
}
#pg-hero .hero-text h1 {
  font-size: 32px; font-weight: 800; color: white;
  margin: 0 0 8px; letter-spacing: -1px;
}
#pg-hero .hero-text p {
  font-size: 15px; color: #bbf7d0; margin: 0 0 20px; line-height: 1.5;
}
#pg-hero .hero-stats {
  display: flex; gap: 20px;
}
#pg-hero .hero-stat {
  text-align: center;
}
#pg-hero .hero-stat-num {
  font-size: 28px; font-weight: 800; color: white;
}
#pg-hero .hero-stat-label {
  font-size: 11px; color: #86efac; font-weight: 600;
}
#pg-hero .hero-crops {
  display: flex; gap: 10px; flex-wrap: wrap;
}
#pg-hero .hero-crop-pill {
  background: #ffffff22; color: white;
  border: 1px solid #ffffff30; border-radius: 24px;
  padding: 8px 16px; font-size: 14px; font-weight: 600;
}

/* Desktop two-column content wrapper */
#pg-content {
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 24px;
  padding: 28px 32px;
  align-items: start;
}

/* Left panel (upload + button + tips) */
#pg-left-panel {
  display: flex; flex-direction: column; gap: 16px;
}
#pg-left-panel .panel-card {
  background: white; border-radius: 20px;
  padding: 20px; box-shadow: 0 2px 16px #0000000d;
}
#pg-left-panel .panel-section-title {
  font-size: 13px; font-weight: 700; color: #374151;
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 12px;
}
#pg-left-tips {
  background: white; border-radius: 16px;
  padding: 16px 18px; box-shadow: 0 1px 8px #0000000a;
}
#pg-left-tips .tips-title {
  font-size: 12px; font-weight: 700; color: #64748b;
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px;
}
#pg-left-tips .tips-list {
  display: flex; flex-direction: column; gap: 6px;
}
#pg-left-tips .tip-item {
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; color: #475569;
}

/* Right panel (results tabs) */
#pg-right-panel {
  background: white; border-radius: 20px;
  box-shadow: 0 2px 16px #0000000d; overflow: hidden;
}

/* Analyze button — desktop */
#pg-analyze-btn {
  width: 100%;
  border-radius: 14px !important;
  font-size: 16px !important; font-weight: 700 !important;
  padding: 14px !important;
  background: linear-gradient(135deg,#15803d,#16a34a) !important;
  border: none !important;
  box-shadow: 0 4px 16px #16a34a40 !important;
  transition: all .2s !important;
}
#pg-analyze-btn:hover {
  background: linear-gradient(135deg,#166534,#15803d) !important;
  box-shadow: 0 6px 20px #16a34a55 !important;
  transform: translateY(-1px) !important;
}

/* Desktop footer */
#pg-footer {
  background: #1e293b;
  padding: 24px 40px;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px;
}
#pg-footer .footer-brand {
  font-size: 16px; font-weight: 700; color: #f1f5f9;
}
#pg-footer .footer-sub {
  font-size: 12px; color: #64748b; margin-top: 2px;
}
#pg-footer .footer-badges {
  display: flex; gap: 8px; flex-wrap: wrap;
}
#pg-footer .footer-badge {
  background: #334155; color: #94a3b8;
  border-radius: 8px; padding: 4px 10px; font-size: 11px;
}

/* Tabs inside right panel */
.tab-nav { background: #f8fafc !important; border-bottom: 2px solid #e2e8f0 !important; }
.tab-nav button { font-size: 13px !important; font-weight: 600 !important;
                  padding: 12px 18px !important; color: #64748b !important; }
.tab-nav button.selected { color: #16a34a !important;
                            border-bottom: 3px solid #16a34a !important; }

/* Image upload zone */
.image-container { border-radius: 14px !important; border: 2px dashed #86efac !important; }
.image-container:hover { border-color: #16a34a !important; }

/* ═══ MOBILE layout (< 768 px) ═════════════════════════════════════════════ */
@media (max-width: 767px) {

  /* Hide desktop navbar & hero, show mobile bar */
  #pg-navbar  { display: none !important; }
  #pg-hero    { display: none !important; }
  #pg-footer  { display: none !important; }
  #pg-mobile-bar { display: block !important; }

  /* Single column, no side padding */
  #pg-content {
    grid-template-columns: 1fr !important;
    padding: 12px 10px !important;
    gap: 14px !important;
  }

  #pg-analyze-btn {
    font-size: 15px !important;
    padding: 13px !important;
  }

  /* Compact tab labels on mobile */
  .tab-nav button { font-size: 12px !important; padding: 10px 12px !important; }
}

/* Mobile top bar — hidden by default, shown on mobile via media query */
#pg-mobile-bar {
  display: none;
  background: linear-gradient(135deg, #166534 0%, #16a34a 100%);
  border-radius: 0 0 22px 22px;
  padding: 18px 16px 24px;
  box-shadow: 0 3px 16px #16653440;
  margin-bottom: -10px;
}
#pg-mobile-bar .mob-header {
  display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
}
#pg-mobile-bar .mob-title {
  font-size: 20px; font-weight: 800; color: white; letter-spacing: -.5px;
}
#pg-mobile-bar .mob-sub {
  font-size: 11px; color: #bbf7d0; margin-top: 1px;
}
#pg-mobile-bar .mob-pills {
  display: flex; gap: 6px; flex-wrap: wrap;
}
#pg-mobile-bar .mob-pill {
  background: #ffffff20; color: #dcfce7;
  border-radius: 20px; padding: 3px 10px;
  font-size: 11px; font-weight: 600;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="🌿 PlantGuard AI", css=CSS) as demo:

        # ── Desktop Navbar ────────────────────────────────────────────────
        gr.HTML("""
        <nav id="pg-navbar">
          <div class="nav-brand">
            <span class="nav-brand-icon">🌿</span>
            <div>
              <div class="nav-brand-title">PlantGuard AI</div>
              <div class="nav-brand-sub">Plant Disease Detection</div>
            </div>
          </div>
          <div class="nav-badges">
            <span class="nav-badge">🥔 Potato</span>
            <span class="nav-badge">🍅 Tomato</span>
            <span class="nav-badge">🌶️ Bell Pepper</span>
            <span class="nav-badge">15 Classes</span>
          </div>
        </nav>
        """)

        # ── Desktop Hero Banner ───────────────────────────────────────────
        gr.HTML("""
        <div id="pg-hero">
          <div class="hero-text">
            <h1>AI-Powered Plant<br>Disease Detection</h1>
            <p>Upload a leaf photo and get an instant diagnosis with<br>
               cause, symptoms, and treatment recommendations.</p>
            <div class="hero-crops">
              <span class="hero-crop-pill">🥔 Potato</span>
              <span class="hero-crop-pill">🍅 Tomato</span>
              <span class="hero-crop-pill">🌶️ Bell Pepper</span>
            </div>
          </div>
          <div class="hero-stats">
            <div class="hero-stat">
              <div class="hero-stat-num">15</div>
              <div class="hero-stat-label">Disease Classes</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-num">74.8%</div>
              <div class="hero-stat-label">Val Accuracy</div>
            </div>
            <div class="hero-stat">
              <div class="hero-stat-num">3.2M</div>
              <div class="hero-stat-label">Parameters</div>
            </div>
          </div>
        </div>
        """)

        # ── Mobile Top Bar (hidden on desktop) ────────────────────────────
        gr.HTML("""
        <div id="pg-mobile-bar">
          <div class="mob-header">
            <span style="font-size:28px;">🌿</span>
            <div>
              <div class="mob-title">PlantGuard AI</div>
              <div class="mob-sub">Plant Disease Detection</div>
            </div>
          </div>
          <div class="mob-pills">
            <span class="mob-pill">🥔 Potato</span>
            <span class="mob-pill">🍅 Tomato</span>
            <span class="mob-pill">🌶️ Bell Pepper</span>
            <span class="mob-pill">15 classes</span>
          </div>
        </div>
        """)

        # ── Main Content Grid ─────────────────────────────────────────────
        with gr.Group(elem_id="pg-content"):

            # Left column
            with gr.Group(elem_id="pg-left-panel"):

                with gr.Group(elem_classes=["panel-card"]):
                    gr.HTML('<div class="panel-section-title">📷 Upload Leaf Photo</div>')
                    img_input = gr.Image(
                        type="pil", label=None, show_label=False,
                        height=280, elem_classes=["image-container"],
                    )

                analyze_btn = gr.Button(
                    "🔍  Analyze Leaf",
                    variant="primary", size="lg",
                    elem_id="pg-analyze-btn",
                )

                gr.HTML("""
                <div id="pg-left-tips">
                  <div class="tips-title">📸 Tips for best results</div>
                  <div class="tips-list">
                    <div class="tip-item">✅ Single leaf, close-up shot</div>
                    <div class="tip-item">✅ Good natural or artificial lighting</div>
                    <div class="tip-item">✅ Sharp focus, no motion blur</div>
                    <div class="tip-item">✅ Plain or blurred background</div>
                  </div>
                </div>
                """)

            # Right column (results)
            with gr.Group(elem_id="pg-right-panel"):
                with gr.Tabs():
                    with gr.TabItem("🩺  Diagnosis"):
                        info_out = gr.HTML(value=_placeholder_html())
                    with gr.TabItem("📊  Confidence"):
                        chart_out = gr.Image(show_label=False)
                    with gr.TabItem("📋  All Results"):
                        table_out = gr.Dataframe(
                            headers=["#", "Disease / Condition", "Severity", "Confidence"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False, show_label=False,
                        )

        # ── Desktop Footer ────────────────────────────────────────────────
        gr.HTML("""
        <footer id="pg-footer">
          <div>
            <div class="footer-brand">🌿 PlantGuard AI</div>
            <div class="footer-sub">Automated Plant Disease Detection · Deep Learning PBL</div>
          </div>
          <div class="footer-badges">
            <span class="footer-badge">Batch-Normalized CNN</span>
            <span class="footer-badge">PlantVillage Dataset</span>
            <span class="footer-badge">PyTorch</span>
            <span class="footer-badge">Gradio</span>
          </div>
        </footer>
        """)

        # ── Wire events ───────────────────────────────────────────────────
        analyze_btn.click(
            fn=predict_disease,
            inputs=[img_input],
            outputs=[info_out, chart_out, table_out],
        )

    return demo

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  🌿 PlantGuard AI — Plant Disease Detection")
    print("=" * 52)
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {BEST_CKPT}")
    print("  Open   : http://127.0.0.1:7860\n")
    demo = build_ui()
    demo.queue()
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=host, server_port=port, show_error=True, share=False)
