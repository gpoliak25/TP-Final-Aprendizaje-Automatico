"""
CNN Pipeline Runner + Predicción en vivo
TP Final — Aprendizaje Automático · Radiografías Veterinarias
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
from pathlib import Path
import nbformat
import nbformat.v4 as nbv4
import subprocess
import re
import tempfile
import io

_ANSI = re.compile(r'\x1b\[[0-9;]*[mGKHFABCDJK]')

# ─── Paths ────────────────────────────────────────────────────────────────────
NOTEBOOK_DIR = Path(__file__).parent

# Locate jupyter-nbconvert: try the local venv first, then fall back to PATH
_LOCAL_NBCONV = Path(r"C:\Users\gpoli\venvs\caece-mineria\Scripts\jupyter-nbconvert.exe")
if _LOCAL_NBCONV.exists():
    JUPYTER_NBCONV = _LOCAL_NBCONV
else:
    import shutil as _shutil
    _which = _shutil.which("jupyter-nbconvert") or _shutil.which("jupyter")
    JUPYTER_NBCONV = Path(_which) if _which else Path("jupyter-nbconvert")
CLASS_NAMES    = ["ok", "patologica"]
IMG_SIZE       = (224, 224)

# ─── Pipeline metadata ────────────────────────────────────────────────────────
PIPELINE = [
    {"pattern": "01_*", "title": "Exploración",      "icon": "🔍", "color": "#7c3aed",
     "desc": "Conteo, balance de clases, resoluciones, imágenes corruptas"},
    {"pattern": "02_*", "title": "Preparación",       "icon": "✂️", "color": "#2563eb",
     "desc": "Split 70/15/15, data augmentation, pesos de clase"},
    {"pattern": "03_*", "title": "CNN desde cero",    "icon": "🧱", "color": "#0891b2",
     "desc": "Arquitectura propia — Conv2D, BatchNorm, Dropout"},
    {"pattern": "04_*", "title": "Transfer Learning", "icon": "🔁", "color": "#059669",
     "desc": "EfficientNetB0 / MobileNetV2 con fine-tuning"},
    {"pattern": "05_*", "title": "Evaluación",        "icon": "📊", "color": "#d97706",
     "desc": "Confusion matrix, ROC, F1-score, comparación de modelos"},
    {"pattern": "06_*", "title": "GradCAM",           "icon": "🗺️", "color": "#dc2626",
     "desc": "Mapas de activación para interpretabilidad"},
]

STATUS_ICON  = {"pending": "○", "running": "◉", "success": "●", "error": "✕"}
STATUS_COLOR = {"pending": "#484f58", "running": "#388bfd", "success": "#3fb950", "error": "#f85149"}
STATUS_LABEL = {"pending": "PENDIENTE", "running": "EJECUTANDO", "success": "OK", "error": "ERROR"}

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Pipeline · Rx Veterinarias",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#0d1117 !important; }
  .main .block-container { padding-top:1.2rem; max-width:1340px; }
  *,p,span,label,div { color:#c9d1d9; }
  a { color:#58a6ff !important; }
  h1 { color:#f0f6fc !important; font-size:1.95rem !important; letter-spacing:-.4px; margin:0 !important; }
  h2 { color:#e6edf3 !important; font-size:1.25rem !important; }
  h3 { color:#8b949e !important; }

  [data-testid="stSidebar"] { background:#161b22 !important; border-right:1px solid #21262d; }

  /* Tabs */
  [data-testid="stTabs"] { background:transparent; }
  button[data-baseweb="tab"] { background:transparent !important; color:#8b949e !important; border-bottom:2px solid transparent !important; font-weight:500; }
  button[data-baseweb="tab"][aria-selected="true"] { color:#e6edf3 !important; border-bottom:2px solid #388bfd !important; }
  [data-testid="stTabsContent"] { padding-top:1.2rem; }

  /* Buttons */
  .stButton>button {
    background:#1f6feb !important; color:#fff !important;
    border:1px solid #388bfd !important; border-radius:10px !important;
    font-weight:700 !important; font-size:1.05rem !important;
    padding:1rem 1.6rem !important; min-height:64px !important;
    width:100% !important; transition:all .15s !important;
  }
  .stButton>button:hover {
    background:#388bfd !important;
    transform:translateY(-2px);
    box-shadow:0 6px 20px #1f6feb55 !important;
  }
  .stButton>button:active { transform:translateY(0) !important; }

  /* Botón reset — gris secundario */
  .stButton>button[kind="secondary"], .btn-reset .stButton>button {
    background:#21262d !important; border-color:#30363d !important;
    color:#c9d1d9 !important;
  }
  .btn-reset .stButton>button:hover { background:#30363d !important; box-shadow:none !important; transform:none !important; }

  /* Inputs */
  .stTextInput>div>div { background:#161b22 !important; border:1px solid #30363d !important; border-radius:6px; }
  input { color:#e6edf3 !important; }
  .stSelectbox>div>div { background:#161b22 !important; border-color:#30363d !important; }
  [data-testid="stToggle"] { accent-color:#1f6feb; }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background:#161b22 !important;
    border:2px dashed #30363d !important;
    border-radius:10px !important;
  }
  [data-testid="stFileUploaderDropzone"] { background:transparent !important; }

  /* Expander */
  details { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; margin-bottom:6px !important; }
  summary { color:#e6edf3 !important; font-size:.88rem; padding:8px 12px; }

  /* Code */
  pre,code,.stCode { background:#010409 !important; color:#79c0ff !important; border:1px solid #21262d !important; border-radius:6px !important; }

  /* Metrics */
  [data-testid="stMetric"] { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:.8rem 1rem; }
  [data-testid="stMetricValue"] { color:#58a6ff !important; font-size:1.6rem !important; }
  [data-testid="stMetricLabel"] { color:#8b949e !important; font-size:.78rem !important; }

  /* Progress */
  .stProgress>div>div { background:#1f6feb !important; border-radius:4px; }

  /* Alerts */
  .stInfo    { background:#0d1b2a !important; border-color:#1f6feb !important; }
  .stSuccess { background:#0d1f12 !important; border-color:#3fb950 !important; }
  .stError   { background:#1f0d0d !important; border-color:#f85149 !important; }
  .stWarning { background:#1f1a0d !important; border-color:#d29922 !important; }

  hr { border-color:#21262d !important; margin:1.2rem 0 !important; }
  footer,#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "statuses" not in st.session_state:
    st.session_state.statuses = ["pending"] * len(PIPELINE)
if "logs" not in st.session_state:
    st.session_state.logs = [""] * len(PIPELINE)
if "model_cache" not in st.session_state:
    st.session_state.model_cache = {}     # path -> model

# ─── Helpers ──────────────────────────────────────────────────────────────────
def find_nb(pattern: str):
    hits = sorted(NOTEBOOK_DIR.glob(f"{pattern}.ipynb"))
    return hits[0] if hits else None

def cell_count(nb_path) -> int:
    try:
        nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
        return sum(1 for c in nb.cells if c.cell_type == "code" and c.source.strip())
    except Exception:
        return 0

# ─── Pipeline card HTML ───────────────────────────────────────────────────────
def render_cards(statuses):
    html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:10px 0 18px;">'
    for i, step in enumerate(PIPELINE):
        s = statuses[i]
        sc, si, sl = STATUS_COLOR[s], STATUS_ICON[s], STATUS_LABEL[s]
        nb = find_nb(step["pattern"])
        cc = cell_count(nb) if nb else "–"
        html += f"""
        <div style="background:#161b22;border:1px solid #21262d;border-top:3px solid {step['color']};
                    border-radius:8px;padding:14px 12px 38px;position:relative;min-height:155px;">
          <div style="font-size:1.55rem;line-height:1">{step['icon']}</div>
          <div style="color:#f0f6fc;font-weight:700;font-size:.88rem;margin:5px 0 3px">{step['title']}</div>
          <div style="color:#8b949e;font-size:.74rem;line-height:1.45">{step['desc']}</div>
          <div style="position:absolute;bottom:10px;left:12px;right:12px;
                      display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#30363d;font-size:.7rem;">{cc} celdas</span>
            <span style="color:{sc};font-size:.72rem;font-weight:700;letter-spacing:.5px">{si} {sl}</span>
          </div>
        </div>"""
    html += "</div>"
    return html

# ─── Terminal log ──────────────────────────────────────────────────────────────
def render_log(lines):
    visible = lines[-35:]
    body = "<br>".join(
        ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").rstrip()
        for ln in visible
    )
    return (
        '<div style="background:#010409;border:1px solid #21262d;border-radius:6px;'
        'padding:12px 14px;font-family:Consolas,monospace;font-size:.78rem;'
        'color:#7ee787;max-height:260px;overflow-y:auto;line-height:1.55;">'
        + body + "</div>"
    )

# ─── Notebook patching ────────────────────────────────────────────────────────
COLAB_STUB = """\
import sys, types as _t
_g=_t.ModuleType("google"); _gc=_t.ModuleType("google.colab")
class _D:
    def mount(self,*a,**k): print("[LOCAL] Drive.mount() omitido")
_gc.drive=_D(); sys.modules["google"]=_g; sys.modules["google.colab"]=_gc
del _g,_gc,_D,_t
"""

def patch_notebook(nb_path, base_dir, mock):
    nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        src = cell.source
        if mock:
            src = src.replace("from google.colab import drive",
                              "# [mock] from google.colab import drive")
            src = re.sub(r"drive\.mount\(['\"][^'\"]*['\"]\)",
                         "print('[LOCAL] drive.mount omitido')", src)
        if base_dir:
            src = re.sub(r"BASE_DIR\s*=\s*['\"][^'\"]*['\"]",
                         f"BASE_DIR = r'{base_dir}'", src)
        cell.source = src
    if mock:
        nb.cells.insert(0, nbv4.new_code_cell(COLAB_STUB))
    tmp = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w", encoding="utf-8")
    nbformat.write(nb, tmp); tmp.close()
    return tmp.name

def replay_nb_outputs(i, log_ph):
    """Reproduce outputs already stored in the notebook (from a prior Colab run)."""
    import time
    step = PIPELINE[i]
    nb_path = find_nb(step["pattern"])
    if nb_path is None:
        return False, f"No encontrado: {step['pattern']}"
    try:
        nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
    except Exception as e:
        return False, str(e)

    log_lines = [f"📄  {nb_path.name}  [MODO DEMO — outputs de Colab]", ""]
    code_cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    for j, cell in enumerate(code_cells):
        preview = cell.source.split("\n")[0][:70].strip()
        log_lines.append(f"▶ Celda {j+1}/{len(code_cells)}  — {preview}")
        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")
            if otype == "stream":
                text = "".join(out.get("text", []))
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {})
                text = "".join(data.get("text/plain", []))
            elif otype == "error":
                text = f"[ERROR en Colab] {out.get('ename')}: {out.get('evalue')}"
            else:
                text = ""
            for line in text.splitlines()[:15]:
                log_lines.append(f"  {line}")
        log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
        time.sleep(0.05)   # small pause so updates are visible

    log_lines.append("")
    log_lines.append(f"✓ {nb_path.name} — outputs reproducidos exitosamente.")
    log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
    return True, "\n".join(log_lines)


def execute_nb(i, base_dir, mock, log_ph):
    step = PIPELINE[i]
    nb_path = find_nb(step["pattern"])
    if nb_path is None:
        return False, f"No encontrado: {step['pattern']}"
    tmp_path = patch_notebook(nb_path, base_dir, mock)
    cmd = [str(JUPYTER_NBCONV), "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=600",
           "--ExecutePreprocessor.kernel_name=caece-mineria",
           "--output", str(nb_path), tmp_path]
    log_lines = [f"$ {JUPYTER_NBCONV.name} — {step['title']}", ""]
    log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log_lines.append(_ANSI.sub("", line.rstrip()))
            log_ph.markdown(render_log(log_lines), unsafe_allow_html=True)
        proc.wait()
        ok = proc.returncode == 0
    except Exception as e:
        log_lines.append(f"ERROR: {e}"); ok = False
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass
    return ok, "\n".join(log_lines)

# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo …")
def load_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(model, image_bytes):
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)               # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]   # shape (2,)
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding-bottom:14px;">
  <span style="font-size:2.8rem">🩻</span>
  <div>
    <h1>CNN Pipeline · Radiografías Veterinarias</h1>
    <p style="margin:2px 0 0;color:#8b949e;font-size:.9rem;">
      TP Final · Aprendizaje Automático &nbsp;|&nbsp;
      Clasificación <strong style="color:#3fb950">normal</strong> /
      <strong style="color:#f85149">patológico</strong>
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_pipeline, tab_viewer, tab_pred, tab_pres = st.tabs([
    "  🔄  Pipeline Runner  ",
    "  📄  Inspeccionar notebooks  ",
    "  🩻  Predicción en vivo  ",
    "  📊  Presentación  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("""
    <p style="color:#8b949e;margin-bottom:1.2rem;">
      Cargá un modelo entrenado (<code>.keras</code> / <code>.h5</code>) y subí una radiografía para obtener la predicción.
    </p>
    """, unsafe_allow_html=True)

    # ── Model selector ────────────────────────────────────────────────────────
    # Buscar automáticamente modelos en subdirectorios comunes
    default_model_dirs = [
        NOTEBOOK_DIR / "modelos",
        NOTEBOOK_DIR / "models",
        NOTEBOOK_DIR,
    ]
    found_models = []
    for d in default_model_dirs:
        found_models += list(d.glob("*.keras")) + list(d.glob("*.h5"))
    found_models = sorted(set(found_models))

    col_model, col_btn = st.columns([5, 1])
    with col_model:
        if found_models:
            model_options = {p.name: str(p) for p in found_models}
            model_options["📂 Otra ruta…"] = "__custom__"
            sel = st.selectbox("Modelo", options=list(model_options.keys()),
                               label_visibility="visible")
            if model_options[sel] == "__custom__":
                model_path = st.text_input("Ruta completa al archivo del modelo:",
                                           placeholder=r"C:\ruta\al\modelo.keras")
            else:
                model_path = model_options[sel]
        else:
            model_path = st.text_input(
                "Ruta al modelo entrenado (`.keras` o `.h5`):",
                placeholder=r"C:\ruta\al\modelo.keras",
                help="Descargá el modelo desde Google Drive y pegá la ruta local aquí.",
            )

    with col_btn:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        load_btn = st.button("⬇ Cargar", use_container_width=True)

    # Modelo status
    model = None
    if model_path and Path(model_path).is_file():
        try:
            model = load_model(model_path)
            st.success(f"✅ Modelo cargado: `{Path(model_path).name}`")
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {e}")
    elif model_path and not Path(model_path).is_file():
        st.warning("⚠️  Archivo no encontrado. Verificá la ruta o descargá el modelo desde Drive.")

    if not found_models and not model_path:
        st.info(
            "**¿Dónde está el modelo?**  \n"
            "Los modelos se guardan en Google Drive en `TP_Final Aprendizaje Automatico/modelos/`.  \n"
            "Descargalos y pegá la ruta local arriba, o copiá la carpeta `modelos/` junto a este archivo."
        )

    st.divider()

    # ── Image upload + prediction ─────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Subir radiografía")
        uploaded = st.file_uploader(
            "Arrastrá o seleccioná la imagen",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            from PIL import Image as PILImage
            img_bytes = uploaded.read()
            img_show = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(img_show, caption=uploaded.name, use_container_width=True)
            predict_btn = st.button("▶  Predecir", use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#161b22;border:2px dashed #30363d;border-radius:10px;
                        padding:40px;text-align:center;color:#484f58;">
              <div style="font-size:2.5rem">🫁</div>
              <div style="margin-top:8px;font-size:.9rem">Arrastrá una Rx aquí</div>
            </div>
            """, unsafe_allow_html=True)
            predict_btn = False

    with col_result:
        st.markdown("#### Resultado")
        result_ph = st.empty()

        if uploaded and predict_btn:
            if model is None:
                result_ph.error("❌ Cargá un modelo primero.")
            else:
                with st.spinner("Analizando imagen …"):
                    try:
                        label, conf, probs = predict(model, img_bytes)
                        is_pat = (label == "patologica")
                        color  = "#f85149" if is_pat else "#3fb950"
                        emoji  = "⚠️" if is_pat else "✅"
                        titulo = "PATOLÓGICA" if is_pat else "NORMAL (OK)"

                        p_ok  = probs[0] * 100
                        p_pat = probs[1] * 100

                        result_ph.markdown(f"""
                        <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                                    padding:28px 24px;border-top:4px solid {color};">
                          <div style="font-size:2.8rem;margin-bottom:8px">{emoji}</div>
                          <div style="color:{color};font-size:2rem;font-weight:800;
                                      letter-spacing:1px;margin-bottom:4px">{titulo}</div>
                          <div style="color:#8b949e;font-size:.85rem;margin-bottom:20px">
                            confianza: <strong style="color:{color}">{conf*100:.1f}%</strong>
                          </div>

                          <div style="margin-bottom:10px">
                            <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:4px;">
                              <span>Normal (ok)</span>
                              <span style="color:#3fb950">{p_ok:.1f}%</span>
                            </div>
                            <div style="background:#21262d;border-radius:4px;height:8px;">
                              <div style="background:#3fb950;border-radius:4px;height:8px;width:{p_ok:.1f}%"></div>
                            </div>
                          </div>

                          <div>
                            <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:4px;">
                              <span>Patológica</span>
                              <span style="color:#f85149">{p_pat:.1f}%</span>
                            </div>
                            <div style="background:#21262d;border-radius:4px;height:8px;">
                              <div style="background:#f85149;border-radius:4px;height:8px;width:{p_pat:.1f}%"></div>
                            </div>
                          </div>

                          <div style="margin-top:20px;padding-top:16px;border-top:1px solid #21262d;
                                      color:#484f58;font-size:.75rem;">
                            Modelo: {Path(model_path).name}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        result_ph.error(f"❌ Error en predicción: {e}")
        else:
            result_ph.markdown("""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                        padding:40px 24px;text-align:center;color:#484f58;">
              <div style="font-size:2.2rem;margin-bottom:8px">📊</div>
              <div style="font-size:.9rem">El resultado aparecerá aquí</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    cards_ph = st.empty()
    cards_ph.markdown(render_cards(st.session_state.statuses), unsafe_allow_html=True)

    total_cells = sum(cell_count(find_nb(s["pattern"])) for s in PIPELINE if find_nb(s["pattern"]))
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Notebooks",      len(PIPELINE))
    mc2.metric("Total celdas",   total_cells)
    mc3.metric("Completados",    sum(1 for s in st.session_state.statuses if s == "success"))
    mc4.metric("Con error",      sum(1 for s in st.session_state.statuses if s == "error"))
    mc5.metric("Pendientes",     sum(1 for s in st.session_state.statuses if s == "pending"))

    st.divider()

    # ── Modo de ejecución ────────────────────────────────────────────────────
    col_mode, col_mock = st.columns([4, 2])
    with col_mode:
        exec_mode = st.radio(
            "Modo",
            ["🎬  Demo (outputs de Colab)", "⚙️  Ejecutar local (requiere datos)"],
            horizontal=True, label_visibility="collapsed",
        )
        demo_mode = exec_mode.startswith("🎬")

    with col_mock:
        if not demo_mode:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            use_mock = st.toggle("Mock Colab", value=True)
        else:
            use_mock = True

    # BASE_DIR sólo relevante en modo local
    if not demo_mode:
        base_dir = st.text_input("base_dir", label_visibility="collapsed",
                                 value="/content/drive/MyDrive/TP_Final Aprendizaje Automatico",
                                 placeholder="📂  BASE_DIR — ruta local a los datos …")
    else:
        base_dir = ""
        st.info(
            "**Modo Demo**: reproduce los outputs ya calculados en Colab. "
            "No necesita datos locales ni conexión a Drive.",
            icon="ℹ️",
        )

    # ── Botones grandes ───────────────────────────────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    ba, bb, bc = st.columns(3, gap="medium")

    with ba:
        btn_all = st.button(
            "▶▶  Ejecutar Todo\n\nCorre los 6 notebooks en orden",
            use_container_width=True, key="run_all"
        )

    with bb:
        btn_one = st.button(
            "▶  Ejecutar Uno\n\nCorre el notebook seleccionado",
            use_container_width=True, key="run_one"
        )

    with bc:
        st.markdown('<div class="btn-reset">', unsafe_allow_html=True)
        btn_reset = st.button(
            "↺  Resetear estados\n\nLimpia todos los indicadores",
            use_container_width=True, key="reset", type="secondary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    nb_sel = st.selectbox(
        "Notebook a ejecutar (botón ▶ Ejecutar Uno):",
        range(len(PIPELINE)),
        format_func=lambda i: f"{PIPELINE[i]['icon']}  {PIPELINE[i]['title']}",
        key="nb_sel",
    )

    if btn_reset:
        st.session_state.statuses = ["pending"] * len(PIPELINE)
        st.session_state.logs     = [""]        * len(PIPELINE)
        st.rerun()

    run_targets = list(range(len(PIPELINE))) if btn_all else ([nb_sel] if btn_one else None)

    if run_targets is not None:
        prog_ph = st.progress(0.0); status_ph = st.empty(); log_ph = st.empty()
        for seq, idx in enumerate(run_targets):
            step = PIPELINE[idx]
            st.session_state.statuses[idx] = "running"
            cards_ph.markdown(render_cards(st.session_state.statuses), unsafe_allow_html=True)
            status_ph.info(f"⚡  **[{seq+1}/{len(run_targets)}] {step['icon']} {step['title']}** …")
            if demo_mode:
                ok, log = replay_nb_outputs(idx, log_ph)
            else:
                ok, log = execute_nb(idx, base_dir, use_mock, log_ph)
            st.session_state.statuses[idx] = "success" if ok else "error"
            st.session_state.logs[idx]     = log
            prog_ph.progress((seq + 1) / len(run_targets))
            cards_ph.markdown(render_cards(st.session_state.statuses), unsafe_allow_html=True)
        all_ok = all(st.session_state.statuses[i] == "success" for i in run_targets)
        if all_ok:
            status_ph.success("✅ Completado exitosamente.")
        else:
            failed = [PIPELINE[i]["title"] for i in run_targets if st.session_state.statuses[i] == "error"]
            status_ph.error(f"❌ Falló: {', '.join(failed)}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NOTEBOOK VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_viewer:
    viewer_idx = st.selectbox(
        "Notebook:",
        range(len(PIPELINE)),
        format_func=lambda i: f"{PIPELINE[i]['icon']}  {PIPELINE[i]['title']}",
        key="viewer_sel",
    )
    nb_path = find_nb(PIPELINE[viewer_idx]["pattern"])
    if nb_path:
        log_content = st.session_state.logs[viewer_idx]
        if log_content.strip():
            with st.expander("📋 Log de la última ejecución"):
                st.code(log_content[-3000:], language=None)
        try:
            nb = nbformat.read(open(nb_path, encoding="utf-8"), as_version=4)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        for j, cell in enumerate(nb.cells):
            if cell.cell_type == "markdown":
                first = cell.source.strip().split("\n")[0]
                if first.startswith("#"):
                    st.markdown(
                        f'<div style="color:#8b949e;font-size:.82rem;margin:10px 0 2px;'
                        f'padding-left:4px;border-left:2px solid #30363d">{first}</div>',
                        unsafe_allow_html=True)
            elif cell.cell_type == "code" and cell.source.strip():
                preview = cell.source.split("\n")[0][:60].strip()
                with st.expander(f"Celda {j+1}  —  `{preview}…`", expanded=(j < 2)):
                    st.code(cell.source, language="python")
                    for out in cell.get("outputs", []):
                        if out.get("output_type") in ("stream", "execute_result"):
                            text = "".join(out.get("text",
                                   out.get("data", {}).get("text/plain", [])))
                            if text.strip():
                                st.text(text[:400])
    else:
        st.info("Notebook no encontrado en este directorio.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRESENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pres:
    import plotly.graph_objects as go

    def card(content, border_color="#1f6feb", pad="22px 24px"):
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;'
            f'border-left:4px solid {border_color};border-radius:10px;'
            f'padding:{pad};margin-bottom:14px">{content}</div>',
            unsafe_allow_html=True,
        )

    # ── SLIDE 1 · Portada ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1117 60%,#0d1f2d);
                border:1px solid #21262d;border-radius:14px;padding:36px 40px;
                margin-bottom:24px;text-align:center;">
      <div style="color:#8b949e;font-size:.82rem;letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:8px;">
        Trabajo Práctico Integrador · Maestría en Ciencia de Datos e Innovación Empresarial
      </div>
      <h2 style="color:#e6edf3;font-size:1.7rem;font-weight:800;
                 line-height:1.3;margin:10px 0 6px;">
        Detección de Patologías en Radiografías de Tórax Veterinarias<br>
        <span style="color:#388bfd">mediante Redes Neuronales Convolucionales</span>
      </h2>
      <div style="color:#8b949e;font-size:.9rem;margin:10px 0 18px">
        Un análisis del costo del error y la interpretabilidad clínica
      </div>
      <div style="color:#c9d1d9;font-size:.88rem">
        Lic. Lorena López &nbsp;·&nbsp; Lic. Gisela Poliak &nbsp;·&nbsp;
        Universidad CAECE &nbsp;·&nbsp; Junio 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SLIDE 2 · El Desafío Clínico ─────────────────────────────────────────
    st.markdown("## El Desafío Clínico")
    card(
        "<span style='color:#e6edf3'>La interpretación de radiografías exige lectura cuidadosa. "
        "Depende de la experiencia del profesional y es <strong>vulnerable a la fatiga</strong> "
        "y alta carga de trabajo.</span>",
        border_color="#388bfd",
    )

    col_trad, col_ai, col_pts = st.columns([2, 2, 1.5])
    with col_trad:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:18px">
          <div style="color:#8b949e;font-size:.8rem;margin-bottom:12px;letter-spacing:1px">
            FLUJO DIAGNÓSTICO TRADICIONAL</div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <div style="background:#21262d;border-radius:6px;padding:10px 14px;color:#c9d1d9;font-size:.84rem">
              👤  Ingreso del Paciente y Toma de Placa</div>
            <div style="text-align:center;color:#484f58">↓</div>
            <div style="background:#21262d;border-radius:6px;padding:10px 14px;color:#c9d1d9;font-size:.84rem">
              🕐  Espera en Cola de Lectura</div>
            <div style="text-align:center;color:#484f58">↓</div>
            <div style="background:#21262d;border-radius:6px;padding:10px 14px;color:#c9d1d9;font-size:.84rem">
              👁  Análisis Manual por Especialista</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ai:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #388bfd;border-radius:8px;padding:18px">
          <div style="color:#388bfd;font-size:.8rem;margin-bottom:12px;letter-spacing:1px">
            FLUJO POTENCIADO POR IA</div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <div style="background:#21262d;border-radius:6px;padding:10px 14px;color:#c9d1d9;font-size:.84rem">
              👤  Ingreso del Paciente y Toma de Placa</div>
            <div style="text-align:center;color:#484f58">↓</div>
            <div style="background:#0d1f38;border:1px solid #388bfd;border-radius:6px;
                        padding:10px 14px;color:#58a6ff;font-size:.84rem">
              🤖  Filtro Automático AI (Triage)</div>
            <div style="text-align:center;color:#484f58">↓</div>
            <div style="background:#0d1f38;border:1px solid #388bfd;border-radius:6px;
                        padding:10px 14px;color:#58a6ff;font-size:.84rem">
              ⚠️  Priorización de Casos Críticos</div>
            <div style="text-align:center;color:#484f58">↓</div>
            <div style="background:#21262d;border-radius:6px;padding:10px 14px;color:#c9d1d9;font-size:.84rem">
              👨‍⚕️  Análisis Enfocado por Especialista</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_pts:
        for icon, title, desc in [
            ("⏱", "Velocidad", "Acelerar el triage de casos urgentes"),
            ("🤝", "Consistencia", "Segunda opinión para reducir variabilidad"),
            ("⚠️", "Priorización", "Destacar estudios con alta probabilidad de hallazgos"),
        ]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                        padding:10px 12px;margin-bottom:8px">
              <div style="color:#388bfd;font-size:.82rem;font-weight:700">{icon} {title}</div>
              <div style="color:#8b949e;font-size:.76rem;margin-top:3px">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── SLIDE 3 · Dataset ─────────────────────────────────────────────────────
    st.markdown("## El Espacio de Datos")
    col_pie, col_split = st.columns([1, 1])

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Normal (ok)", "Patológica"],
            values=[245, 196],
            hole=0.55,
            marker_colors=["#58a6ff", "#d29922"],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#c9d1d9", margin=dict(t=20, b=20, l=0, r=0),
            showlegend=False, height=280,
            annotations=[dict(text="<b>441</b><br>imágenes", x=0.5, y=0.5,
                              font_size=16, font_color="#e6edf3", showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_split:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        for split, n, pct, color in [
            ("Train", 308, "70%", "#388bfd"),
            ("Val",   66,  "15%", "#d29922"),
            ("Test",  67,  "15%", "#3fb950"),
        ]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-left:4px solid {color};
                        border-radius:8px;padding:12px 16px;margin-bottom:8px;
                        display:flex;justify-content:space-between;align-items:center">
              <span style="color:{color};font-weight:700;font-size:.9rem">{split}</span>
              <span style="color:#e6edf3;font-size:1.1rem;font-weight:800">{n}</span>
              <span style="color:#8b949e;font-size:.85rem">{pct}</span>
            </div>""", unsafe_allow_html=True)
        card(
            "<span style='color:#8b949e;font-size:.82rem'>"
            "Data augmentation aplicada <strong style='color:#c9d1d9'>solo en train</strong>: "
            "flip horizontal, rotación ±18°, zoom 10%, contraste 10%.</span>",
            border_color="#d29922",
        )

    st.divider()

    # ── SLIDE 4 · Métrica Estrella ────────────────────────────────────────────
    st.markdown("## La Métrica Estrella: Costo Asimétrico del Error")
    card(
        "<span style='color:#e6edf3'>En diagnóstico médico, la <strong>Exactitud (Accuracy) es engañosa</strong>. "
        "Un modelo que prediga todo como "sano" ignorará a los pacientes críticos.</span>",
        border_color="#f85149",
    )

    st.markdown("""
    <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;max-width:640px;margin:0 auto 16px">
      <div></div>
      <div style="text-align:center;color:#8b949e;font-size:.8rem;padding:6px">Pred: Sano</div>
      <div style="text-align:center;color:#8b949e;font-size:.8rem;padding:6px">Pred: Patológico</div>

      <div style="color:#8b949e;font-size:.8rem;padding:8px;writing-mode:vertical-lr;
                  text-align:center;transform:rotate(180deg)">Real</div>

      <div style="background:#21262d;border-radius:6px;padding:18px;text-align:center">
        <div style="color:#3fb950;font-size:1.4rem">✔</div>
        <div style="color:#c9d1d9;font-size:.82rem;font-weight:600">Verdadero Negativo</div>
      </div>
      <div style="background:#2d0f0f;border:1px solid #f85149;border-radius:6px;padding:18px;text-align:center">
        <div style="color:#f85149;font-weight:700;font-size:.88rem">Falso Negativo</div>
        <div style="color:#f85149;font-size:.76rem;margin-top:4px">Error Crítico: consecuencias irreversibles</div>
      </div>

      <div></div>

      <div style="background:#2d1f0f;border:1px solid #d29922;border-radius:6px;padding:18px;text-align:center">
        <div style="color:#d29922;font-weight:700;font-size:.88rem">Falso Positivo</div>
        <div style="color:#d29922;font-size:.76rem;margin-top:4px">Falsa Alarma: costo de revisión adicional</div>
      </div>
      <div style="background:#21262d;border-radius:6px;padding:18px;text-align:center">
        <div style="color:#3fb950;font-size:1.4rem">✔</div>
        <div style="color:#c9d1d9;font-size:.82rem;font-weight:600">Verdadero Positivo</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    card(
        "🎯  <strong style='color:#388bfd'>Objetivo Clínico:</strong> "
        "<span style='color:#c9d1d9'>Maximizar el <strong>Recall (Sensibilidad)</strong>. "
        "Priorizamos detectar la mayor cantidad posible de patologías reales, "
        "aceptando conscientemente una mayor tasa de falsas alarmas.</span>",
        border_color="#388bfd",
    )

    st.divider()

    # ── SLIDE 5 · Arquitecturas ───────────────────────────────────────────────
    st.markdown("## El Enfrentamiento Analítico: Dos Enfoques")
    col_cnn, col_tl = st.columns(2, gap="large")

    def arch_layer(name, extra=""):
        color = "#0d1f38"
        return (f'<div style="background:{color};border:1px solid #388bfd;border-radius:5px;'
                f'padding:7px 12px;text-align:center;color:#58a6ff;font-size:.78rem;margin:3px 0">'
                f'{name}'
                + (f'<span style="color:#484f58;font-size:.7rem;margin-left:6px">{extra}</span>' if extra else "")
                + '</div>')

    with col_cnn:
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:18px">
          <div style="color:#e6edf3;font-weight:700;font-size:.95rem;margin-bottom:12px">
            🧱 CNN desde Cero (Línea Base)</div>
          {''.join([
            arch_layer("Input (224×224×3)"),
            arch_layer("Rescaling (1./255)"),
            arch_layer("Conv2D(32) + BN + MaxPool", "896 params"),
            arch_layer("Conv2D(64) + BN + MaxPool", "18,496 params"),
            arch_layer("Conv2D(128) + BN + MaxPool", "73,856 params"),
            arch_layer("Conv2D(128) + BN + MaxPool", "147,584 params"),
            arch_layer("GlobalAvgPool2D"),
            arch_layer("Dense(256) + Dropout", "33,024 params"),
            arch_layer("Dense(2, softmax)", "514 params"),
          ])}
          <div style="margin-top:12px;border-top:1px solid #21262d;padding-top:10px">
            <div style="color:#8b949e;font-size:.78rem">
              ▪ ~423.000 parámetros entrenables<br>
              ▪ <span style="color:#f85149">Riesgo: memorización (overfitting) con 441 imágenes</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col_tl:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:18px">
          <div style="color:#e6edf3;font-weight:700;font-size:.95rem;margin-bottom:12px">
            🔁 Transfer Learning (MobileNetV2)</div>
          <div style="background:#0d2d0d;border:1px solid #3fb950;border-radius:6px;
                      padding:10px 14px;text-align:center;color:#3fb950;font-size:.8rem;margin-bottom:8px">
            MobileNetV2 Base — Congelada<br>
            <span style="color:#484f58;font-size:.72rem">Entrenada en &gt;1M imágenes ImageNet</span>
          </div>
          <div style="text-align:center;color:#484f58">↓  Fine-tuning  ↓</div>
          <div style="margin-top:8px">""" +
          ''.join([
            arch_layer("GlobalAvgPool2D"),
            arch_layer("Dense(256) + Dropout"),
            arch_layer("Dense(2, softmax)"),
          ]) + """
          </div>
          <div style="margin-top:12px;border-top:1px solid #21262d;padding-top:10px">
            <div style="color:#8b949e;font-size:.78rem">
              ▪ Fase 1: extracción (base congelada)<br>
              ▪ Fase 2: fine-tuning sutil capas superiores<br>
              ▪ <span style="color:#3fb950">Transfiere comprensión geométrica general</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── SLIDES 6-7 · Resultados ───────────────────────────────────────────────
    st.markdown("## Resultados: CNN vs Transfer Learning")

    col_r1, col_r2 = st.columns(2, gap="large")

    def confusion_fig(tn, fp, fn, tp, title, color):
        z    = [[tn, fp], [fn, tp]]
        text = [[str(tn), str(fp)], [str(fn), str(tp)]]
        fig  = go.Figure(go.Heatmap(
            z=z, text=text, texttemplate="%{text}",
            textfont={"size": 22, "color": "white"},
            colorscale=[[0, "#0d1117"], [1, color]],
            showscale=False,
            x=["ok", "patologica"], y=["ok", "patologica"],
        ))
        fig.update_layout(
            title=dict(text=title, font_color="#c9d1d9", font_size=13, x=0.5),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#c9d1d9", height=220,
            margin=dict(t=36, b=30, l=60, r=10),
            xaxis=dict(title="Predicción", color="#8b949e", gridcolor="#21262d"),
            yaxis=dict(title="Real", color="#8b949e", gridcolor="#21262d"),
        )
        return fig

    with col_r1:
        st.markdown("""
        <div style="background:#1f0d0d;border:1px solid #f85149;border-radius:8px;
                    padding:10px 14px;margin-bottom:8px;text-align:center">
          <span style="color:#f85149;font-weight:700">⚠ La Ilusión de Aprender</span>
          <span style="color:#8b949e;font-size:.8rem"> — CNN desde Cero</span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(confusion_fig(0, 37, 0, 30, "Confusion Matrix · CNN", "#1f6feb"),
                        use_container_width=True, key="cm_cnn")
        card(
            "<span style='color:#8b949e;font-size:.8rem'>Clasifica <strong style='color:#f85149'>TODAS</strong> "
            "las imágenes como patológicas → Recall 100% artificial. "
            "<strong style='color:#f85149'>AUC = 0.451</strong> (peor que el azar). "
            "Modelo clínicamente inútil.</span>",
            border_color="#f85149", pad="12px 14px",
        )

    with col_r2:
        st.markdown("""
        <div style="background:#0d1f12;border:1px solid #3fb950;border-radius:8px;
                    padding:10px 14px;margin-bottom:8px;text-align:center">
          <span style="color:#3fb950;font-weight:700">✓ El Poder de la Transferencia</span>
          <span style="color:#8b949e;font-size:.8rem"> — MobileNetV2</span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(confusion_fig(31, 6, 7, 23, "Confusion Matrix · Transfer Learning", "#3fb950"),
                        use_container_width=True, key="cm_tl")
        card(
            "<span style='color:#8b949e;font-size:.8rem'>Discriminación real con balance genuino. "
            "<strong style='color:#3fb950'>AUC = 0.852</strong>. "
            "El fine-tuning consolida lo aprendido sin degradar la generalización.</span>",
            border_color="#3fb950", pad="12px 14px",
        )

    st.divider()

    # ── SLIDE 8 · Síntesis ────────────────────────────────────────────────────
    st.markdown("## Síntesis de Desempeño")
    col_tab, col_roc = st.columns([1, 1], gap="large")

    with col_tab:
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;font-size:.84rem">
          <thead>
            <tr style="color:#8b949e;border-bottom:1px solid #21262d">
              <th style="padding:8px;text-align:left">Modelo</th>
              <th style="padding:8px;text-align:center">Acc</th>
              <th style="padding:8px;text-align:center">Prec</th>
              <th style="padding:8px;text-align:center">Recall</th>
              <th style="padding:8px;text-align:center">AUC</th>
            </tr>
          </thead>
          <tbody>
            <tr style="color:#484f58;border-bottom:1px solid #21262d">
              <td style="padding:8px">CNN desde cero</td>
              <td style="padding:8px;text-align:center">0.448</td>
              <td style="padding:8px;text-align:center">0.448</td>
              <td style="padding:8px;text-align:center">1.000</td>
              <td style="padding:8px;text-align:center">0.451</td>
            </tr>
            <tr style="background:#0d2d1a;color:#3fb950;font-weight:700">
              <td style="padding:10px 8px;border-radius:6px 0 0 6px">Transfer Learning ✓</td>
              <td style="padding:10px 8px;text-align:center">0.806</td>
              <td style="padding:10px 8px;text-align:center">0.793</td>
              <td style="padding:10px 8px;text-align:center">0.767</td>
              <td style="padding:10px 8px;text-align:center;border-radius:0 6px 6px 0">0.852</td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,0.5,0.6,0.8,1], y=[0,0.45,0.5,0.55,1],
            mode="lines", name="CNN (AUC=0.451)", line=dict(color="#388bfd", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,0.05,0.1,0.2,0.4,1], y=[0,0.6,0.77,0.87,0.97,1],
            mode="lines", name="Transfer (AUC=0.852)", line=dict(color="#d29922", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#484f58", dash="dash", width=1), showlegend=False))
        fig_roc.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#c9d1d9",
            height=220, margin=dict(t=10, b=40, l=40, r=10),
            xaxis=dict(title="Falsos Positivos", gridcolor="#21262d", zeroline=False),
            yaxis=dict(title="Recall", gridcolor="#21262d", zeroline=False),
            legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                        font_size=11, x=0.4, y=0.15),
        )
        st.plotly_chart(fig_roc, use_container_width=True, key="roc")

    st.divider()

    # ── SLIDE 9 · Calibración ─────────────────────────────────────────────────
    st.markdown("## Calibrando la Decisión: El Umbral Clínico")
    col_u1, col_u2 = st.columns(2, gap="large")

    for col, umbral, recall, prec, highlight, color in [
        (col_u1, "0.50", "0.767", "0.793", False, "#388bfd"),
        (col_u2, "0.30", "0.867", "0.650", True,  "#d29922"),
    ]:
        with col:
            border = f"border:2px solid {color}" if highlight else "border:1px solid #21262d"
            st.markdown(f"""
            <div style="background:#161b22;{border};border-radius:10px;padding:20px;text-align:center">
              <div style="color:{color};font-size:1.5rem;font-weight:800;margin-bottom:4px">
                Umbral {umbral}</div>
              <div style="display:flex;justify-content:space-around;margin-top:14px">
                <div>
                  <div style="color:#d29922;font-size:1.4rem;font-weight:800">{recall}</div>
                  <div style="color:#8b949e;font-size:.78rem">Recall ↑</div>
                </div>
                <div>
                  <div style="color:#388bfd;font-size:1.4rem;font-weight:800">{prec}</div>
                  <div style="color:#8b949e;font-size:.78rem">Precision</div>
                </div>
              </div>
              {'<div style="margin-top:12px;background:#2d1e00;border:1px solid #d29922;border-radius:6px;' +
               'padding:8px;color:#d29922;font-size:.78rem">⭐ Seleccionado para producción</div>' if highlight else ''}
            </div>""", unsafe_allow_html=True)

    card(
        "⚕️  <strong style='color:#d29922'>Impacto Clínico:</strong> "
        "<span style='color:#c9d1d9'>Bajar el umbral a 0.30 aumenta el Recall a 0.867, "
        "minimizando los falsos negativos. Asumimos conscientemente la caída en Precisión "
        "porque <em>es preferible revisar a un animal sano que ignorar a uno enfermo.</em></span>",
        border_color="#d29922",
    )

    st.divider()

    # ── SLIDE 10 · Grad-CAM ───────────────────────────────────────────────────
    st.markdown("## Abriendo la 'Caja Negra': Interpretabilidad con Grad-CAM")
    col_g1, col_g2, col_g3 = st.columns(3, gap="medium")

    for col, icon, title, color, body in [
        (col_g1, "🔍", "El Problema", "#f85149",
         "La adopción clínica exige confianza. Un modelo diagnóstico no puede ser una caja negra ciega."),
        (col_g2, "🗺️", "La Solución (Grad-CAM)", "#388bfd",
         "Interroga las últimas capas convolucionales para visualizar qué píxeles influyeron más en la decisión."),
        (col_g3, "✅", "Objetivo de Auditoría", "#d29922",
         "Validar que la red observa estructuras anatómicas plausibles (pulmones, corazón) y no atajos espurios."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-top:3px solid {color};
                        border-radius:8px;padding:16px;height:160px">
              <div style="font-size:1.4rem">{icon}</div>
              <div style="color:{color};font-weight:700;font-size:.85rem;margin:6px 0">{title}</div>
              <div style="color:#8b949e;font-size:.78rem;line-height:1.5">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── SLIDE 11 · Anatomía del Error ────────────────────────────────────────
    st.markdown("## Anatomía del Error: Falsos Negativos y Positivos")

    for tipo, prob, pred, color, desc in [
        ("Falso Negativo", "p=0.36", "ok",         "#f85149",
         "La activación térmica es dispersa y periférica. Al no encontrar un patrón consolidado fuerte, "
         "la atención de la red se diluye y la probabilidad no logra superar el umbral clínico."),
        ("Falso Positivo",  "p=0.54", "patologica", "#d29922",
         "El mapa de calor se aferra a una región focal del tórax con mayor densidad radiológica, "
         "que la red confundió matemáticamente con un hallazgo limítrofe."),
    ]:
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #21262d;border-left:4px solid {color};
                    border-radius:8px;padding:16px 20px;margin-bottom:12px;
                    display:flex;gap:16px;align-items:center">
          <div style="min-width:180px">
            <div style="color:{color};font-weight:700;font-size:.9rem">{tipo}</div>
            <div style="color:#8b949e;font-size:.78rem">Pred: <code>{pred}</code> ({prob})</div>
          </div>
          <div style="color:#8b949e;font-size:.82rem;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

    card(
        "🔬 <strong style='color:#c9d1d9'>Conclusión Diagnóstica:</strong> "
        "<span style='color:#8b949e'>Los errores del modelo radican en <strong style='color:#c9d1d9'>"
        "casos visualmente sutiles y zonas de densidad ambigua</strong>, "
        "no en fallas sistémicas por artefactos irrelevantes.</span>",
        border_color="#388bfd",
    )

    st.divider()

    # ── SLIDE 12 · Conclusiones ───────────────────────────────────────────────
    st.markdown("## Conclusiones y Evolución del Sistema")
    card(
        "🏆 <strong style='color:#e6edf3'>Logro Principal:</strong> "
        "<span style='color:#c9d1d9'>Se demostró que el <strong style='color:#3fb950'>Transfer Learning "
        "(MobileNetV2 con umbral ajustado a 0.30)</strong> supera la severa escasez de datos médicos, "
        "construyendo un <strong>filtro de triage interpretable</strong> que prioriza la sensibilidad.</span>",
        border_color="#3fb950",
    )

    col_f1, col_f2, col_f3 = st.columns(3, gap="medium")
    for col, num, title, color, body in [
        (col_f1, "1", "Validación Cruzada (K-fold)", "#388bfd",
         "Robustecer las estimaciones métricas mitigando la varianza del conjunto de test reducido (~67 imágenes)."),
        (col_f2, "2", "Robustez de Origen",          "#d29922",
         "Entrenar con imágenes de múltiples centros veterinarios para eliminar sesgos de equipo y técnica."),
        (col_f3, "3", "Transición a Multiclase",     "#3fb950",
         "Evolucionar hacia una red capaz de clasificar afecciones específicas: infiltrados, derrames, cardiomegalia."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-top:3px solid {color};
                        border-radius:8px;padding:18px;text-align:center;min-height:170px">
              <div style="color:{color};font-size:1.8rem;font-weight:800">Fase {num}</div>
              <div style="color:#e6edf3;font-weight:700;font-size:.88rem;margin:8px 0">{title}</div>
              <div style="color:#8b949e;font-size:.78rem;line-height:1.5">{body}</div>
            </div>""", unsafe_allow_html=True)
