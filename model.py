import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak',
    'bacterial_panicle_blight', 'brown_spot', 'dead_heart',
    'downy_mildew', 'healthy', 'hispa', 'leaf_blast',
    'leaf_smut', 'neck_blast', 'sheath_blight', 'tungro',
    'harvest_stage',   
]

# ─────────────────────────────────────────────────────────────────
# LABEL DATASET — untuk tampilan di response API
# ─────────────────────────────────────────────────────────────────
DATASET_LABELS = {
    "Citra_Daun_Padi"    : "Citra Daun Padi",
    "JENIS_PENYAKIT_PADI": "Jenis Penyakit Padi",
    "paddy_dataset_v3"   : "Paddy Dataset V3 Augmentasi",
    "Paddy_disease"      : "Paddy Disease Classification",
}

ARCH_LABELS = {
    "efficientnet_b0" : "EfficientNet-B0",
    "inception_v3"    : "InceptionV3",
    "resnet50"        : "ResNet-50",
    "swin_base"       : "Swin Transformer Base",
    "vit"             : "ViT-Base/16",
}

# ─────────────────────────────────────────────────────────────────
# SWIN — 4 dataset dibaca dari .env
# Key env  : MODEL_swin_base_{dataset}
# ─────────────────────────────────────────────────────────────────
SWIN_DATASET_LIST = [
    "Citra_Daun_Padi",
    "JENIS_PENYAKIT_PADI",
    "paddy_dataset_v3",
    "Paddy_disease",
]

# ─────────────────────────────────────────────────────────────────
# 16 MODEL NON-SWIN — path hardcode langsung di sini
# Tidak perlu .env karena path-nya tetap dan tidak sering diubah.
# ─────────────────────────────────────────────────────────────────
HARDCODED_MODEL_PATHS = {
    # ── EfficientNet-B0 ──────────────────────────────────────────
    "efficientnet_b0__Citra_Daun_Padi"    : "models/efficientnet_b0_Citra_Daun_Padi_best.h5",
    "efficientnet_b0__JENIS_PENYAKIT_PADI": "models/efficientnet_b0_JENIS_PENYAKIT_PADI_best.h5",
    "efficientnet_b0__paddy_dataset_v3"   : "models/efficientnet_b0_paddy-dataset-v3-augmentasi_best.h5",
    "efficientnet_b0__Paddy_disease"      : "models/efficientnet_b0_Paddy-disease-classification_best.h5",

    # ── InceptionV3 ──────────────────────────────────────────────
    "inception_v3__Citra_Daun_Padi"       : "models/inception_v3_Citra_Daun_Padi_best.h5",
    "inception_v3__JENIS_PENYAKIT_PADI"   : "models/inception_v3_JENIS_PENYAKIT_PADI_best.h5",
    "inception_v3__paddy_dataset_v3"      : "models/inception_v3_paddy-dataset-v3-augmentasi_best.h5",
    "inception_v3__Paddy_disease"         : "models/inception_v3_Paddy-disease-classification_best.h5",

    # ── ResNet-50 ─────────────────────────────────────────────────
    "resnet50__Citra_Daun_Padi"           : "models/resnet50_Citra_Daun_Padi_best.h5",
    "resnet50__JENIS_PENYAKIT_PADI"       : "models/resnet50_JENIS_PENYAKIT_PADI_best.h5",
    "resnet50__paddy_dataset_v3"          : "models/resnet50_paddy-dataset-v3-augmentasi_best.h5",
    "resnet50__Paddy_disease"             : "models/resnet50_Paddy-disease-classification_best.h5",

    # ── ViT-Base/16 ───────────────────────────────────────────────
    "vit__Citra_Daun_Padi"                : "models/vit_Citra_Daun_Padi_best.h5",
    "vit__JENIS_PENYAKIT_PADI"            : "models/vit_JENIS_PENYAKIT_PADI_best.h5",
    "vit__paddy_dataset_v3"               : "models/vit_paddy-dataset-v3-augmentasi_best.h5",
    "vit__Paddy_disease"                  : "models/vit_Paddy-disease-classification_best.h5",
}


# ═════════════════════════════════════════════════════════════════
# LOAD MODEL UTAMA (dari MODEL_PATH di .env)
# Dipakai oleh main.py saat startup untuk /predict
# ═════════════════════════════════════════════════════════════════
def load_model():
    """
    Load model utama dari MODEL_PATH di .env.
    Default: models/swin_base_Citra_Daun_Padi_best.h5
    """
    model_path = os.getenv("MODEL_PATH", "models/swin_base_Citra_Daun_Padi_best.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}\n"
            f"Pastikan file .h5 ada di folder models/"
        )
    return load_model_from_path(model_path)


# ═════════════════════════════════════════════════════════════════
# LOAD SEMUA 20 MODEL
# Swin  (4)  → path dari .env (MODEL_swin_base_{dataset})
# Lainnya (16) → path hardcode di HARDCODED_MODEL_PATHS
# ═════════════════════════════════════════════════════════════════
def load_all_models() -> dict:
    """
    Load semua 20 model (5 arsitektur × 4 dataset).

    Return dict dengan key format: "{arsitektur}__{dataset}"
    Contoh key: "swin_base__Citra_Daun_Padi", "vit__Paddy_disease"

    Setiap value adalah dict:
    {
        "model"        : <torch model>  | None jika gagal,
        "class_names"  : [...],
        "model_name"   : str,
        "arch_key"     : str,   # misal "swin_base"
        "dataset_key"  : str,   # misal "Citra_Daun_Padi"
        "arch_label"   : str,   # misal "Swin Transformer Base"
        "dataset_label": str,   # misal "Citra Daun Padi"
        "path"         : str | None,
        "status"       : "loaded" | "error" | "not_configured",
        "error"        : str | None,
    }
    """
    all_models = {}

    # ── 1. Swin Transformer — baca path dari .env ─────────────────
    print("\n══ Loading Swin Transformer (path dari .env) ══")
    for dataset in SWIN_DATASET_LIST:
        env_key   = f"MODEL_swin_base_{dataset}"
        model_key = f"swin_base__{dataset}"
        path      = os.getenv(env_key)

        if not path:
            print(f"⚠️  ENV '{env_key}' tidak ditemukan — skip.")
            all_models[model_key] = _make_error_entry(
                arch    = "swin_base",
                dataset = dataset,
                path    = None,
                error   = f"ENV {env_key} tidak diset di .env",
                status  = "not_configured",
            )
            continue

        all_models[model_key] = _try_load(
            arch    = "swin_base",
            dataset = dataset,
            path    = path,
        )

    # ── 2. EfficientNet, Inception, ResNet, ViT — path hardcode ───
    print("\n══ Loading 16 model Non-Swin (path hardcode) ══")
    for model_key, path in HARDCODED_MODEL_PATHS.items():
        # model_key format: "efficientnet_b0__Citra_Daun_Padi"
        arch, dataset = model_key.split("__", 1)
        all_models[model_key] = _try_load(
            arch    = arch,
            dataset = dataset,
            path    = path,
        )

    # ── Ringkasan ──────────────────────────────────────────────────
    loaded  = sum(1 for v in all_models.values() if v["status"] == "loaded")
    total   = len(all_models)
    print(f"\n{'═'*50}")
    print(f"  ✅ Model berhasil di-load : {loaded} / {total}")
    print(f"{'═'*50}\n")

    return all_models


# ─────────────────────────────────────────────────────────────────
# HELPER — Load & error entry
# ─────────────────────────────────────────────────────────────────
def _try_load(arch: str, dataset: str, path: str) -> dict:
    """Coba load model dari path, return dict status."""
    try:
        model, class_names, model_name = load_model_from_path(path)
        return {
            "model"        : model,
            "class_names"  : class_names,
            "model_name"   : model_name,
            "arch_key"     : arch,
            "dataset_key"  : dataset,
            "arch_label"   : ARCH_LABELS.get(arch, arch),
            "dataset_label": DATASET_LABELS.get(dataset, dataset),
            "path"         : path,
            "status"       : "loaded",
            "error"        : None,
        }
    except Exception as e:
        print(f"❌ Gagal load {arch}__{dataset}: {e}")
        return _make_error_entry(arch, dataset, path, str(e), "error")


def _make_error_entry(
    arch: str, dataset: str, path, error: str, status: str = "error"
) -> dict:
    return {
        "model"        : None,
        "class_names"  : CLASS_NAMES,
        "model_name"   : arch,
        "arch_key"     : arch,
        "dataset_key"  : dataset,
        "arch_label"   : ARCH_LABELS.get(arch, arch),
        "dataset_label": DATASET_LABELS.get(dataset, dataset),
        "path"         : path,
        "status"       : status,
        "error"        : error,
    }


# ═════════════════════════════════════════════════════════════════
# LOAD MODEL DARI PATH TERTENTU
# ═════════════════════════════════════════════════════════════════
def _get_best_acc(checkpoint: dict) -> str:
    for key in ('best_val_acc', 'val_acc', 'best_acc', 'accuracy', 'acc'):
        val = checkpoint.get(key)
        if val is not None:
            return val
    return 'N/A'


def load_model_from_path(model_path: str):
    """
    Load model dari path tertentu.
    Return: (model, class_names, model_name)
    """
    checkpoint  = torch.load(model_path, map_location=device, weights_only=False)
    model_name  = checkpoint.get('model_name', 'swin_base')
    num_classes = checkpoint.get('num_classes', len(CLASS_NAMES))
    class_names = checkpoint.get('class_names', CLASS_NAMES)

    best_acc = _get_best_acc(checkpoint)
    best_f1  = checkpoint.get('best_val_f1', checkpoint.get('val_f1', 'N/A'))

    print(f"─────────────────────────────────────────────")
    print(f"  Model     : {model_name}")
    print(f"  Path      : {model_path}")
    print(f"  Kelas     : {num_classes}")
    print(f"  Best Acc  : {best_acc}")
    print(f"  Best F1   : {best_f1}")
    print(f"─────────────────────────────────────────────")

    model = _build_architecture(model_name, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✅ '{model_name}' berhasil di-load dari {os.path.basename(model_path)}")
    return model, class_names, model_name


# ═════════════════════════════════════════════════════════════════
# BANGUN ARSITEKTUR MODEL
# ═════════════════════════════════════════════════════════════════
def _build_architecture(model_name: str, num_classes: int):
    """
    Rebuild arsitektur model sesuai struktur yang disimpan di checkpoint.

    Struktur classifier per model:
    ─ resnet50        : Linear(2048→512) + BN + ReLU + Dropout(0.5) + Linear(512→C)
    ─ efficientnet_b0 : Dropout(0.5) + Linear(1280→C)
    ─ inception_v3    : Dropout(0.5) + Linear(2048→C)
                        AuxLogits.fc = Linear(768→C)
    ─ vit             : timm, drop_rate=0.3, drop_path_rate=0.1
    ─ swin_base       : timm, drop_rate=0.3, drop_path_rate=0.2
    """

    # ── ResNet-50 ──────────────────────────────────────────────────
    if model_name == 'resnet50':
        m    = models.resnet50(weights=None)
        in_f = m.fc.in_features                   # 2048
        m.fc = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    # ── EfficientNet-B0 ────────────────────────────────────────────
    elif model_name == 'efficientnet_b0':
        m    = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features        # 1280
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, num_classes),
        )

    # ── InceptionV3 ────────────────────────────────────────────────
    elif model_name == 'inception_v3':
        m = models.inception_v3(weights=None, init_weights=False)
        m.aux_logits = True
        in_f = m.fc.in_features                   # 2048
        m.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, num_classes),
        )
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)

    # ── ViT-Base/16 ────────────────────────────────────────────────
    elif model_name == 'vit':
        import timm
        m = timm.create_model(
            'vit_base_patch16_224',
            pretrained     = False,
            num_classes    = num_classes,
            drop_rate      = 0.3,
            drop_path_rate = 0.1,
        )

    # ── Swin Transformer Base / Tiny ──────────────────────────────
    elif model_name in ('swin_base', 'swin_tiny'):
        import timm
        arch = (
            'swin_base_patch4_window7_224'
            if model_name == 'swin_base'
            else 'swin_tiny_patch4_window7_224'
        )
        m = timm.create_model(
            arch,
            pretrained     = False,
            num_classes    = num_classes,
            drop_rate      = 0.3,
            drop_path_rate = 0.2,
        )

    else:
        raise ValueError(
            f"Model '{model_name}' tidak dikenali. "
            f"Pilihan: resnet50 | efficientnet_b0 | inception_v3 | vit | swin_base"
        )

    return m


# ═════════════════════════════════════════════════════════════════
# TRANSFORM GAMBAR
# ═════════════════════════════════════════════════════════════════
def _get_transform(model_name: str):
    """
    Transform gambar sesuai model.
    InceptionV3 → 299×299; model lain → 224×224.
    """
    size = 299 if model_name == 'inception_v3' else 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


# ═════════════════════════════════════════════════════════════════
# PREDIKSI
# ═════════════════════════════════════════════════════════════════
def predict(
    image_bytes : bytes,
    model,
    class_names : list,
    model_name  : str = 'swin_base',
) -> tuple[str, float]:
    """
    Prediksi penyakit dari bytes gambar.

    Return:
        predicted_class : nama kelas penyakit
        confidence      : persentase keyakinan model (0–100)
    """
    transform  = _get_transform(model_name)
    img        = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        # InceptionV3 saat eval() tidak return tuple, tapi handle jika ada
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs     = torch.softmax(outputs, dim=1)
        conf, idx = probs.max(dim=1)

    predicted_class = class_names[idx.item()]
    confidence      = round(conf.item() * 100, 2)
    return predicted_class, confidence
