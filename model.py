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
    'leaf_smut', 'neck_blast', 'sheath_blight', 'tungro'
]


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


def load_model_from_path(model_path: str):
    """
    Load model dari path tertentu.
    Dipakai untuk load semua 20 model sekaligus di startup.
    """
    checkpoint  = torch.load(model_path, map_location=device, weights_only=False)
    model_name  = checkpoint.get('model_name', 'swin_base')
    num_classes = checkpoint.get('num_classes', len(CLASS_NAMES))
    class_names = checkpoint.get('class_names', CLASS_NAMES)

    print(f"─────────────────────────────────────────────")
    print(f"  Model     : {model_name}")
    print(f"  Path      : {model_path}")
    print(f"  Kelas     : {num_classes}")
    print(f"  Best Acc  : {checkpoint.get('best_val_acc', 'N/A')}")
    print(f"  Best F1   : {checkpoint.get('best_val_f1', 'N/A')}")
    print(f"─────────────────────────────────────────────")

    model = _build_architecture(model_name, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✅ '{model_name}' berhasil di-load dari {os.path.basename(model_path)}")
    return model, class_names, model_name


def _build_architecture(model_name: str, num_classes: int):
    """
    Rebuild arsitektur model sesuai notebook Kaggle (FIXED2).

    Struktur classifier per model:
    ─ resnet50        : Linear(2048→512) + BN + ReLU + Dropout(0.5) + Linear(512→C)
    ─ efficientnet_b0 : Dropout(0.5) + Linear(1280→C)
    ─ inception_v3    : fc sama dengan resnet50; AuxLogits.fc = Linear(768→C)
    ─ vit             : timm, drop_rate=0.3, drop_path_rate=0.1
    ─ swin_base       : timm, drop_rate=0.3, drop_path_rate=0.2
    """

    # ── ResNet-50 ──────────────────────────────────────────────────
    if model_name == 'resnet50':
        m = models.resnet50(weights=None)
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
        m = models.efficientnet_b0(weights=None)
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
            nn.Linear(in_f, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
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

    # ── Swin Transformer Base ──────────────────────────────────────
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
        outputs   = model(img_tensor)
        probs     = torch.softmax(outputs, dim=1)
        conf, idx = probs.max(dim=1)

    predicted_class = class_names[idx.item()]
    confidence      = round(conf.item() * 100, 2)
    return predicted_class, confidence
