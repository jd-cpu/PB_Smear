"""
PB Smear 백혈구 탐지 웹 애플리케이션
- Detection      : YOLOv8 (wbc_detector.pt)
- Classification : EfficientNetB0 (model.h5)
- Web server     : Flask  (JSON API + Bootstrap 5 SPA)
"""

import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# =============================================
# 설정값
# =============================================
UPLOAD_FOLDER      = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
MAX_CONTENT_LENGTH = 64 * 1024 * 1024   # 64 MB

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================
# Crop 저장 설정 (fine-tuning 데이터 수집용)
# True로 켜두면 YOLO가 탐지한 모든 WBC crop을 static/crops/ 에 저장
# 저장 후 직접 클래스 폴더로 분류 → Google Drive 업로드 → Colab fine-tuning
# =============================================
SAVE_CROPS = True
CROPS_DIR  = os.path.join('static', 'crops')
os.makedirs(CROPS_DIR, exist_ok=True)

# =============================================
# 백혈구 클래스 매핑
# =============================================
CLASS_NAMES = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

CLASS_NAMES_KO = {
    'Basophil':   '호염기구',
    'Eosinophil': '호산구',
    'Lymphocyte': '림프구',
    'Monocyte':   '단핵구',
    'Neutrophil': '호중구',
}

# 이미지 위 바운딩박스 색 (BGR)
CLASS_COLORS = {
    'Neutrophil': (255, 100,   0),
    'Lymphocyte': (  0, 200, 255),
    'Monocyte':   (255,   0, 200),
    'Eosinophil': (  0,  50, 255),
    'Basophil':   ( 50,  50,  50),
    'Unknown':    (  0, 255,   0),
}
GIANT_COLOR = (0, 0, 255)   # Giant cell 강조 색 (BGR 빨간색)

# cv2.putText 한글 미지원 → 영문 약칭 사용
CLASS_NAMES_EN_SHORT = {
    'Basophil':   'Baso',
    'Eosinophil': 'Eosi',
    'Lymphocyte': 'Lymph',
    'Monocyte':   'Mono',
    'Neutrophil': 'Neut',
}

IMG_SIZE = 224   # EfficientNetB0 입력 크기

_CLAHE       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLASS_PRIOR = np.array([0.01, 0.03, 0.30, 0.05, 0.60], dtype=np.float32)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================
# 모델 로드
# =============================================
YOLO_MODEL       = None
YOLO_CUSTOM_PATH = 'wbc_detector.pt'
YOLO_FALLBACK    = 'yolov8n.pt'
CLASSIFIER       = None
CLASSIFIER_PATH  = 'model_finetuned.h5'


def load_models():
    global YOLO_MODEL, CLASSIFIER

    # YOLOv8
    try:
        from ultralytics import YOLO
        if os.path.exists(YOLO_CUSTOM_PATH):
            YOLO_MODEL = YOLO(YOLO_CUSTOM_PATH)
            print(f'[INFO] YOLOv8 커스텀 모델 로드: {YOLO_CUSTOM_PATH}')
        else:
            YOLO_MODEL = YOLO(YOLO_FALLBACK)
            print(f'[WARNING] {YOLO_CUSTOM_PATH} 없음 → {YOLO_FALLBACK} 사용')
    except Exception as e:
        print(f'[ERROR] YOLOv8 로드 실패: {e}')

    # EfficientNetB0
    if os.path.exists(CLASSIFIER_PATH):
        try:
            import tensorflow as tf
            CLASSIFIER = tf.keras.models.load_model(CLASSIFIER_PATH)
            print(f'[INFO] 분류 모델 로드: {CLASSIFIER_PATH}')
        except Exception as e:
            print(f'[WARNING] 분류 모델 로드 실패: {e}')
    else:
        print('[INFO] model.h5 없음 → 탐지만 수행')


# =============================================
# 탐지: YOLOv8 슬라이딩 윈도우
# =============================================
TILE_SIZE    = 640
TILE_OVERLAP = 0.25


def detect_wbc_yolo(image_bgr, conf_threshold=0.25, iou_threshold=0.35):
    if YOLO_MODEL is None:
        return []

    device   = 'mps' if _mps_available() else 'cpu'
    img_h, img_w = image_bgr.shape[:2]
    stride   = int(TILE_SIZE * (1 - TILE_OVERLAP))
    raw_boxes, raw_scores = [], []

    y_starts = sorted(set(max(0, min(y, img_h - TILE_SIZE))
                          for y in range(0, img_h, stride) if y < img_h))
    x_starts = sorted(set(max(0, min(x, img_w - TILE_SIZE))
                          for x in range(0, img_w, stride) if x < img_w))

    for y0 in y_starts:
        for x0 in x_starts:
            tile = image_bgr[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue
            results = YOLO_MODEL.predict(
                source=tile, conf=conf_threshold, iou=iou_threshold,
                device=device, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    x1 += x0; x2 += x0; y1 += y0; y2 += y0
                    if (x2 - x1) < 15 or (y2 - y1) < 15:
                        continue
                    raw_boxes.append([x1, y1, x2, y2])
                    raw_scores.append(conf)

    if not raw_boxes:
        return []

    boxes_np  = np.array(raw_boxes,  dtype=np.float32)
    scores_np = np.array(raw_scores, dtype=np.float32)

    keep = _nms(boxes_np, scores_np, iou_thresh=0.35)
    boxes_np, scores_np = boxes_np[keep], scores_np[keep]
    keep2 = _center_dedup(boxes_np, scores_np, dist_ratio=0.7)
    boxes_np = boxes_np[keep2]

    result_boxes = []
    for i in range(len(boxes_np)):
        x1, y1, x2, y2 = map(int, boxes_np[i])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)
        result_boxes.append((x1, y1, x2 - x1, y2 - y1))
    return result_boxes


def _nms(boxes, scores, iou_thresh):
    """IoU + containment 이중 조건 NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter       = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou         = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        containment = inter / (np.minimum(areas[i], areas[order[1:]]) + 1e-6)
        suppress = (iou > iou_thresh) | (containment > 0.6)
        order = order[1:][~suppress]
    return keep


def _center_dedup(boxes, scores, dist_ratio=0.5):
    """중심 거리 기반 중복 제거."""
    if len(boxes) == 0:
        return []
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w  = boxes[:, 2] - boxes[:, 0]
    h  = boxes[:, 3] - boxes[:, 1]
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        min_size = np.minimum(np.sqrt(w[i] * h[i]),
                              np.sqrt(w[order[1:]] * h[order[1:]]))
        dist = np.sqrt((cx[i] - cx[order[1:]])**2 + (cy[i] - cy[order[1:]])**2)
        order = order[1:][dist > min_size * dist_ratio]
    return keep


def _mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# =============================================
# 분류: EfficientNetB0
# =============================================
def classify_crop(crop_bgr):
    if CLASSIFIER is None or crop_bgr.size == 0:
        return 'Unknown', 0.0

    from tensorflow.keras.applications.efficientnet import preprocess_input

    # CLAHE → 염색 편차 정규화
    lab       = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(lab)
    lab_eq    = cv2.merge([_CLAHE.apply(l), a, b])
    crop_norm = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 미세 샤프닝 → prior 과보정 억제 효과
    blur       = cv2.GaussianBlur(crop_norm, (0, 0), 1.0)
    crop_sharp = cv2.addWeighted(crop_norm, 1.2, blur, -0.2, 0)

    img_rgb     = cv2.cvtColor(crop_sharp, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)
    img_input   = np.expand_dims(
        preprocess_input(img_resized.astype(np.float32)), axis=0)

    preds     = CLASSIFIER.predict(img_input, verbose=0)
    raw_probs = preds[0].astype(np.float32)

    adjusted  = raw_probs * _CLASS_PRIOR
    adjusted  = adjusted / (adjusted.sum() + 1e-8)

    class_idx  = int(np.argmax(adjusted))
    confidence = float(adjusted[class_idx])
    return CLASS_NAMES[class_idx], confidence


# =============================================
# 파이프라인: 탐지 → 분류 → Giant 판별 → 결과 이미지 + 통계 + NLR + 소견
# =============================================
GIANT_RATIO = 1.5   # 클래스 평균 면적의 이 배수 이상이면 Giant로 표시


def process_image(filepath, original_filename):
    image = cv2.imread(filepath)
    if image is None:
        return None

    MAX_DIM = 3200
    h, w = image.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    result_image = image.copy()
    boxes        = detect_wbc_yolo(image)
    total        = len(boxes)

    # ── 1차 루프: 분류 + 면적 수집 ──────────────────────────────
    detections   = []   # (x, y, w_box, h_box, class_name, confidence, area)
    areas_by_cls = {}   # class_name -> [area, ...]

    for x, y, w_box, h_box in boxes:
        crop = image[y:y + h_box, x:x + w_box]
        # fine-tuning용 crop 저장 (SAVE_CROPS=True 일 때)
        if SAVE_CROPS and crop.size > 0:
            cv2.imwrite(os.path.join(CROPS_DIR, f'{uuid.uuid4().hex[:10]}.jpg'), crop)
        class_name, conf = classify_crop(crop)
        area             = w_box * h_box
        detections.append((x, y, w_box, h_box, class_name, conf, area))
        areas_by_cls.setdefault(class_name, []).append(area)

    # 클래스별 평균 면적
    mean_area_by_cls = {cls: float(np.mean(arr)) for cls, arr in areas_by_cls.items()}

    # ── 2차 루프: 이미지 렌더링 ────────────────────────────────
    class_counts = {}   # ko_name -> count

    for i, (x, y, w_box, h_box, class_name, confidence, area) in enumerate(detections):
        class_ko  = CLASS_NAMES_KO.get(class_name, class_name)
        class_counts[class_ko] = class_counts.get(class_ko, 0) + 1

        mean_area = mean_area_by_cls.get(class_name, area)
        is_giant  = (area > mean_area * GIANT_RATIO) and (len(areas_by_cls.get(class_name, [])) > 1)

        color = GIANT_COLOR if is_giant else CLASS_COLORS.get(class_name, CLASS_COLORS['Unknown'])

        cv2.rectangle(result_image, (x, y), (x + w_box, y + h_box), color, 2)

        short = CLASS_NAMES_EN_SHORT.get(class_name, 'WBC')
        if class_name != 'Unknown':
            label = f'[Giant] {short} {confidence:.0%}' if is_giant else f'{short} {confidence:.0%}'
        else:
            label = f'WBC #{i + 1}'

        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result_image, (x, y - 22), (x + lsz[0] + 4, y), color, -1)
        cv2.putText(result_image, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 결과 이미지 저장
    base_name       = os.path.splitext(original_filename)[0]
    result_filename = f'result_{base_name}_{uuid.uuid4().hex[:6]}.jpg'
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, result_filename),
                result_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 클래스별 통계 (count + pct, 내림차순)
    classes_list = [
        {'name': ko,
         'count': cnt,
         'pct': round(cnt / total * 100, 1) if total > 0 else 0}
        for ko, cnt in sorted(class_counts.items(), key=lambda kv: -kv[1])
    ]
    pct_map = {c['name']: c['pct'] for c in classes_list}

    # ── NLR 계산 ────────────────────────────────────────────────
    neut_n  = class_counts.get(CLASS_NAMES_KO['Neutrophil'], 0)
    lymph_n = class_counts.get(CLASS_NAMES_KO['Lymphocyte'],  0)
    if lymph_n > 0:
        nlr = round(neut_n / lymph_n, 2)
        if   nlr < 1.0:  nlr_status = 'low'
        elif nlr <= 3.0: nlr_status = 'normal'
        elif nlr <= 5.0: nlr_status = 'elevated'
        else:            nlr_status = 'high'
    else:
        nlr, nlr_status = None, 'na'

    # ── 자동 임상 소견 생성 ──────────────────────────────────────
    findings = []
    eosi_pct  = pct_map.get(CLASS_NAMES_KO['Eosinophil'], 0)
    neut_pct  = pct_map.get(CLASS_NAMES_KO['Neutrophil'], 0)
    lymph_pct = pct_map.get(CLASS_NAMES_KO['Lymphocyte'], 0)

    if eosi_pct > 5:
        findings.append('🚨 호산구 비율 증가: 알레르기 질환 또는 기생충 감염 차별 진단 요망')
    if neut_pct > 75:
        findings.append('🚨 호중구 비율 증가: 급성 세균성 감염증 의심')
    if lymph_pct > 45:
        findings.append('🚨 림프구 비율 증가: 바이러스성 감염증 의심')
    if not findings:
        findings.append('✅ 특이 소견 없음: 정상 범위 내의 백혈구 분율 소견입니다.')

    return {
        'result_filename': result_filename,
        'total':           total,
        'classes':         classes_list,
        'nlr':             nlr,
        'nlr_status':      nlr_status,
        'findings':        findings,
    }


# =============================================
# Flask 라우팅
# =============================================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           yolo_loaded=(YOLO_MODEL is not None),
                           classifier_loaded=(CLASSIFIER is not None))


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '파일을 선택해 주세요.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False,
                        'error': '지원하지 않는 형식입니다. (PNG / JPG / JPEG / TIF / BMP)'}), 400

    filename        = secure_filename(file.filename)
    unique_filename = f'{uuid.uuid4().hex}_{filename}'
    upload_path     = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(upload_path)

    result = process_image(upload_path, unique_filename)
    if result is None:
        return jsonify({'success': False,
                        'error': '이미지를 처리할 수 없습니다. 다른 파일을 시도해 주세요.'}), 500

    return jsonify({
        'success':           True,
        'original_url':      url_for('static', filename=f'uploads/{unique_filename}'),
        'result_url':        url_for('static', filename=f'uploads/{result["result_filename"]}'),
        'total':             result['total'],
        'classes':           result['classes'],
        'nlr':               result['nlr'],
        'nlr_status':        result['nlr_status'],
        'findings':          result['findings'],
        'yolo_loaded':       YOLO_MODEL is not None,
        'classifier_loaded': CLASSIFIER is not None,
    })


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5001)
