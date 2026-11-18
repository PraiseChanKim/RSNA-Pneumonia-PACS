
from pathlib import Path

import streamlit as st
import pydicom
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from pydicom.pixels import apply_voi_lut


# =========================================
# 1. Streamlit 기본 설정
# =========================================

st.set_page_config(
    page_title="RSNA Pneumonia PACS",
    layout="wide"
)

st.title("RSNA Pneumonia PACS")
st.write("흉부 X-ray 영상 조회 및 폐 음영 영역 시각화")
st.info("교육용 프로젝트이며 실제 의료 진단용이 아닙니다.")


# =========================================
# 2. 데이터 경로
# =========================================

DATA_DIR = Path("data")

IMAGE_DIR = DATA_DIR / "stage_2_train_images"

LABEL_PATH = DATA_DIR / "stage_2_train_labels.csv"

CLASS_PATH = DATA_DIR / "stage_2_detailed_class_info.csv"


# =========================================
# 3. CSV 데이터 불러오기
# =========================================

@st.cache_data
def load_csv():
    labels = pd.read_csv(LABEL_PATH)
    classes = pd.read_csv(CLASS_PATH)

    return labels, classes


# =========================================
# 4. DICOM 영상 변환 함수
# =========================================

def load_dicom(path):

    dicom = pydicom.dcmread(path)

    image = apply_voi_lut(
        dicom.pixel_array,
        dicom
    )

    image = image.astype(np.float32)

    if image.ndim != 2:
        raise ValueError("2차원 흑백 영상만 지원합니다.")

    min_value = image.min()
    max_value = image.max()

    if max_value > min_value:
        image = (
            (image - min_value)
            / (max_value - min_value)
            * 255
        )
    else:
        image = np.zeros_like(image)

    image = image.astype(np.uint8)

    if dicom.get("PhotometricInterpretation") == "MONOCHROME1":
        image = 255 - image

    return Image.fromarray(image).convert("RGB")


# =========================================
# 5. 파일 확인
# =========================================

if (
    not IMAGE_DIR.exists()
    or not LABEL_PATH.exists()
    or not CLASS_PATH.exists()
):
    st.error("필요한 학습용 영상 또는 CSV 파일이 없습니다.")
    st.stop()


labels, classes = load_csv()


# =========================================
# 6. 분류 선택
# =========================================

st.subheader("X-ray 영상 선택")

category = st.selectbox(
    "영상 분류를 선택하세요",
    [
        "Normal",
        "Lung Opacity",
        "No Lung Opacity / Not Normal"
    ]
)

filtered = classes[
    classes["class"] == category
]

patient_ids = (
    filtered["patientId"]
    .drop_duplicates()
    .tolist()
)

patient_id = st.selectbox(
    "영상 ID 선택",
    patient_ids
)

show_bbox = st.checkbox(
    "Bounding Box 표시",
    value=True
)


# =========================================
# 7. DICOM 영상 읽기
# =========================================

dicom_path = IMAGE_DIR / f"{patient_id}.dcm"

if not dicom_path.exists():
    st.warning("해당 DICOM 파일이 없습니다.")
    st.stop()

try:
    original = load_dicom(dicom_path)

    result = original.copy()

    # =====================================
    # 8. Bounding Box 데이터 검색
    # =====================================

    boxes = labels[
        (labels["patientId"] == patient_id)
        & (labels["Target"] == 1)
    ]

    if show_bbox:

        draw = ImageDraw.Draw(result)

        for _, box in boxes.iterrows():

            x1 = int(box["x"])
            y1 = int(box["y"])

            x2 = x1 + int(box["width"])
            y2 = y1 + int(box["height"])

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=4
            )

    # =====================================
    # 9. 결과 출력
    # =====================================

    st.subheader("영상 비교")

    col1, col2 = st.columns(2)

    with col1:
        st.write("원본 X-ray")
        st.image(
            original,
            use_container_width=True
        )

    with col2:
        st.write("Bounding Box 결과")
        st.image(
            result,
            use_container_width=True
        )

    # =====================================
    # 10. 분류 정보
    # =====================================

    st.subheader("영상 분석 정보")

    st.write("영상 ID:", patient_id)
    st.write("데이터셋 분류:", category)
    st.write("Bounding Box 개수:", len(boxes))

    if category == "Normal":
        st.success("데이터셋에서 정상으로 분류된 영상입니다.")

    elif category == "Lung Opacity":
        st.warning("폐 음영 의심 영역이 표시된 영상입니다.")

    else:
        st.info("정상은 아니지만 해당 폐 음영 라벨은 없습니다.")

    # =====================================
    # 11. Bounding Box 좌표 출력
    # =====================================

    if len(boxes) > 0:

        st.subheader("Bounding Box 좌표")

        st.dataframe(
            boxes[["x", "y", "width", "height"]],
            hide_index=True
        )

except Exception as e:
    st.error(f"영상 처리 오류: {e}")
