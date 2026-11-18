
import streamlit as st
import pydicom
import numpy as np

from pydicom.pixels import apply_voi_lut

st.set_page_config(
    page_title="RSNA Pneumonia PACS",
    layout="wide"
)

st.title("RSNA Pneumonia PACS")
st.write("폐렴 의심 영역 시각화 및 의료영상 검토 시스템")
st.info("교육용 프로젝트이며 실제 의료 진단에 사용할 수 없습니다.")

uploaded_file = st.file_uploader(
    "DICOM 파일을 업로드하세요",
    type=["dcm"]
)

if uploaded_file is not None:
    try:
        dicom = pydicom.dcmread(uploaded_file)

        image = apply_voi_lut(dicom.pixel_array, dicom)
        image = image.astype(np.float32)

        if image.ndim != 2:
            raise ValueError("2차원 흑백 DICOM만 지원합니다.")

        min_value = np.min(image)
        max_value = np.max(image)

        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value) * 255
        else:
            image = np.zeros_like(image)

        image = image.astype(np.uint8)

        if dicom.get("PhotometricInterpretation") == "MONOCHROME1":
            image = 255 - image

        st.success("DICOM 영상 불러오기 성공!")

        st.image(
            image,
            caption="Original Chest X-ray",
            clamp=True
        )

        st.subheader("영상 정보")
        st.write("가로:", dicom.Columns)
        st.write("세로:", dicom.Rows)

    except Exception as e:
        st.error(f"DICOM 영상 처리 실패: {e}")
