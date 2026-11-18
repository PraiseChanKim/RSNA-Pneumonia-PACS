
import streamlit as st

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
    st.success(f"파일 선택 완료: {uploaded_file.name}")

