import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from io import BytesIO

st.set_page_config(page_title="MSMarineToxNet", layout="centered")

st.markdown("## MSMarineToxNet")
st.markdown("仅需上传含 m/z、intensity、RI、species 的 Excel 表格，即可预测毒性")
st.markdown("---")

# 下载模板
with open("demo_input_template.xlsx", "rb") as file:
    st.download_button("📥 下载 demo 模板文件", file, file_name="demo_input_template.xlsx")

# 加载模型（只加载一次）
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final.h5", compile=False)

# 处理上传的 Excel 文件
def preprocess_input(df):
    mz_cols = [col for col in df.columns if "m/z" in col]
    intensity_cols = [col for col in df.columns if "intensity" in col]
    mz_intensity_cols = mz_cols + intensity_cols

    X_mz_intensity = df[mz_intensity_cols].values.astype(float)
    X_mz_intensity = X_mz_intensity.reshape((X_mz_intensity.shape[0], -1, 1))

    X_ri = df[['RI']].values.astype(float)
    X_species = df[['species']].values.astype(float)

    return X_ri, X_mz_intensity, X_species

# 上传预测文件
uploaded_file = st.file_uploader("📤 上传填写好的 Excel 文件", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        X_ri, X_mz_intensity, X_species = preprocess_input(df)

        model = load_model()
        preds = model.predict([X_ri, X_mz_intensity, X_species], verbose=0)

        df["Pred_LC50"] = preds.flatten()

        # 直接下载修改后的原始表
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        st.success("✅ 预测完成，点击下方按钮下载结果")
        st.download_button(
            "📥 下载带预测值的 Excel 文件",
            data=output.getvalue(),
            file_name="prediction_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"❌ 出错了：{e}")
