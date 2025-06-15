import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="MSMarineToxNet", layout="centered")

st.markdown(" MSMarineToxNet")
st.markdown("*Mass Spectrum-Based Marine Toxicity Prediction*")
st.markdown("---")


# 下载 demo 文件
with open("demo_input_template.xlsx", "rb") as file:
    st.download_button(
        label="📥 下载预测模板文件（demo_input_template.xlsx）",
        data=file,
        file_name="demo_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 加载模型
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mass_to_aquatic_toxicity_final_best_RI and species_attention_20250531.h5", compile=False)
    return model

# 数据预处理
def preprocess_data(df):
    try:
        # 提取 m/z + intensity 特征
        mz_cols = [col for col in df.columns if "m/z" in col]
        intensity_cols = [col for col in df.columns if "intensity" in col]
        mz_intensity_cols = mz_cols + intensity_cols

        X_mz_intensity = df[mz_intensity_cols].values.astype(float)
        X_mz_intensity = X_mz_intensity.reshape((X_mz_intensity.shape[0], -1, 1))

        # RI 和 species 特征
        X_ri = df[['RI']].values.astype(float)
        X_species = df[['species']].values.astype(float)

        return X_ri, X_mz_intensity, X_species
    except Exception as e:
        st.error(f"❌ 数据预处理出错：{e}")
        return None, None, None

# 上传 Excel 数据
uploaded_file = st.file_uploader("📤 上传填写好的预测文件（Excel 格式）", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ 文件读取成功，数据预览如下：")
        st.dataframe(df.head())

        # 模型预测
        X_ri, X_mz_intensity, X_species = preprocess_data(df)
        if X_ri is not None:
            model = load_model()
            preds = model.predict([X_ri, X_mz_intensity, X_species], verbose=0)

            # 生成结果表
            pred_df = pd.DataFrame(preds, columns=[f"Pred_LC50_{i+1}" for i in range(preds.shape[1])])
            result_df = pd.concat([df, pred_df], axis=1)

            st.success("✅ 毒性预测完成，结果如下：")
            st.dataframe(result_df.head())

            # 提供下载
            result_file = result_df.to_excel(index=False, engine="openpyxl")
            st.download_button(
                label="📥 下载预测结果",
                data=result_file,
                file_name="toxicity_prediction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"❌ 读取 Excel 文件失败：{e}")
