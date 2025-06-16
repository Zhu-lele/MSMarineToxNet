import streamlit as st
import pandas as pd
import numpy as np
import torch
from io import BytesIO
from torch import nn

# ✅ Streamlit 页面设置
st.set_page_config(page_title="MSMarineToxNet", layout="centered")
st.markdown("## MSMarineToxNet")
st.markdown("仅需上传含 m/z、intensity、RI、species 的 Excel 表格，即可预测毒性")
st.markdown("---")

# ✅ 下载模板按钮
with open("demo_input_template.xlsx", "rb") as file:
    st.download_button("📥 下载 demo 模板文件", file, file_name="demo_input_template.xlsx")

# ✅ 定义模型结构（与训练时一致）
class AttentionCNNModel(nn.Module):
    def __init__(self):
        super(AttentionCNNModel, self).__init__()
        self.ri_dense = nn.Linear(1, 16)
        self.species_dense = nn.Linear(1, 16)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.244),
            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.244),
            nn.Conv1d(64, 128, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.244),
            nn.Conv1d(128, 256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.244),
        )

        self.attention_dense = nn.Linear(32, 1)
        self.attention_sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(16 + 16 + 3072, 128)  # 12×256=3072
        self.dropout_fc1 = nn.Dropout(0.244)
        self.fc2 = nn.Linear(128, 64)
        self.dropout_fc2 = nn.Dropout(0.244)
        self.output = nn.Linear(64, 1)

    def forward(self, ri, mz_intensity, species):
        ri_feat = torch.tanh(self.ri_dense(ri))
        species_feat = torch.tanh(self.species_dense(species))
        x = mz_intensity.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        steps = x.shape[1]
        attention_input = torch.cat((ri_feat, species_feat), dim=1)
        attention_weights = self.attention_sigmoid(self.attention_dense(attention_input))
        attention_weights = attention_weights.repeat(1, steps).unsqueeze(-1)
        x = x * attention_weights
        x = x.reshape(x.size(0), -1)
        merged = torch.cat((ri_feat, x, species_feat), dim=1)
        fc = torch.tanh(self.fc1(merged))
        fc = self.dropout_fc1(fc)
        fc = torch.tanh(self.fc2(fc))
        fc = self.dropout_fc2(fc)
        return self.output(fc)

# ✅ 加载模型（只加载一次）
@st.cache_resource
def load_model():
    model = AttentionCNNModel()
    model.load_state_dict(torch.load("pytorchfinal.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

# ✅ 处理上传数据
def preprocess_input(df):
    mz_cols = [col for col in df.columns if "m/z" in col]
    intensity_cols = [col for col in df.columns if "intensity" in col]
    mz_intensity_cols = mz_cols + intensity_cols

    X_mz_intensity = df[mz_intensity_cols].values.astype(float)
    X_mz_intensity = X_mz_intensity.reshape((X_mz_intensity.shape[0], 200, 1))
    X_ri = df[['RI']].values.astype(float)
    X_species = df[['species']].values.astype(float)

    return torch.tensor(X_ri, dtype=torch.float32), \
           torch.tensor(X_mz_intensity, dtype=torch.float32), \
           torch.tensor(X_species, dtype=torch.float32)

# ✅ 上传文件
uploaded_file = st.file_uploader("📤 上传填写好的 Excel 文件", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        X_ri, X_mz_intensity, X_species = preprocess_input(df)

        model = load_model()
        with torch.no_grad():
            preds = model(X_ri, X_mz_intensity, X_species).numpy().flatten()

        df["Pred_LC50"] = preds

        # 下载 Excel
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
