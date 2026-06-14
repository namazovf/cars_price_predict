import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Turbo.az Qiymət Proqnozu",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
<style>
    .main { padding: 1rem; }
    .stSelectbox label, .stNumberInput label { font-size: 14px !important; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("turboaz_xgboost.pkl")

model = load_model()

st.title("🚗 Turbo.az Qiymət Proqnozu")
st.caption("Turbo.az məlumatları əsasında XGBoost modeli ilə qiymət proqnozu")

st.divider()

col1, col2 = st.columns(2)

with col1:
    marka = st.selectbox("Marka", [
        "Toyota", "Hyundai", "BMW", "Mercedes", "Kia", "Chevrolet",
        "Volkswagen", "Nissan", "Honda", "LADA (VAZ)", "Opel", "Audi",
        "Lexus", "Ford", "Mitsubishi", "Daewoo", "Skoda", "Mazda",
        "Subaru", "Renault", "Peugeot", "Digər"
    ])

with col2:
    model_adi = st.text_input("Model", placeholder="məs: Camry, i30, 520...")

col3, col4 = st.columns(2)

with col3:
    il = st.number_input("Buraxılış ili", min_value=1990, max_value=2024, value=2018, step=1)

with col4:
    muherrik = st.number_input("Mühərrik (L)", min_value=0.5, max_value=8.0, value=1.6, step=0.1)

col5, col6 = st.columns(2)

with col5:
    yuruş = st.number_input("Yürüş (km)", min_value=0, max_value=1000000, value=100000, step=5000)

with col6:
    sahibler = st.number_input("Sahiblərin sayı", min_value=0, max_value=10, value=1, step=1)

st.divider()

col7, col8 = st.columns(2)

with col7:
    ban_novu = st.selectbox("Ban növü", [
        "Sedan", "Hetçbek", "Offroader / SUV", "Universal",
        "Liftbek", "Kupé", "Minivan", "Pikap"
    ])

with col8:
    surətler = st.selectbox("Sürətlər qutusu", ["Avtomat", "Mexaniki", "Variator", "Robot"])

col9, col10 = st.columns(2)

with col9:
    oturucu = st.selectbox("Ötürücü", ["Ön", "Arxa", "Tam"])

with col10:
    reng = st.selectbox("Rəng", [
        "Ağ", "Qara", "Gümüşü", "Boz", "Qırmızı",
        "Mavi", "Yaşıl", "Bej", "Sarı", "Narıncı", "Bənövşəyi", "Digər"
    ])

col11, col12 = st.columns(2)

with col11:
    bazar = st.selectbox("Hansı bazar üçün", [
        "Naməlum", "Amerika", "Rəsmi diler", "Avropa", "Yaponiya", "Dubay", "Gürcüstan"
    ])

with col12:
    seher = st.selectbox("Şəhər", [
        "Bakı", "Sumqayıt", "Gəncə", "Lənkəran", "Mingəçevir", "Naxçıvan", "Digər"
    ])

col13, col14 = st.columns(2)

with col13:
    qezali = st.selectbox("Qəzalı?", ["Yox", "Bəli"])

with col14:
    salon = st.selectbox("Salon?", ["Yox", "Bəli"])

st.divider()

col15, col16, col17, col18 = st.columns(4)
with col15:
    barter = st.checkbox("Barter")
with col16:
    kredit = st.checkbox("Kredit")
with col17:
    spare = st.checkbox("Ehtiyat hissə")
with col18:
    vip = st.checkbox("VIP")

st.divider()

if st.button("🔍 Qiyməti hesabla", use_container_width=True, type="primary"):
    car_age = 2024 - il
    km_per_year = yuruş / (car_age + 1)

    input_data = {
        'production_year': il,
        'engine_displacement_num': muherrik,
        'kilometrage_num': yuruş,
        'car_age': car_age,
        'km_per_year': km_per_year,
        'barter': 1 if barter else 0,
        'loan': 1 if kredit else 0,
        'salon': 1 if salon == "Bəli" else 0,
        'spare_parts': 1 if spare else 0,
        'vip': 1 if vip else 0,
        'featured': 0,
        'Ban növü': ban_novu,
        'Marka': marka,
        'Model': model_adi if model_adi else 'Naməlum',
        'Sürətlər qutusu': surətler,
        'Ötürücü': oturucu,
        'Hansı bazar üçün yığılıb': bazar,
        'Rəng': reng,
        'Sahiblər': str(sahibler),
        'Qəzalı': 'Bəli' if qezali == 'Bəli' else '',
        'Şəhər': seher
    }

    X_input = pd.DataFrame([input_data])

    with st.spinner("Hesablanır..."):
        pred = model.predict(X_input)[0]
        pred = max(0, pred)

    st.success("Proqnoz hazırdır!")

    st.metric(
        label="Proqnoz qiymət (AZN)",
        value=f"{pred:,.0f} ₼"
    )

    low = pred * 0.90
    high = pred * 1.10

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Aşağı hədd", f"{low:,.0f} ₼")
    with col_b:
        st.metric("Yuxarı hədd", f"{high:,.0f} ₼")

    st.caption(f"Model: XGBoost · R²: 0.9507 · RMSE: ~7,343 AZN")