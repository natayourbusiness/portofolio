import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# --- KONFIGURASI HALAMAN (Hanya boleh dipanggil 1 kali di paling atas) ---
st.set_page_config(
    page_title="Data Portfolio | HR Data & Operations Analyst",
    page_icon="📊",
    layout="wide"
)

# --- STYLE CSS KUSTOM ---
st.markdown("""
    <style>
    .main {background-color: #f5f7f9;}
    .metric-container { background-color: #FFFFFF; padding: 20px; border-radius: 8px; border: 1px solid #E5E7EB; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .metric-red { border-bottom: 4px solid #DC2626; }
    .insight-box { background-color: #FEF2F2; border-left: 4px solid #DC2626; padding: 15px; color: #7F1D1D; font-weight: 500; margin-top: 10px; }
    .action-box { background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 15px; color: #14532D; font-weight: 500; margin-top: 10px; }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Navigasi Portofolio")
page = st.sidebar.radio("Pilih Halaman:", [
    "Profil Profesional", 
    "Proyek 1: Optimasi Tenaga Kerja & Reduksi Biaya Lembur",
    "Proyek 2: Executive Churn Intelligence",
    "Proyek 3: Customer Value Segmentation (RFM)"
])

st.sidebar.divider()
st.sidebar.info("""
**Kontak & Tautan:**
- 📧 donnypranata45@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/donny-pranata/)
- 💻 [WhatsApp / Mobile Phone](https://wa.me/6285796135624)
""")

# ==========================================
# HALAMAN 1: PROFIL PROFESIONAL
# ==========================================
if page == "Profil Profesional":
    st.title("🚀 Data-Driven Operations Professional")
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.image(r"C:\Users\lenovo\Downloads\X\porto\Fotoaman.jpg", use_container_width=True)
        st.success("📍 Domisili Saat Ini: Jawa Tengah")

    with col2:
        st.subheader("Tentang Saya")
        st.write("""
        Sebagai lulusan Teknik Informatika (S.Kom), saya tidak menempuh jalur konvensional sebagai *Software Developer*. 
        Saya terjun langsung ke lantai operasional sebagai **Administrator Sistem & Data** untuk menjembatani 
        kesenjangan antara teknologi dan eksekusi bisnis.
        
        Saat ini, saya mengelola integritas *database* karyawan, arsitektur data *payroll*, dan pelaporan analitik 
        di PT. Atarindo Prima Internusa. Keunggulan saya terletak pada kemampuan menggabungkan logika *programming* (SQL/Python) dengan ketajaman manajerial untuk memastikan efisiensi dan kerahasiaan data tingkat tinggi.
        """)

        st.markdown("""
       **Keahlian Inti:**
        - **Data Management:** Pemrosesan data kehadiran & *overtime*, validasi data *payroll*, integrasi ERP.
        - **Analytics & Reporting:** Pembuatan laporan *headcount* bulanan dan analisis *turnover* (Excel Pivot, VLOOKUP).
        - **Technical Stack:** SQL, Python, & Advanced Excel.
        """)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pencapaian Operasional", "Efisiensi +20%", delta="Target Tahunan")
        with c2:
            st.metric("Data Processing", "2,5k+ Baris/Hari", delta="Automated")
        with c3:
            st.metric("Data Confidentiality", "100%", delta="BPJS & PPh 21 Aman")
        with c4:
            st.metric("SLA Pelaporan Bulanan", "On-Time", delta="Turnover & Headcount")

# ==========================================
# HALAMAN 2: PROYEK 1 - WORKFORCE OPTIMIZATION
# ==========================================
elif page == "Proyek 1: Optimasi KUR & NPL": # Ubah nama menu di sidebar Anda nanti menjadi "Proyek 1: Workforce Optimization"
    st.markdown("<h1 style='text-align: center;'>⏱️ Optimasi Tenaga Kerja & Reduksi Biaya Lembur</h1>", unsafe_allow_html=True)
    st.caption("<p style='text-align: center;'>Studi Kasus: Prediksi Beban Kerja Cabang & Penjadwalan Staf Otomatis</p>", unsafe_allow_html=True)
    
    # --- ALUR KERJA PROYEK 1 ---
    with st.expander("🔍 Lihat Arsitektur Solusi (Workforce Pipeline)", expanded=True):
        col_flow1, col_flow2, col_flow3 = st.columns(3)
        with col_flow1:
            st.markdown("**1. Workload Forecasting**")
            st.caption("Menggunakan model Time-Series (Prophet/ARIMA) untuk memprediksi volume transaksi harian berdasarkan pola historis (akhir bulan, hari gajian, dll).")
        with col_flow2:
            st.markdown("**2. Shift Optimization**")
            st.caption("Menerapkan Linear Programming (PuLP) untuk menyusun jadwal shift yang meminimalkan *idle time* staf saat sepi dan mencegah *understaffing* saat ramai.")
        with col_flow3:
            st.markdown("**3. Cost Reduction**")
            st.caption("Mengonversi efisiensi jadwal menjadi metrik finansial: menekan biaya *overtime* tanpa mengorbankan Service Level Agreement (SLA) nasabah.")
    
    st.divider()

    # --- DATA MOCKUP (Simulasi) ---
    @st.cache_data
    def load_workforce_data():
        np.random.seed(42)
        days = pd.date_range(start="2026-03-01", periods=30)
        # Asumsi: Transaksi melonjak di awal dan akhir bulan
        base_demand = np.array([120 if d.day < 5 or d.day > 25 else 60 for d in days]) 
        noise = np.random.normal(0, 10, 30)
        demand = np.maximum(base_demand + noise, 0)
        
        return pd.DataFrame({
            'Tanggal': days,
            'Prediksi_Transaksi': demand,
            'Staf_Dibutuhkan': np.ceil(demand / 15) # Asumsi 1 staf menangani 15 transaksi
        })
    
    wf_data = load_workforce_data()

    # --- INTERACTIVE SIMULATOR ---
    st.subheader("🎛️ Simulator Kebijakan Operasional")
    st.write("Ubah parameter di bawah untuk melihat bagaimana Service Level (SLA) berdampak pada anggaran lembur bulanan.")
    
    c_sim1, c_sim2 = st.columns([1, 2])
    
    with c_sim1:
        sla_target = st.slider("Target Service Level (%)", 80, 100, 95, help="Persentase nasabah yang dilayani tepat waktu")
        staf_buffer = st.number_input("Buffer Staf Cadangan", min_value=0, max_value=5, value=1)
        
        # Logika Bisnis: Semakin tinggi SLA, semakin banyak staf & lembur yang dibutuhkan
        base_overtime_cost = 45000000 # Rp 45 Juta (kondisi tidak optimal)
        efficiency_multiplier = (100 - sla_target) / 100
        optimized_overtime = base_overtime_cost * (0.4 + efficiency_multiplier) + (staf_buffer * 2000000)
        penghematan = base_overtime_cost - optimized_overtime

    with c_sim2:
        k1, k2, k3 = st.columns(3)
        k1.metric("Proyeksi Biaya Lembur", f"Rp {optimized_overtime/1e6:.1f} Jt", delta=f"-Rp {penghematan/1e6:.1f} Jt", delta_color="inverse")
        k2.metric("Reduksi Idle Time", "24%", delta="Optimal")
        k3.metric("Pemenuhan SLA", f"{sla_target}%", delta="Sesuai Target")
        
        st.info(f"💡 **Insight:** Dengan target SLA {sla_target}%, model Linear Programming merekomendasikan efisiensi yang menghemat anggaran sebesar **Rp {penghematan:,.0f}** bulan ini dibandingkan penjadwalan manual.")

    st.divider()

    # --- VISUALISASI ---
    st.subheader("📈 Prediksi Beban Kerja vs Alokasi Staf")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Line chart untuk Prediksi Transaksi
    fig.add_trace(go.Scatter(x=wf_data['Tanggal'], y=wf_data['Prediksi_Transaksi'], 
                             name="Prediksi Transaksi", line=dict(color='blue', width=2)), 
                  secondary_y=False)
    
    # Bar chart untuk Alokasi Staf
    alokasi_staf = wf_data['Staf_Dibutuhkan'] + staf_buffer + (sla_target - 80)/10 # Penyesuaian dinamis
    fig.add_trace(go.Bar(x=wf_data['Tanggal'], y=alokasi_staf, 
                         name="Alokasi Staf Otomatis", marker_color='orange', opacity=0.6), 
                  secondary_y=True)
    
    fig.update_layout(title="Penyesuaian Staf Dinamis Menghadapi Lonjakan Akhir Bulan",
                      xaxis_title="Tanggal", hovermode="x unified")
    fig.update_yaxes(title_text="Volume Transaksi", secondary_y=False)
    fig.update_yaxes(title_text="Jumlah Staf Bertugas", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# HALAMAN 3: EXECUTIVE CHURN INTELLIGENCE
# ==========================================
elif page == "Proyek 2: Executive Churn Intelligence":
    
    # --- ML Engine ---
    @st.cache_data
    def load_churn_data():
        file_path = 'Customer-Churn-Records.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Fallback jika file tidak ada: Generate Dummy Data agar aplikasi tidak crash
            st.warning("⚠️ File 'Customer-Churn-Records.csv' tidak ditemukan di direktori. Menggunakan data simulasi.")
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'CreditScore': np.random.randint(400, 850, n_samples),
                'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
                'Gender': np.random.choice(['Male', 'Female'], n_samples),
                'Age': np.random.randint(18, 80, n_samples),
                'Tenure': np.random.randint(0, 11, n_samples),
                'Balance': np.random.uniform(0, 200000, n_samples),
                'NumOfProducts': np.random.randint(1, 5, n_samples),
                'HasCrCard': np.random.choice([0, 1], n_samples),
                'IsActiveMember': np.random.choice([0, 1], n_samples),
                'EstimatedSalary': np.random.uniform(20000, 200000, n_samples),
                'Satisfaction Score': np.random.randint(1, 6, n_samples),
                'Card Type': np.random.choice(['DIAMOND', 'GOLD', 'SILVER', 'PLATINUM'], n_samples),
                'Point Earned': np.random.randint(0, 1000, n_samples),
                'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            })
            
        df['Churn_Label'] = df['Exited'].map({1: 'Churn', 0: 'Retained'})
        return df

    @st.cache_resource
    def train_churn_model(df):
        features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                    'Satisfaction Score', 'Card Type', 'Point Earned']
        
        X = df[features].copy()
        y = df['Exited']
        
        encoders = {}
        for col in ['Geography', 'Gender', 'Card Type']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        return model, encoders, importance, features

    df_churn = load_churn_data()
    model_churn, encoders_churn, importance_churn, features_list_churn = train_churn_model(df_churn)

    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>CUSTOMER ATTRITION COMMAND CENTER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.2rem; margin-bottom: 30px;'>Q3 Retentions, Root Cause Diagnostics, & Predictive Warning System</p>", unsafe_allow_html=True)

    tab1_c, tab2_c, tab3_c = st.tabs(["📑 1. Executive Briefing (Realitas)", "🧬 2. Root Cause Engine (Diagnosa)", "🔮 3. Early Warning System (Prediksi)"])

    with tab1_c:
        total_customers = len(df_churn)
        churn_rate = df_churn['Exited'].mean() * 100
        balance_lost = pd.to_numeric(df_churn[df_churn['Exited']==1]['Balance'], errors='coerce').sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <p style="color: #6B7280; margin:0; font-size:14px;">Total Pelanggan Aktif</p>
                <h2 style="margin:0; font-size:32px;">{total_customers:,}</h2>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-container metric-red">
                <p style="color: #6B7280; margin:0; font-size:14px;">Tingkat Churn (Attrition Rate)</p>
                <h2 style="margin:0; font-size:32px; color: #DC2626;">{churn_rate:.1f}%</h2>
                <p style="color: #DC2626; margin:0; font-size:12px; font-weight:bold;">▲ Kritis (Standar: < 10%)</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <p style="color: #6B7280; margin:0; font-size:14px;">Capital Outflow (Aset Hilang)</p>
                <h2 style="margin:0; font-size:32px;">${balance_lost/1e6:.1f} Juta</h2>
            </div>""", unsafe_allow_html=True)

        st.write("---")
        
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            df_churn['Age_Group'] = pd.cut(df_churn['Age'], bins=[18, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '50+'])
            age_churn = df_churn.groupby('Age_Group', observed=False)['Exited'].mean().reset_index()
            age_churn['Exited'] = age_churn['Exited'] * 100
            colors = ['#E5E7EB', '#E5E7EB', '#FCA5A5', '#DC2626'] 
            fig_age = px.bar(age_churn, x='Age_Group', y='Exited', text=age_churn['Exited'].apply(lambda x: f'{x:.1f}%'),
                             title="Kebocoran Retensi Berdasarkan Demografi Usia", template="plotly_white")
            fig_age.update_traces(marker_color=colors, textposition='outside', textfont=dict(size=14, color='black'))
            fig_age.update_layout(yaxis_title="Probabilitas Churn (%)", xaxis_title="Rentang Usia", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True)

        with col_b:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("#### 🚨 Temuan Tingkat Eksekutif")
            st.markdown("Model operasional saat ini beracun bagi demografi senior. Pelanggan di atas usia 50 tahun memiliki tingkat churn **44%**.")
            st.markdown("""
            <div class="insight-box">
                <strong>Tindakan Segera:</strong> Audit UI/UX aplikasi untuk lansia dan evaluasi produk tabungan pensiun yang ditawarkan kompetitor.
            </div>
            """, unsafe_allow_html=True)

    with tab2_c:
        st.markdown("### 🧬 Analisis Driver Utama (Tanpa Bias)")
        st.write("Algoritma Random Forest membedah variabel mana yang secara matematis mendorong pelanggan keluar, mengabaikan opini manusia.")
        
        fig_imp = px.bar(importance_churn, x='Importance', y='Feature', orientation='h',
                         title="Bobot Pengaruh terhadap Keputusan Churn",
                         template="plotly_white", color='Importance', color_continuous_scale='Reds')
        fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>Realitas Pahit:</strong> 'Age' dan 'NumOfProducts' mendominasi. Ini berarti produk kita salah sasaran secara demografis, atau pelanggan yang menggunakan banyak produk justru mengalami gesekan layanan yang paling tinggi.
        </div>
        """, unsafe_allow_html=True)

    with tab3_c:
        st.markdown("### 🛡️ Simulasi Intervensi Pelanggan Baru")
        st.write("Jangan tunggu sampai mereka mengeluh. Masukkan profil pelanggan dan ketahui probabilitas mereka pergi bulan depan.")
        
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                p_age = st.number_input("Usia", 18, 100, 55)
                p_geo = st.selectbox("Negara", encoders_churn['Geography'].classes_)
                p_gender = st.selectbox("Gender", encoders_churn['Gender'].classes_)
                p_tenure = st.number_input("Tenure (Tahun)", 0, 10, 5)
            with c2:
                p_balance = st.number_input("Saldo ($)", 0.0, 500000.0, 120000.0)
                p_products = st.number_input("Jumlah Produk", 1, 4, 3)
                p_active = st.selectbox("Member Aktif?", [1, 0], index=0)
                p_salary = st.number_input("Estimasi Gaji ($)", 0.0, 500000.0, 90000.0)
            with c3:
                p_credit = st.number_input("Credit Score", 300, 850, 650)
                p_card = st.selectbox("Punya Kartu Kredit?", [1, 0], index=1)
                p_cardtype = st.selectbox("Tipe Kartu", encoders_churn['Card Type'].classes_)
                p_satisfaction = st.slider("Skor Kepuasan", 1, 5, 2)
                p_points = st.number_input("Poin Terkumpul", 0, 1000, 300)
                
            submit = st.form_submit_button("Hitung Risiko Attrition")

        if submit:
            input_data = pd.DataFrame([[
                p_credit, p_geo, p_gender, p_age, p_tenure, p_balance, 
                p_products, p_card, p_active, p_salary, p_satisfaction, p_cardtype, p_points
            ]], columns=features_list_churn)
            
            for col in ['Geography', 'Gender', 'Card Type']:
                input_data[col] = encoders_churn[col].transform(input_data[col])
                
            prob = model_churn.predict_proba(input_data)[0][1] * 100
            
            st.markdown("---")
            if prob >= 50:
                st.markdown(f"""
                <div class="insight-box" style="font-size: 18px;">
                    ⚠️ <strong>RISIKO FATAL: Probabilitas Churn {prob:.1f}%</strong><br>
                    Pelanggan ini sedang mencari jalan keluar. Intervensi segera diperlukan. Jangan kirim email otomatis; tugaskan Account Manager untuk menelepon hari ini.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="action-box" style="font-size: 18px;">
                    ✅ <strong>AMAN: Probabilitas Churn {prob:.1f}%</strong><br>
                    Aset stabil. Pelanggan ini tidak memiliki intensi keluar dalam waktu dekat. Fokuskan sumber daya pada metrik lain.
                </div>""", unsafe_allow_html=True)
# ==========================================
# HALAMAN 4: CLUSTERING (Based on clustering.py)
# ==========================================
elif page == "Proyek 3: Customer Value Segmentation (RFM)":
    st.markdown("<h1 style='text-align: center;'>🏗️ Arsitektur Data: Dari Transaksi ke Strategi</h1>", unsafe_allow_html=True)
    
    # --- BAGIAN BARU: PENJELASAN ALUR ---
    with st.expander("🔍 Lihat Alur Kerja Teknis (Pipeline Logics)", expanded=True):
        col_flow1, col_flow2, col_flow3 = st.columns(3)
        with col_flow1:
            st.markdown("**1. Data Cleansing**")
            st.caption("Menghapus Invoice 'C' (Cancel) dan menangani Outlier Harga/Jumlah menggunakan metode IQR agar model tidak terdistorsi oleh data ekstrem.")
        with col_flow2:
            st.markdown("**2. RFM Processing**")
            st.caption("Mengubah ribuan baris transaksi menjadi 3 metrik kunci per pelanggan: Kapan terakhir belanja? Seberapa sering? Berapa total uangnya?")
        with col_flow3:
            st.markdown("**3. K-Means Optimization**")
            st.caption("Menggunakan algoritma K-Means untuk mencari pola tersembunyi dan mengelompokkan pelanggan secara matematis tanpa bias manusia.")

    st.divider()
    
    # --- BAGIAN BARU: TRANSFORMASI DATA (MENTAH VS BERSIH) ---
    st.subheader("🔬 Transparansi Logika: Menjinakkan Data Mentah")
    st.write("""
    Algoritma secanggih apa pun akan gagal jika diberi data sampah (*Garbage In, Garbage Out*). 
    Berikut adalah simulasi bagaimana saya mengubah jutaan baris struk belanja kotor menjadi metrik perilaku yang siap dianalisis.
    """)

    tab_mentah, tab_bersih = st.tabs(["❌ 1. Data Mentah (Kotor & Bising)", "✅ 2. Data RFM (Siap Analisis)"])

    with tab_mentah:
        st.markdown("**Struk Belanja Digital (Simulasi Point of Sale)**")
        raw_df = pd.DataFrame({
            'InvoiceNo': ['536365', '536366', 'C536379', '536381', '536382'],
            'StockCode': ['85123A', '22633', 'D', '22752', '10002'],
            'Description': ['WHITE HANGING HEART', 'HAND WARMER', 'Discount', 'NESTING BOXES', 'POLITICAL GLOBE'],
            'Quantity': [6, 6, -1, 1000, 12],
            'UnitPrice': [2.55, 1.85, 27.50, 7.65, 0.85],
            'CustomerID': ['17850', '17850', '14527', '15311', 'NaN']
        })
        st.dataframe(raw_df, use_container_width=True)
        st.error("""
        **Analisis Anomali:** 1. Baris ke-3 adalah *Refund* (Invoice 'C' dan Kuantitas negatif).
        2. Baris ke-4 memiliki Outlier kuantitas ekstrem (1000 unit).
        3. Baris ke-5 tidak memiliki CustomerID (Transaksi anonim).
        Memasukkan data ini mentah-mentah ke K-Means adalah sebuah malapraktik analitik.
        """)

    with tab_bersih:
        st.markdown("**Profil Nasabah (Agregasi Recency, Frequency, Monetary)**")
        clean_df = pd.DataFrame({
            'CustomerID': ['17850', '14527', '15311'],
            'Recency (Hari)': [302, 2, 0],
            'Frequency (Trx)': [34, 55, 91],
            'Monetary ($)': [5391.21, 8508.34, 15000.50]
        })
        st.dataframe(clean_df, use_container_width=True)
        st.success("""
        **Penyelesaian:** Anomali dihapus, outlier dipangkas menggunakan IQR, dan data diagregasi berdasarkan ID Pelanggan. 
        Kini, algoritma memiliki fondasi matematika yang valid untuk mulai mengelompokkan pelanggan.
        """)
        
    st.divider()

    @st.cache_data
    def load_rfm_data():
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            'Recency': np.random.randint(1, 365, n),
            'Frequency': np.random.randint(1, 50, n),
            'Monetary': np.random.uniform(100, 10000, n),
            'Interpurchase_Time': np.random.randint(1, 60, n)
        }, index=range(1, n+1))

    rfm_data = load_rfm_data()
    
    # Engine Clustering
    scaler = MinMaxScaler()
    rfm_norm = scaler.fit_transform(rfm_data)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(rfm_norm) # K=3 sebagai default optimal
    rfm_data['Cluster'] = kmeans.labels_

    # --- PENERJEMAHAN BISNIS ---
    summary = rfm_data.groupby('Cluster').mean().round(2)
    summary['Count'] = rfm_data.groupby('Cluster').size()
    
    # Identifikasi Segmen secara Otomatis
    vip_idx = summary['Monetary'].idxmax()
    at_risk_idx = summary['Recency'].idxmax()
    loyal_idx = [i for i in summary.index if i not in [vip_idx, at_risk_idx]][0]

    def name_segment(cluster):
        if cluster == vip_idx: return "👑 THE CHAMPIONS (VIP)"
        if cluster == at_risk_idx: return "⚠️ AT RISK (Hampir Hilang)"
        return "📈 POTENTIAL LOYALISTS"

    rfm_data['Segment'] = rfm_data['Cluster'].apply(name_segment)
    
    # --- TAMPILAN DASHBOARD ---
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Ringkasan Segmen")
        for seg in rfm_data['Segment'].unique():
            count = len(rfm_data[rfm_data['Segment'] == seg])
            avg_spend = rfm_data[rfm_data['Segment'] == seg]['Monetary'].mean()
            st.metric(seg, f"{count} Org", f"Avg: ${avg_spend:,.0f}")

    with col_b:
        fig_3d = px.scatter_3d(rfm_data, x='Recency', y='Frequency', z='Monetary', 
                                color='Segment', title="Peta Kekuatan Nasabah",
                                color_discrete_map={
                                    "👑 THE CHAMPIONS (VIP)": "gold",
                                    "⚠️ AT RISK (Hampir Hilang)": "red",
                                    "📈 POTENTIAL LOYALISTS": "blue"
                                })
        st.plotly_chart(fig_3d, use_container_width=True)

    st.subheader("📋 Interpretasi & Rencana Aksi Bisnis")
    st.write("Setelah alur teknis selesai, inilah hasil yang digunakan oleh Tim Pemasaran untuk mengambil keputusan:")
    
    # Menggunakan Container untuk Visualisasi Strategi
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.success("**THE CHAMPIONS**")
        st.write("""
        - **Target:** Pertahankan Eksklusivitas.
        - **Aksi:** Berikan akses 'First Pick' produk baru dan layanan Priority Lounge.
        - **Tujuan:** Meningkatkan Advocacy (Word of Mouth).
        """)
        
    with c2:
        st.info("**POTENTIAL LOYALISTS**")
        st.write("""
        - **Target:** Tingkatkan Frekuensi Belanja.
        - **Aksi:** Program 'Loyalty Points' dan promo bundling produk.
        - **Tujuan:** Menggeser mereka menjadi VIP.
        """)

    with c3:
        st.error("**AT RISK CUSTOMERS**")
        st.write("""
        - **Target:** Re-aktivasi Segera.
        - **Aksi:** Kirimkan kupon 'We Miss You' dengan diskon signifikan (20-30%).
        - **Tujuan:** Mencegah mereka pindah ke kompetitor.
        """)