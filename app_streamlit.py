import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set Judul Halaman
st.set_page_config(page_title="Placement Predictor", layout="wide")

# Load Artefak
@st.cache_resource
def load_artifacts():
    return {
        "clf": joblib.load('artifacts/placement_classifier.pkl'),
        "reg": joblib.load('artifacts/salary_regressor.pkl'),
        "le": joblib.load('artifacts/label_encoder.pkl'),
        "ord_enc": joblib.load('artifacts/ordinal_encoder.pkl'),
        "ohe": joblib.load('artifacts/ohe_encoder.pkl'),
        "scaler": joblib.load('artifacts/scaler.pkl')
    }

art = load_artifacts()

st.title("Student Placement & Salary Prediction")
st.write("Masukkan data akademik mahasiswa di bawah ini untuk memprediksi peluang kerja.")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
        tenth = st.number_input("10th Percentage", 0.0, 100.0, 80.0)
        twelfth = st.number_input("12th Percentage", 0.0, 100.0, 80.0)
        attendance = st.number_input("Attendance Percentage (%)", 0.0, 100.0, 80.0)
        backlogs = st.number_input("Total Backlogs", 0, 10, 0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"])

    with col2:
        coding = st.slider("Coding Skill", 1, 10, 5)
        comm = st.slider("Communication Skill", 1, 10, 5)
        apt = st.slider("Aptitude Skill", 1, 10, 5)
        certs = st.number_input("Certifications Count", 0, 20, 1)
        hacks = st.number_input("Hackathons Participated", 0, 20, 1)
        projects = st.number_input("Projects Completed", 0, 20, 2)
        internships = st.number_input("Internships Completed", 0, 5, 0)

    with col3:
        income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        extra = st.selectbox("Extracurricular", ["None", "Low", "Medium", "High"])
        stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        study_h = st.number_input("Study Hours/Day", 1.0, 15.0, 5.0)
        sleep_h = st.number_input("Sleep Hours/Day", 1.0, 12.0, 7.0)
        part_time = st.radio("Part Time Job?", ["No", "Yes"])
        internet = st.radio("Internet Access?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Result")

if submitted:

    input_df = pd.DataFrame([{
        'cgpa': cgpa, 'tenth_percentage': tenth, 'twelfth_percentage': twelfth,
        'attendance_percentage': attendance, 'backlogs': backlogs,
        'coding_skill_rating': coding, 'communication_skill_rating': comm, 'aptitude_skill_rating': apt,
        'certifications_count': certs, 'hackathons_participated': hacks,
        'projects_completed': projects, 'internships_completed': internships,
        'study_hours_per_day': study_h, 'sleep_hours': sleep_h,
        'stress_level': stress_level,
        'family_income_level': income, 'city_tier': city, 'extracurricular_involvement': extra,
        'gender': gender, 'branch': branch, 'part_time_job': part_time, 'internet_access': internet
    }])

    input_df['academic_growth'] = input_df['twelfth_percentage'] - input_df['tenth_percentage']
    input_df['total_skill_score'] = input_df['coding_skill_rating'] + input_df['communication_skill_rating'] + input_df['aptitude_skill_rating']
    input_df['total_experience'] = input_df['projects_completed'] + input_df['internships_completed']
    input_df['study_sleep_ratio'] = input_df['study_hours_per_day'] / (input_df['sleep_hours'] + 0.1)

    df_proc = input_df.copy()
    
    known_ord_cols = [col for col in art["ord_enc"].feature_names_in_ if col in df_proc.columns]
    if known_ord_cols:
        df_proc[known_ord_cols] = art["ord_enc"].transform(df_proc[known_ord_cols])
    

    ohe_cols = ['gender', 'branch', 'part_time_job', 'internet_access']
    ohe_data = art["ohe"].transform(df_proc[ohe_cols])
    df_final = pd.concat([
        df_proc.drop(columns=ohe_cols), 
        pd.DataFrame(ohe_data, columns=art["ohe"].get_feature_names_out())
    ], axis=1)
    
    df_final = df_final[art["scaler"].feature_names_in_]
    
    mapping_manual = {
        'Low': 0, 'Medium': 1, 'High': 2, 'None': 0, 
        'Tier 3': 0, 'Tier 2': 1, 'Tier 1': 2
    }
    for col in df_final.columns:
        if df_final[col].dtype == 'object': 
            df_final[col] = df_final[col].replace(mapping_manual)
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
    
    scaled_data = art["scaler"].transform(df_final)

    status_enc = art["clf"].predict(scaled_data)[0]
    status = art["le"].inverse_transform([status_enc])[0]
    
    st.divider()
    if status == 'Placed':
        salary = art["reg"].predict(scaled_data)[0]
        st.success(f"### Status: {status}")
        st.metric("Estimated Salary (LPA)", f"{salary:.2f}")
    else:
        st.error(f"### Status: {status}")
        st.info("Saran: Tingkatkan skill coding dan perbanyak project portofolio.")
