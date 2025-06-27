import streamlit as st
import joblib
import numpy as np
import os

# Page config
st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Success Predictor")
st.markdown("Predict if a student will succeed based on academic and personal information.")

# === Sidebar Model Selection ===
st.sidebar.title("⚙️ Model Selection")

model_options = {
    "KNN Classifier": "knn_model.pkl",
    "Decision Tree": "tree_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Compare ALL Models": None
}

selected_model_name = st.sidebar.selectbox("Choose ML Model", list(model_options.keys()))

# Load model(s)
models = {}
if selected_model_name == "Compare ALL Models":
    for name, path in model_options.items():
        if path and os.path.exists(path):
            models[name] = joblib.load(path)
else:
    selected_model_path = model_options[selected_model_name]
    if not os.path.exists(selected_model_path):
        st.sidebar.error(f"❌ Model file '{selected_model_path}' not found!")
        st.stop()
    models[selected_model_name] = joblib.load(selected_model_path)
    st.sidebar.success(f"✅ {selected_model_name} loaded!")

# === Label Mappings ===
marital_status_dict = {"Single": 1, "Married": 2}
attendance_dict = {"Daytime": 1, "Evening": 0}
gender_dict = {"Female": 0, "Male": 1}
yes_no_dict = {"No": 0, "Yes": 1}

pca_mapping = {
    "🌟 Excellent": -20.0,
    "✅ Good": -10.0,
    "🟡 Average": 0.0,
    "⚠️ Poor": 3.0,
    "🚨 Very Poor": 6.0
}

application_mode_dict = {
    "General Admission": 1,
    "Over 23 Years Old": 2,
    "International Student": 3,
    "Change of Institution/Course": 4,
    "Technological Specialization": 5,
    "Vocational Course": 6,
    "Degree Holder": 7,
    "Re-enrollment": 8,
    "Transfer": 9,
    "Change of Study Cycle": 10,
    "Disability Access": 11,
    "Sport Practitioners": 12,
    "Military Personnel": 13,
    "Diplomats' Children": 14,
    "Distance Learning": 15,
    "Special Admission Test": 16,
    "Other/Unknown": 17
}

previous_qualification_dict = {
    "Unknown": 1,
    "Basic Education (10th Grade)": 2,
    "High School Diploma (12th Grade)": 3,
    "Vocational/Technical High School": 4,
    "Undergraduate Degree": 5,
    "Bachelor’s Degree (or equivalent)": 6,
    "Master’s Degree or Higher": 7
}

course_dict = {
    "Biofuel Production Technologies": 1,
    "Animation and Multimedia Design": 2,
    "Social Service (evening attendance)": 3,
    "Agronomy": 4,
    "Communication Design": 5,
    "Veterinary Nursing": 6,
    "Informatics Engineering": 7,
    "Equiniculture": 8,
    "Management": 9,
    "Social Service": 10,
    "Tourism": 11,
    "Nursing": 12,
    "Oral Hygiene": 13,
    "Advertising and Marketing Management": 14,
    "Journalism and Communication": 15,
    "Basic Education": 16,
    "Management (evening attendance)": 17
}

mother_qualification_dict = {
    "Unknown": 1, "Basic 4th grade": 2, "Basic 6th grade": 3, "Basic 10th grade": 4,
    "Secondary Education 12th": 5, "Higher Education - Bachelor's": 6, "Master's": 7,
    "PhD": 8, "Professional Training": 9, "Technical Course": 10,
    "10th Grade": 11, "11th Grade Dropout": 12, "12th Grade Completed": 13,
    "Diploma Program": 14, "Associate Degree": 15, "Undergraduate Dropout": 16,
    "Undergraduate Complete": 17, "Graduate Incomplete": 18,
    "Graduate Complete": 19, "Others": 20, "No Schooling": 21,
    "Not Declared": 22, "Illiterate": 23
}

father_occupation_dict = {
    "Unemployed": 1, "Manual Labor": 2, "Skilled Worker": 3,
    "Business Owner": 4, "Technician": 5, "Clerical": 6,
    "Teacher": 7, "Engineer": 8, "Doctor": 9,
    "Lawyer": 10, "Other": 11
}

# === UI Inputs ===
with st.expander("👤 Personal Information"):
    col1, col2 = st.columns(2)
    marital_status = col1.selectbox("Marital Status", list(marital_status_dict.keys()))
    age = col2.slider("Age at Enrollment", 17, 50, 20)
    gender = col1.selectbox("Gender", list(gender_dict.keys()))
    international = col2.selectbox("International Student", list(yes_no_dict.keys()))

with st.expander("📄 Application & Course Details"):
    col1, col2 = st.columns(2)
    application_mode = col1.selectbox("Application Mode", list(application_mode_dict.keys()))
    application_order = col2.slider("Application Order (Preference)", 1, 9, 1)
    course = col1.selectbox("Course", list(course_dict.keys()))
    attendance = col2.selectbox("Attendance Type", list(attendance_dict.keys()))

with st.expander("🎓 Academic & Family Background"):
    col1, col2 = st.columns(2)
    previous_qualification = col1.selectbox("Previous Qualification", list(previous_qualification_dict.keys()))
    mother_qualification = col2.selectbox("Mother's Qualification", list(mother_qualification_dict.keys()))
    father_occupation = col1.selectbox("Father's Occupation", list(father_occupation_dict.keys()))

with st.expander("💼 Financial & Support Status"):
    col1, col2 = st.columns(2)
    displaced = col1.selectbox("Displaced Student", list(yes_no_dict.keys()))
    special_needs = col2.selectbox("Special Educational Needs", list(yes_no_dict.keys()))
    debtor = col1.selectbox("Currently in Debt", list(yes_no_dict.keys()))
    fees_up_to_date = col2.selectbox("Tuition Fees Paid", list(yes_no_dict.keys()))
    scholarship = col1.selectbox("Scholarship Holder", list(yes_no_dict.keys()))

with st.expander("📊 Academic Performance"):
    pca_label = st.selectbox(
        "Performance in 1st & 2nd Semesters",
        options=list(pca_mapping.keys()),
        help="Select the student's academic performance level for the first two semesters."
    )
    
    pca_value = pca_mapping[pca_label]

# === Input Features ===
input_features = np.array([[ 
    marital_status_dict[marital_status],
    application_mode_dict[application_mode],
    application_order,
    course_dict[course],
    attendance_dict[attendance],
    previous_qualification_dict[previous_qualification],
    mother_qualification_dict[mother_qualification],
    father_occupation_dict[father_occupation],
    yes_no_dict[displaced],
    yes_no_dict[special_needs],
    yes_no_dict[debtor],
    yes_no_dict[fees_up_to_date],
    gender_dict[gender],
    yes_no_dict[scholarship],
    age,
    yes_no_dict[international],
    pca_value
]])

# === Prediction ===
st.markdown("---")
if st.button("🔍 Predict Student Outcome"):
    result = ""
    emoji = ""
    color = ""

    if selected_model_name == "Compare ALL Models":
        votes = [model.predict(input_features)[0] for model in models.values()]
        success_votes = votes.count(1)
        fail_votes = votes.count(0)

        if fail_votes >= 2:
            result = "❌ Student is at risk of dropping or failing."
            emoji = "😞"
            color = "red"
            st.error(f"❌ Majority Vote: At Risk! (❌: {fail_votes}, ✅: {success_votes})")
            st.snow()
        else:
            result = "✅ Student is likely to succeed!"
            emoji = "🎉"
            color = "green"
            st.success(f"✅ Majority Vote: Likely to Succeed! (✅: {success_votes}, ❌: {fail_votes})")
            st.balloons()
    else:
        for name, model in models.items():
            prediction = model.predict(input_features)
            if prediction[0]:
                result = f"✅ {name} Prediction: Student is likely to succeed!"
                emoji = "🎉"
                color = "green"
                st.success(result)
                st.balloons()
            else:
                result = f"❌ {name} Prediction: Student is at risk of failing."
                emoji = "😞"
                color = "red"
                st.error(result)
                st.snow()

    # Final Card
    st.markdown("---")
    st.markdown(f"""
        <div style="background-color:{color}; padding: 15px; border-radius: 10px; text-align:center;">
            <h2 style="color:white;">{emoji} {result}</h2>
        </div>
    """, unsafe_allow_html=True)
