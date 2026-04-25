import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ TITLE ------------------
st.title("🎯 AI Placement Prediction Dashboard")
st.markdown("---")

# ------------------ INPUT SECTION ------------------
st.header("Enter Student Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
internships = st.number_input("Internships", 0, 5, 1)
projects = st.number_input("Projects", 0, 10, 2)
skills = st.slider("Skills Score", 0, 10, 5)

st.markdown("---")

# ------------------ PREDICTION ------------------
if st.button("Predict Placement"):

    data = np.array([[cgpa, internships, projects, skills]])

    prediction = model.predict(data)
    prob = model.predict_proba(data)

    st.subheader("Prediction Result 🎯")

    if prediction[0] == 1:
        st.success(f"Placed ✅ (Chance: {prob[0][1]*100:.2f}%)")
    else:
        st.error(f"Not Placed ❌ (Chance: {prob[0][1]*100:.2f}%)")

    # ------------------ SUGGESTIONS ------------------
    st.subheader("Suggestions 💡")

    if cgpa < 7:
        st.warning("Improve your CGPA to increase chances")

    if internships == 0:
        st.warning("Try to get at least 1 internship")

    if skills < 5:
        st.warning("Work on your technical skills")

st.markdown("---")

# ------------------ MODEL ACCURACY ------------------
st.subheader("Model Accuracy 📊")
st.write("Accuracy: ~80% (based on test data)")

st.markdown("---")

# ------------------ DATA VISUALIZATION ------------------
st.subheader("CGPA vs Placement 📈")

df = pd.read_csv("placement.csv")

fig, ax = plt.subplots()
ax.scatter(df['cgpa'], df['placed'])
ax.set_xlabel("CGPA")
ax.set_ylabel("Placed (0 = No, 1 = Yes)")

st.pyplot(fig)

st.markdown("---")

# ------------------ FEATURE IMPORTANCE ------------------
st.subheader("Feature Importance 🔍")

importance = model.coef_[0]
features = ['CGPA', 'Internships', 'Projects', 'Skills']

for i in range(len(features)):
    st.write(f"{features[i]}: {importance[i]:.2f}")