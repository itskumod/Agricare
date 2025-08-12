import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(page_title="AyurSage 360", layout="wide")
st.title("🌿 AyurSage 360: AI-Powered Vedic-Ayurvedic Wellness")

# Sidebar navigation
st.sidebar.title("🔍 Diagnostic Sections")
options = [
    "Home",
    "Dosha Quiz",
    "Symptom Analyzer",
    "Eye Health (Upload Image)",
    "PCOD & Women's Health",
    "Hair/Skin/Body Care",
    "Pre-Diabetes Detector",
    "Ayurvedic Q&A"
]
selection = st.sidebar.radio("Choose a section:", options)

# Home Page
if selection == "Home":
    st.markdown("""
    ### Welcome to AyurSage 360
    This platform combines Ayurvedic principles with AI to offer personalized health insights. 
    Select a module from the sidebar to begin your wellness journey.
    """)

# Dosha Quiz
elif selection == "Dosha Quiz":
    st.header("🧬 Dosha Type Identification Quiz")
    q1 = st.radio("How would you describe your body type?", ["Thin/Light", "Medium/Muscular", "Heavy/Sturdy"])
    q2 = st.radio("What is your skin nature?", ["Dry", "Sensitive/Reddish", "Oily/Thick"])
    q3 = st.radio("How is your digestion?", ["Irregular", "Strong", "Slow"])

    if st.button("Get Dosha Type"):
        if q1 == "Thin/Light" and q2 == "Dry":
            st.success("You are likely Vata dominant")
        elif q2 == "Sensitive/Reddish" and q3 == "Strong":
            st.success("You are likely Pitta dominant")
        elif q1 == "Heavy/Sturdy" and q3 == "Slow":
            st.success("You are likely Kapha dominant")
        else:
            st.info("You may have a dual Dosha constitution")

# Symptom Analyzer
elif selection == "Symptom Analyzer":
    st.header("💬 Symptom-Based Diagnosis")
    symptoms = st.text_area("Enter your symptoms (comma-separated):")
    if st.button("Analyze Symptoms"):
        if "hair fall" in symptoms.lower():
            st.write("🔍 Detected: Hair fall. Possible Vata/Pitta imbalance.")
            st.write("💡 Suggestion: Use Bhringraj oil, avoid late nights, and practice head massage.")
        elif "irregular periods" in symptoms.lower():
            st.write("🔍 Detected: PCOD indicators.")
            st.write("💡 Suggestion: Use Shatavari, regular yoga, and avoid spicy food.")
        else:
            st.warning("No matching Ayurvedic condition found. Please try another symptom.")

# Eye Health
elif selection == "Eye Health (Upload Image)":
    st.header("👁️ Eye Disease Detection (Prototype)")
    uploaded_file = st.file_uploader("Upload your eye image")
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Eye Image", use_column_width=True)
        st.success("Triphala and Netra Tarpan may help if you're experiencing redness or dryness.")

# PCOD & Women's Health
elif selection == "PCOD & Women's Health":
    st.header("👩‍⚕️ Women's Health Diagnostic")
    symptoms = st.multiselect("Select symptoms:", ["Irregular periods", "Acne", "Facial hair", "Mood swings", "Lower abdominal pain"])
    if st.button("Predict Condition"):
        if "Irregular periods" in symptoms and "Acne" in symptoms:
            st.error("⚠️ Possible PCOD detected. Consider Ayurvedic support with Shatavari and Ashoka.")
        else:
            st.info("Condition unclear. Regular observation and Ayurvedic lifestyle suggested.")

# Hair/Skin/Body Care
elif selection == "Hair/Skin/Body Care":
    st.header("🧴 Common Conditions")
    issue = st.selectbox("Choose your issue:", ["Hair Fall", "Acne", "Body Pain", "Fatigue"])
    if st.button("Get Ayurvedic Suggestion"):
        if issue == "Hair Fall":
            st.write("💡 Bhringraj oil, Brahmi, and regular sleep routine recommended.")
        elif issue == "Acne":
            st.write("💡 Neem powder, turmeric, and Pitta-pacifying diet.")
        elif issue == "Body Pain":
            st.write("💡 Abhyanga with warm sesame oil and Dashmool decoction.")
        elif issue == "Fatigue":
            st.write("💡 Ashwagandha, ghee, and pranayama.")

# Pre-Diabetes Detector
elif selection == "Pre-Diabetes Detector":
    st.header("🩺 Pre-Diabetes Risk Checker")
    thirst = st.slider("Thirst Level", 0, 10, 5)
    urination = st.slider("Frequency of Urination", 0, 10, 5)
    fatigue = st.slider("Fatigue Level", 0, 10, 5)
    if st.button("Check Risk"):
        score = thirst + urination + fatigue
        if score > 20:
            st.error("High risk of Pre-Diabetes. Recommend Gudmar, Neem, and lifestyle correction.")
        elif score > 12:
            st.warning("Moderate risk. Ayurvedic guidance advised.")
        else:
            st.success("Low risk. Maintain healthy routine.")

# Knowledge Q&A
elif selection == "Ayurvedic Q&A":
    st.header("📖 Ask the Ayurvedic Knowledge Base")
    query = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if "triphala" in query.lower():
            st.write("🌿 Triphala is a combination of three fruits: Amalaki, Bibhitaki, and Haritaki. It aids digestion and eye health.")
        elif "vata" in query.lower():
            st.write("💨 Vata governs movement. Imbalance causes dryness, anxiety, constipation.")
        else:
            st.info("This prototype supports only limited answers. Full knowledge base coming soon!")
