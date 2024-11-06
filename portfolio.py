import streamlit as st 
from PIL import Image
import numpy as np

img = Image.open('avtr.png')

col1,col2 = st.columns(2)
with col1:
    st.image(img,caption="Vishal",width=200)
    st.write("[9816290993](+91-9816290993)")
    st.write("[vishaal03.it@gmail.com](vishaal03.it@gmail.com)")
    st.write("[Git](https://github.com/VISHAL-038/)")
    st.write("[linkdin](www.linkedin.com/in/vishal-datascience)")
with col2:
    st.header("Summary") 
    st.write("A final-year B.Tech student in Computer Science and Engineering at Rayat Bahra University, graduating in 2025 with a CGPA of 7.86. Holds certifications in Data Science (IBM), Machine Learning and Deep Learning (NPTEL), and Full Stack Development. Skilled in Python, data analysis, and machine learning with expertise in libraries like Pandas, Scikit-learn, Seaborn, and OpenCV, alongside experience with MongoDB and SQL. Developed a Spam Comment Detection model with 92% accuracy, a Credit Card Fraud Detection system using multiple classifiers, a real-time Q&A bot with Streamlit and Cohere API, and an Object Detection system using YOLOv3. Known for problem-solving, effective communication, and leadership, with projects and skills showcased on LinkedIn and GitHub.")
st.success(" ")

# Qualifications
st.subheader("Qualifications")
st.write("""
**Bachelors of Technology - Computer Science Engineering** (2021-2025)

- Rayat Bahra University, Punjab 
- CGPA: 7.86/10


##### Certifications & Self-Learning

- **IBM Data Science Professional Certificate:** Completed training in Python, data analysis, and machine learning, focusing on statistical techniques and model evaluation.
  
- **Machine Learning and Deep Learning - NPTEL:** Acquired knowledge in supervised and unsupervised learning, neural networks, and their real-world applications.

- **Full Stack Development Program - Excellence Education:** Developed skills in JavaScript, React, and backend technologies using Node.js and MySQL.

- **Full Stack Web Development - MCP Technology:** Enhanced expertise in both front-end and back-end technologies for building scalable web applications.
""")

st.success(" ")
# Projects
st.header("Projects")
# Load your images (you can use paths or URLs)
image1 = "QA-Bot.png"
image2 = "spam.png"
image3 = "Object-detection.png"

# Create three columns for displaying images side-by-side
col1, col2, col3 = st.columns(3)

# Display each image in its respective column
with col1:
    st.image(image1, caption="QA Bot", use_column_width=True,width=200)
    st.write("[GitHub](https://github.com/VISHAL-038/Q-A-Bot)")
    st.write("""
    **Objective:** Built a QA Bot using Retrieval-Augmented Generation (RAG).

    **Tech:** Python, RAG model, Vector Database.

    **Description:** Created a bot that combines retrieval and generative models for efficient document search and accurate answers.

    **Process:**
    - Architected the QA Bot.
    - Integrated vector database for fast retrieval.
    - Applied RAG model for high-quality responses.

    **Outcome:** Achieved `85%` accuracy and reduced response time.

    """)

with col2:
    st.image(image2, caption="Spam Comment Detection", use_column_width=True,width=200)
    st.write("[GitHub](https://github.com/VISHAL-038/Spam_comment_detection)")

    st.write("""
    **Objective:** Built a machine learning model to classify spam comments.

    **Role:** Project Lead, focusing on model design and optimization.

    **Tech:** Python, Scikit-Learn, NLP, NLTK, Pandas, NumPy.

    **Process:**
    - Preprocessed comment datasets using NLP for cleaning and feature extraction.
    - Employed TF-IDF and word embeddings for effective text representation.
    - Trained classifiers (Logistic Regression, Naive Bayes, Random Forest) and fine-tuned via cross-validation.

    **Outcome:** 
    - Achieved over 95% accuracy in spam detection, reducing false positives.
    - Deployed a scalable model for real-time comment moderation, enhancing user experience.
    """)





with col3:
    st.image(image3, caption="Real time Object Detection", use_column_width=True,width=200)
    st.write("[GitHub](https://github.com/VISHAL-038/Real_Time_Object_Detection)")
    st.write("""
    **Objective:** Created a real-time object detection system with YOLOv3 and OpenCV.

    **Role:** Project Lead, focused on pipeline development and optimization.

    **Tech:** Python, YOLOv3, OpenCV, NumPy.

    **Process:** 
    - Integrated YOLOv3 for detecting `40+` object types in live video.
    - Optimized processing to reach 30 FPS, cutting latency by 45%.
    - Enhanced detection accuracy and minimized false positives.

    **Outcome:** Achieved over `90%` detection accuracy with smooth, real-time performance.
    """)

st.success(" ")

# Skills 
st.header("Skills")
st.write("""
- **Data Analysis:** `Pandas`, `NumPy`, `Feature Engineering`, `Model Evaluation`
- **Machine Learning:** `Regression`, `K-Means`, `Random Forest`, `Deep Learning`
- **Programming Languages:** `Python`, `Java`
- **Database Management:** `MySQL`, `MongoDB`
- **Tools:** `Jupyter Notebook`, `Git`
- **Frameworks:** `OpenCV`, `Scikit-Learn`, `YOLOv3`
""")