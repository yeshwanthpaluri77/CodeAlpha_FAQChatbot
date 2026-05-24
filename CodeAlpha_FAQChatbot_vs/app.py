import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #0f172a;
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 35px;
}

/* Input Box */
textarea, input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Chat Box */
.chat-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 16px;
    margin-top: 20px;
    border: 1px solid #334155;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎓 College Admission FAQ Chatbot</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-text">Ask any college admission related question!</div>',
    unsafe_allow_html=True
)

st.markdown("""
###Suggested Example Questions
- What courses are offered?
- Is hostel available?
- What is the fee structure?
- Where is the college located?
- How can I apply?
""")

# FAQ DB
faqs = {
    "What courses are offered?":
    "We offer CSE,ECE,EEE,AI,MECH,CIVIL,MME,CHEMICAL engineering courses.",

    "Is hostel available?":
    "Yes,Separate hostel facilities are available for boys and girls. There are 3 hostels for both boys and girlss",

    "Where is the college located?":
    "The college is located in Basar, which is in nirmal district, Telangana.",

    "What is the fee structure?":
    "The fee structure varies depending on the course you choose and the category of the admission",

    "How can I apply?":
    "Every year there will be specific notification relased by tge college and then you can apply through the official college admission portal.",

    "Is transportation available?":
    "Yes, bus transportation facilities are available.",

    "Do you provide placements?":
    "Yes, placement assistance is provided for eligible students and we also have many companys in which the founders are the alumni of our company ",

    "What are the college timings?":
    "College timings are from 9 AM to 4 PM, including with lunch break",

    "Is there a library in the college?":
    "Yes, the college has 2 librarys one is the normal reading library with many number no of books in it, and the other one is digital library in which it contains laptops with high speed wifi",

    "Are scholarships available?":
    "Yes, scholarships are available based on the category and eligibility."
}
questions = list(faqs.keys())

#input
user_question = st.text_input(
    "Enter your question"
)

if st.button("Ask Question"):
    if user_question:
        #TEXT VECTORIZATION
        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            questions + [user_question]
        )

#similarity check
        similarity = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )
#BEst match
        best_match_index = similarity.argmax()

#Best Score
        best_score = similarity[0][best_match_index]

#Threshold Check
        if best_score > 0.3:
            best_question = questions[best_match_index]
            answer = faqs[best_question]

#Display Response
            st.markdown(f"""
            <div class="chat-box">
                <h4>🤖 Bot Response</h4>
                <p style="font-size:18px;">
                {answer}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(
                "Sorry, I can only answer college admission related questions."
            )
    else:
        st.warning("Please enter a question.")