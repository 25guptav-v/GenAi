import streamlit as st
import nltk
import spacy
import pandas as pd
from datetime import datetime
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Advanced NLP Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; color: #1E88E5; font-weight: bold; margin-bottom: 0px; }
    .date-badge { font-size: 1rem; color: #757575; margin-bottom: 15px; font-weight: 500; }
    .sub-title { font-size: 1.1rem; color: #616161; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# DOWNLOAD NLTK RESOURCES (Cached)
@st.cache_resource(show_spinner="Loading NLTK resources...")
def download_nltk_resources():
    resources = [
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)

download_nltk_resources()

# LOAD SPACY MODEL (Cached)
@st.cache_resource(show_spinner="Loading SpaCy model...")
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("### User Profile")
    st.write("**Name:** Vidya Gupta")
    st.write("**Roll No:** 55")
    st.markdown("---")
    st.markdown("### Settings")
    show_explanations = st.checkbox("Show NLP Explanations", value=True)

# HEADER WITH TODAY'S DATE
current_date = datetime.now().strftime("%B %d, %Y")

st.markdown('<p class="main-title">🧠 Advanced NLP Text Analyzer</p>', unsafe_allow_html=True)
st.markdown(f'<p class="date-badge">📅 Date: {current_date}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter a paragraph below to perform comprehensive Natural Language Processing operations.</p>', unsafe_allow_html=True)

# INPUT AREA
default_text = "Barack Obama was born in Hawaii. He served as the 44th President of the United States. Google is a massive tech company based in Mountain View, California."
text = st.text_area("Input Text:", height=150, value=default_text)

# BUTTONS
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
with col2:
    clear_button = st.button("🗑 Clear", use_container_width=True)

if clear_button:
    st.rerun()

# ANALYSIS LOGIC
if analyze_button:
    if text.strip() == "":
        st.error("⚠️ Please enter some text before analyzing.")
    else:
        doc = nlp(text)
        
        # Create Tabs for organized viewing
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Basic Tokenization", 
            "✂️ Stemming & Lemmatization", 
            "🏷️ POS & Entities", 
            "📊 Analytics & Dependencies"
        ])

        # --- TAB 1: BASIC TOKENIZATION ---
        with tab1:
            st.subheader("1️⃣ Sentence Segmentation")
            sentences = sent_tokenize(text)
            for i, sentence in enumerate(sentences, 1):
                st.info(f"**{i}.** {sentence}")

            st.markdown("---")
            st.subheader("2️⃣ Word Tokenization & Cleaning")
            words = word_tokenize(text)
            
            stop_words = set(stopwords.words("english"))
            # Filter out punctuation and stopwords for clean analysis
            clean_words = [word for word in words if word.isalnum()]
            filtered_words = [word for word in clean_words if word.lower() not in stop_words]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Raw Tokens ({len(words)}):**")
                st.write(words)
            with col_b:
                st.write(f"**Without Stopwords ({len(filtered_words)}):**")
                st.write(filtered_words)

        # --- TAB 2: STEMMING & LEMMATIZATION ---
        with tab2:
            st.subheader("3️⃣ Stemming vs. Lemmatization")
            if show_explanations:
                st.caption("*Stemming chops off word endings, while Lemmatization converts words to their dictionary root based on context.*")
            
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            
            # Create a comparison dataframe
            morph_data = []
            for word in filtered_words:
                morph_data.append({
                    "Original Word": word,
                    "Stemmed (Porter)": stemmer.stem(word),
                    "Lemmatized (WordNet)": lemmatizer.lemmatize(word)
                })
            
            df_morph = pd.DataFrame(morph_data)
            st.dataframe(df_morph, use_container_width=True)

        # --- TAB 3: POS & NER ---
        with tab3:
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.subheader("4️⃣ Part-of-Speech (POS) Tagging")
                if show_explanations:
                    st.caption("*Assigns grammatical categories (Noun, Verb, Adjective) to each word.*")
                
                pos_tags = nltk.pos_tag(clean_words) # Using clean words without punctuation
                df_pos = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                st.dataframe(df_pos, use_container_width=True, height=300)

            with col_d:
                st.subheader("5️⃣ Named Entity Recognition (NER)")
                if show_explanations:
                    st.caption("*Identifies proper nouns like People, Organizations, and Locations.*")
                
                if len(doc.ents) == 0:
                    st.warning("No Named Entities Found.")
                else:
                    ner_data = []
                    for entity in doc.ents:
                        ner_data.append({
                            "Entity": entity.text,
                            "Label": entity.label_,
                            "Description": spacy.explain(entity.label_)
                        })
                    df_ner = pd.DataFrame(ner_data)
                    st.dataframe(df_ner, use_container_width=True, height=300)

        # --- TAB 4: ANALYTICS & DEPENDENCIES ---
        with tab4:
            col_e, col_f = st.columns([1, 1])
            
            with col_e:
                st.subheader("6️⃣ Word Frequency (Top 10)")
                word_freq = Counter([w.lower() for w in filtered_words])
                df_freq = pd.DataFrame(word_freq.most_common(10), columns=["Word", "Count"])
                st.bar_chart(df_freq.set_index("Word"))

            with col_f:
                st.subheader("7️⃣ Dependency Parsing")
                if show_explanations:
                    st.caption("*Shows the syntactic relationships between words.*")
                
                dep_data = []
                for token in doc:
                    if token.is_alpha: # Skip punctuation
                        dep_data.append({
                            "Word": token.text,
                            "Dependency": token.dep_,
                            "Head Word": token.head.text
                        })
                df_dep = pd.DataFrame(dep_data)
                st.dataframe(df_dep, use_container_width=True, height=300)

        st.success("✅ Advanced NLP Analysis Completed Successfully!")