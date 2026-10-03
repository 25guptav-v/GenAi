import streamlit as st
import nltk
import spacy
import pandas as pd
from datetime import datetime
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced NLP Text Analyzer & Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main-title { font-size: 2.3rem; color: #1E88E5; font-weight: bold; margin-bottom: 0px; }
    .date-badge { font-size: 0.95rem; color: #757575; margin-bottom: 15px; font-weight: 500; }
    .sub-title { font-size: 1.05rem; color: #616161; margin-bottom: 20px; }
    .metric-card { background-color: #F8F9FA; padding: 15px; border-radius: 8px; border-left: 5px solid #1E88E5; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLTK resources...")
def download_nltk_resources():
    resources = [
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
    ]
    for resource in resources:
        nltk.download(resource, quiet=True)

download_nltk_resources()

@st.cache_resource(show_spinner="Loading SpaCy model...")
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("### User Profile")
    st.write("**Name:** Vidya Gupta")
    st.write("**Roll No:** 55")
    st.markdown("---")
    st.markdown("### Settings")
    show_explanations = st.checkbox("Show NLP Explanations", value=True)

# -----------------------------------------------------------------------------
# HEADER SECTION
# -----------------------------------------------------------------------------
current_date = datetime.now().strftime("%B %d, %Y")

st.markdown('<p class="main-title">🧠 Advanced NLP Text Analyzer & Executive Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="date-badge">📅 Date: {current_date}</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter a paragraph below to generate real-time metrics and deep NLP linguistic breakdowns.</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INPUT AREA & CONTROL BUTTONS
# -----------------------------------------------------------------------------
default_text = (
    "Barack Obama was born in Hawaii. He served as the 44th President of the United States. "
    "Google is a massive tech company based in Mountain View, California. Natural language "
    "processing allows computers to derive meaning from human text automatically."
)
text = st.text_area("Input Text:", height=140, value=default_text)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
with col2:
    clear_button = st.button("🗑 Clear", use_container_width=True)

if clear_button:
    st.rerun()

# -----------------------------------------------------------------------------
# ANALYSIS EXECUTION
# -----------------------------------------------------------------------------
if analyze_button:
    if not text.strip():
        st.error("⚠️ Please enter valid text before executing the analysis.")
    else:
        doc = nlp(text)
        
        # Token Pre-processing
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        clean_words = [word for word in words if word.isalnum()]
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in clean_words if word.lower() not in stop_words]

        # ---------------------------------------------------------------------
        # TABBED NAVIGATION STRUCTURE
        # ---------------------------------------------------------------------
        tab_dash, tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Executive Dashboard",
            "📝 Basic Tokenization", 
            "✂️ Morphological Analysis", 
            "🏷️ POS & Entities", 
            "📈 Syntax & Analytics"
        ])

        # =====================================================================
        # TAB 0: EXECUTIVE DASHBOARD
        # =====================================================================
        with tab_dash:
            st.subheader("📈 Summary Metrics")
            
            # Key Metrics Calculation
            total_words = len(clean_words)
            total_sentences = len(sentences)
            unique_tokens = len(set(w.lower() for w in clean_words))
            lexical_diversity = (unique_tokens / total_words * 100) if total_words > 0 else 0
            est_reading_time = max(1, round(total_words / 200 * 60)) # Seconds @ 200 wpm
            total_entities = len(doc.ents)

            # Metric Cards Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Sentences", total_sentences)
            m2.metric("Total Word Tokens", total_words)
            m3.metric("Unique Words", unique_tokens)
            m4.metric("Lexical Diversity", f"{lexical_diversity:.1f}%")
            m5.metric("Named Entities", total_entities)

            st.markdown("---")

            # Dashboard Visualizations & Breakdown
            d_col1, d_col2 = st.columns(2)

            with d_col1:
                st.subheader("🔤 Top 10 Most Frequent Words")
                word_freq = Counter([w.lower() for w in filtered_words])
                df_freq = pd.DataFrame(word_freq.most_common(10), columns=["Word", "Count"])
                if not df_freq.empty:
                    st.bar_chart(df_freq.set_index("Word"))
                else:
                    st.info("No words remaining after stopword filter.")

            with d_col2:
                st.subheader("🏷️ POS Category Distribution")
                pos_tags = nltk.pos_tag(clean_words)
                pos_counts = Counter([tag for _, tag in pos_tags])
                df_pos_dist = pd.DataFrame(pos_counts.items(), columns=["POS Tag", "Count"]).sort_values("Count", ascending=False)
                st.bar_chart(df_pos_dist.set_index("POS Tag"))

            st.info(f"⏱️ **Estimated Reading Time:** ~{est_reading_time} seconds (based on 200 WPM standard reading speed).")

        # =====================================================================
        # TAB 1: BASIC TOKENIZATION
        # =====================================================================
        with tab1:
            st.subheader("1️⃣ Sentence Segmentation")
            for i, sentence in enumerate(sentences, 1):
                st.info(f"**Sentence {i}:** {sentence}")

            st.markdown("---")
            st.subheader("2️⃣ Word Tokenization & Filtering")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Raw Tokens ({len(words)}):**")
                st.json(words)
            with col_b:
                st.write(f"**Filtered Tokens (No Stopwords/Punctuation - {len(filtered_words)}):**")
                st.json(filtered_words)

        # =====================================================================
        # TAB 2: MORPHOLOGICAL ANALYSIS
        # =====================================================================
        with tab2:
            st.subheader("3️⃣ Stemming vs. Lemmatization Comparison")
            if show_explanations:
                st.caption("*Stemming uses heuristic rules to trim suffixes, whereas Lemmatization uses vocabulary and morphological analysis to return valid dictionary roots.*")
            
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            
            morph_data = [
                {
                    "Original Word": word,
                    "Stemmed (Porter)": stemmer.stem(word),
                    "Lemmatized (WordNet)": lemmatizer.lemmatize(word)
                }
                for word in filtered_words
            ]
            
            df_morph = pd.DataFrame(morph_data)
            st.dataframe(df_morph, use_container_width=True, height=350)

        # =====================================================================
        # TAB 3: POS & NAMED ENTITIES
        # =====================================================================
        with tab3:
            col_c, col_d = st.columns(2)
            
            with col_c:
                st.subheader("4️⃣ Part-of-Speech (POS) Tagging")
                if show_explanations:
                    st.caption("*Assigns grammatical roles (Nouns, Verbs, Adjectives, Adverbs) using the Penn Treebank tagset.*")
                
                pos_tags = nltk.pos_tag(clean_words)
                df_pos = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                st.dataframe(df_pos, use_container_width=True, height=350)

            with col_d:
                st.subheader("5️⃣ Named Entity Recognition (NER)")
                if show_explanations:
                    st.caption("*Extracts real-world entities (Persons, Organizations, Locations, Numbers).*")
                
                if len(doc.ents) == 0:
                    st.warning("No Named Entities detected in the text.")
                else:
                    ner_data = [
                        {
                            "Entity Text": entity.text,
                            "Type Label": entity.label_,
                            "Label Description": spacy.explain(entity.label_)
                        }
                        for entity in doc.ents
                    ]
                    df_ner = pd.DataFrame(ner_data)
                    st.dataframe(df_ner, use_container_width=True, height=350)

        # =====================================================================
        # TAB 4: SYNTAX & DEPENDENCIES
        # =====================================================================
        with tab4:
            st.subheader("6️⃣ Dependency Tree Structure")
            if show_explanations:
                st.caption("*Maps grammatical relationships connecting head words to dependent tokens.*")
            
            dep_data = [
                {
                    "Token Text": token.text,
                    "POS Role": token.pos_,
                    "Dependency Relationship": token.dep_,
                    "Head Word": token.head.text
                }
                for token in doc if token.is_alpha
            ]
            df_dep = pd.DataFrame(dep_data)
            st.dataframe(df_dep, use_container_width=True, height=350)

        st.success("✅ Complete Analysis & Dashboard Generation Finalized!")