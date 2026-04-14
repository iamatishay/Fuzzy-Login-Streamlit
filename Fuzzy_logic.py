import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import re

st.set_page_config(
    page_title="Fuzzy Logic Tool",
    layout="wide",
    page_icon="🔍"
)

# ============================================================
# 🔥 NEW: CAMEL CASE SPLITTER
# ============================================================
def split_camel_case(text):
    if not text:
        return text
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

# ============================================================
# UI STYLING
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2f7 0%, #d9e4f5 100%);
}
.block-container {
    background-color: white;
    padding: 2rem;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Fuzzy Logic Tool")

# ============================================================
# RULE SETS
# ============================================================
STOPWORDS = {"THE", "LTD", "PVT", "CO", "AND", "LLP", "INC", "CORP", "COMPANY"}
ESSENTIALS = {"BANK", "INSURANCE", "LIFE"}

REPLACEMENTS = {
    "LIMITED": "LTD",
    "PRIVATE": "PVT",
    "COMPANY": "CO",
    "CORPORATION": "CORP",
}

# ============================================================
# CLEANING
# ============================================================
def normalize_variants(s):
    for k, v in REPLACEMENTS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s

def clean_text(s):
    if pd.isna(s):
        return None

    s = str(s).strip()

    # 🔥 FIX 1: Split CamelCase BEFORE uppercase
    s = split_camel_case(s)

    s = s.upper()
    s = normalize_variants(s)

    # 🔥 FIX 2: Remove special chars
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else None

def strip_stopwords(s):
    if not s:
        return None
    tokens = s.split()
    tokens = [t for t in tokens if t not in STOPWORDS or t in ESSENTIALS]
    return " ".join(tokens)

# ============================================================
# 🔥 IMPROVED MATCH SCORING
# ============================================================
def final_match_score(a, b, **kwargs):

    if not a or not b:
        return 0

    # ========================================================
    # 🔥 FIX 3: SPACE-INSENSITIVE MATCH (FirstCry vs First Cry)
    # ========================================================
    if fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")) > 92:
        return 96

    # Strong substring rule
    if len(a) > 5 and len(b) > 5 and (a in b or b in a):
        return 95

    a_tokens = a.split()
    b_tokens = b.split()

    if not a_tokens or not b_tokens:
        return 0

    a_first = a_tokens[0]
    b_first = b_tokens[0]

    # ========================================================
    # FIRST WORD VALIDATION
    # ========================================================
    shorter = min(len(a_first), len(b_first))

    common_chars = sum(
        min(a_first.count(c), b_first.count(c))
        for c in set(a_first)
    )

    char_match_percent = (common_chars / shorter) * 100 if shorter else 0

    if char_match_percent < 80:
        return 0

    first_score = fuzz.ratio(a_first, b_first)

    remaining_a = " ".join(a_tokens[1:])
    remaining_b = " ".join(b_tokens[1:])

    remaining_score = 0
    if remaining_a and remaining_b:
        remaining_score = fuzz.token_set_ratio(remaining_a, remaining_b)

    final_score = (0.8 * first_score) + (0.2 * remaining_score)

    return round(final_score, 2)

# ============================================================
# MATCH FUNCTION
# ============================================================
def find_best_match(name, cleaned_choices, original_choices, threshold=85):

    main_clean = strip_stopwords(clean_text(name))

    if not main_clean:
        return None, 0

    result = process.extractOne(
        main_clean,
        cleaned_choices,
        scorer=final_match_score
    )

    if result:
        _, score, idx = result
        if score >= threshold:
            return original_choices[idx], score

    return None, 0

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    source_col = st.selectbox("Source Column", df.columns, index=0)
    target_col = st.selectbox("Target Column", df.columns, index=1)

    if st.button("🚀 Run Matching"):

        source_names = df[source_col].dropna().tolist()
        target_names = df[target_col].dropna().tolist()

        target_clean = [strip_stopwords(clean_text(x)) for x in target_names]

        results = []

        for name in source_names:
            match, score = find_best_match(
                name,
                target_clean,
                target_names
            )

            results.append({
                source_col: name,
                "Matched Name": match,
                "Score": score
            })

        out_df = pd.DataFrame(results)

        st.success("Matching Completed")
        st.dataframe(out_df)

        csv = out_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "matched_results.csv",
            "text/csv"
        )
