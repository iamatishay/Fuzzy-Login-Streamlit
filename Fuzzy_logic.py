import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import re

st.set_page_config(
    page_title="Fuzzy Logic Tool",
    layout="wide",
    page_icon="🔍"
)

# ===== Custom Styling =====
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }
    .sub-text {
        font-size: 16px;
        color: #555555;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f6ff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d0e2ff;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    
    /* ===== App Background ===== */
    .stApp {
        background: linear-gradient(135deg, #eef2f7 0%, #d9e4f5 100%);
        background-attachment: fixed;
    }

    /* ===== Main Content Container ===== */
    .block-container {
        background-color: white;
        padding: 2rem 3rem 3rem 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    /* ===== Title Styling ===== */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }

    .sub-text {
        font-size: 16px;
        color: #555555;
        margin-bottom: 20px;
    }

    /* ===== Instruction Box ===== */
    .info-box {
        background-color: #f0f6ff;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #1f4e79;
        margin-bottom: 25px;
        font-size: 15px;
    }

    /* ===== Buttons Styling ===== */
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
    }

    .stButton>button:hover {
        background-color: #163a5c;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)


# ===== Header Section =====
st.markdown('<div class="main-title">🔍 Fuzzy Logic Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Match company names intelligently using advanced fuzzy logic, token scoring, and rule-based validation.</div>',
    unsafe_allow_html=True
)

# ===== Instruction Box =====
st.markdown("""
<div class="info-box">
<b>📌 How to Use:</b><br>
1️⃣ Download the sample file below to understand the required format (2 columns).<br>
2️⃣ Upload your file (CSV or Excel).<br>
3️⃣ Select Source (Match FROM) and Target (Match TO) columns.<br>
4️⃣ Click <b>Run Matching</b> to generate results.<br><br>
⚠️ Ensure your file contains at least two columns for matching.
</div>
""", unsafe_allow_html=True)


# ============================================================
# SAMPLE FILE DOWNLOAD
# ============================================================
sample_df = pd.DataFrame({
    "Source_Name": [
        "TANLA PLATFORMS LTD",
        "HDFC BANK",
        "SBI LIFE INSURANCE",
        "VODAFONE IDEA",
        "TITAN SERVICES"
    ],
    "Target_Name": [
        "TANLA PLATFORMS LIMITED",
        "HDFC BANK LIMITED",
        "STATE BANK OF INDIA LIFE INSURANCE",
        "VODAFONE IDEA LIMITED",
        "TITAN COMPANY LIMITED"
    ]
})

sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Sample CSV (2 Columns)",
    data=sample_csv,
    file_name="sample_2_column_name_matching.csv",
    mime="text/csv"
)

# ============================================================
# RULE SETS (EXPANDED FOR PRODUCTION)
# ============================================================
STOPWORDS = {"THE", "LTD", "PVT", "CO", "AND", "LLP", "INC", "CORP", "COMPANY", "SERVICES", "GROUP", "INDIA"}
ESSENTIALS = {"ENERGY", "ELECTRIC", "BANK", "INSURANCE", "LIFE", "GENERAL"}
REGIONAL_WORDS = {"NORTH", "SOUTH", "EAST", "WEST", "UTTAR", "DAKSHIN", "CENTRAL", "NORTHERN", "SOUTHERN", "EASTERN", "WESTERN"}
INSURANCE_LIFE = {"LIFE"}
INSURANCE_GENERAL = {"GENERAL", "GI", "INSURANCE", "NON-LIFE"}
SERVICE_WORDS = {"SERVICES", "GROUP", "SOLUTIONS", "SYSTEMS"}
MESSAGE_WORDS = {"MESSAGE", "MESSAGING", "SMS", "COMMUNICATION"}
TELECOM_WORDS = {"TELECOM", "VODAFONE", "IDEA", "AIRTEL", "JIO", "BHARTI"}

REPLACEMENTS = {
    "PVT.": "PVT",
    "PRIVATE": "PVT",
    "LIMITED": "LTD",
    "LTD.": "LTD",
    "&": "AND",
    "CO.": "CO",
    "TECHNOLOGY": "TECH",
    "TECHNOLOGIES": "TECH",
    "CORPORATION": "CORP",
    "COMPANY": "CO",
    "SERVICES": "SVC",
    "GROUP": "GRP",
    "BANK": "BK",
    "INSURANCE": "INS",
    "LIFE": "LF",
    "GENERAL": "GEN",
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def get_acronym(s):
    """Generate acronym from words, ignoring stopwords."""
    words = [w for w in s.split() if w not in STOPWORDS]
    return "".join(w[0] for w in words if w)

def is_acronym_match(a, b):
    """Check if one is an acronym of the other, with fuzzy and substring checks."""
    a_acr = get_acronym(a)
    b_acr = get_acronym(b)
    if a_acr == b_acr:
        return True
    if len(a_acr) < len(b_acr) and a_acr in b_acr:
        return True
    if len(b_acr) < len(a_acr) and b_acr in a_acr:
        return True
    if fuzz.ratio(a_acr, b_acr) > 80:
        return True
    return False

# ============================================================
# 3. Cleaning Functions
# ============================================================
# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def normalize_variants(s):
    for k, v in REPLACEMENTS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s

def clean_text(s):
    if pd.isna(s):
        return None
    s = str(s).upper().strip()
    s = normalize_variants(s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

def strip_stopwords(s):
    if not s:
        return None
    tokens = s.split()
    tokens = [
        t for t in tokens
        if (t not in STOPWORDS) or (t in ESSENTIALS)
    ]
    return " ".join(tokens) if tokens else None

# ============================================================
# HARD VALIDATION RULES
# ============================================================

def has_conflicting_region(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    for w in REGIONAL_WORDS:
        if (w in a_tokens) != (w in b_tokens):
            return True
    return False

def insurance_conflict(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if (INSURANCE_LIFE & a_tokens and INSURANCE_GENERAL & b_tokens) or \
       (INSURANCE_LIFE & b_tokens and INSURANCE_GENERAL & a_tokens):
        return True
    return False

# ============================================================
# ACRONYM LOGIC
# ============================================================

def get_acronym(s):
    words = [w for w in s.split() if w not in STOPWORDS]
    return "".join(w[0] for w in words if w)

def is_acronym_match(a, b):
    a_acr = get_acronym(a)
    b_acr = get_acronym(b)

    if a_acr == b_acr:
        return True

    if len(a_acr) < len(b_acr) and a_acr in b_acr:
        return True

    if len(b_acr) < len(a_acr) and b_acr in a_acr:
        return True

    if fuzz.ratio(a_acr, b_acr) > 85:
        return True

    return False

# ============================================================
# TOKEN SCORING ENGINE (LEGAL WORDS HAVE ZERO WEIGHT)
# ============================================================

def token_based_score(a, b):
    a_tokens = a.split()
    b_tokens = b.split()

    if not a_tokens or not b_tokens:
        return 0

    token_matches = []

    for t1 in a_tokens:
        best = 0
        for t2 in b_tokens:
            best = max(best, fuzz.ratio(t1, t2))
        token_matches.append(best)

    # First meaningful word gets weight 1.5
    weights = [1.5] + [1.0] * (len(token_matches) - 1)

    weighted_sum = sum(t * w for t, w in zip(token_matches, weights))
    max_possible = sum(weights) * 100

    return (weighted_sum / max_possible) * 100

def expand_name_variants(name):
    
    variants = [name]

        # Extract bracket content
    bracket_match = re.search(r"\((.*?)\)", name)
    if bracket_match:
        inside = bracket_match.group(1)
        outside = re.sub(r"\(.*?\)", "", name).strip()

        variants.append(inside)
        variants.append(outside)

    return list(set(variants))


# ============================================================
# FINAL MATCH ENGINE
# ============================================================

def final_match_score(a, b, **kwargs):

    # Strong substring rule (exact containment)
    if a in b or b in a:
        return 95


    if not a or not b:
        return 0

    a_tokens = a.split()
    b_tokens = b.split()

    if not a_tokens or not b_tokens:
        return 0

    a_first = a_tokens[0]
    b_first = b_tokens[0]

    # ---------------------------------------------------
    # 80% CHARACTER MATCH RULE (ORDER INDEPENDENT)
    # ---------------------------------------------------
    shorter = min(len(a_first), len(b_first))

    common_chars = sum(
        min(a_first.count(c), b_first.count(c))
        for c in set(a_first)
    )

    char_match_percent = (common_chars / shorter) * 100 if shorter > 0 else 0

    # If first word < 80% → reject completely
    if char_match_percent < 80:
        return 0

    # Strong first word score
    first_word_score = fuzz.ratio(a_first, b_first)

    # Remaining tokens similarity
    remaining_a = " ".join(a_tokens[1:])
    remaining_b = " ".join(b_tokens[1:])

    remaining_score = 0
    if remaining_a and remaining_b:
        remaining_score = fuzz.token_set_ratio(remaining_a, remaining_b)

    # ---------------------------------------------------
    # FINAL WEIGHTING
    # 80% first word
    # 20% remaining words
    # ---------------------------------------------------
    final_score = (0.8 * first_word_score) + (0.2 * remaining_score)

    return round(min(100, final_score), 2)

# ============================================================
# MATCHING FUNCTION
# ============================================================

def find_best_match(main_name, cleaned_choices, original_choices, threshold=92):

    if not main_name:
        return None, 0, "No Match"

    parts = []
    for p in str(main_name).split("/"):
        parts.extend(expand_name_variants(p.strip()))


    best_match = None
    best_score = 0

    for part in parts:

        main_clean = strip_stopwords(clean_text(part))
        if not main_clean:
            continue

        result = process.extractOne(
            main_clean,
            cleaned_choices,
            scorer=final_match_score,
            score_cutoff=threshold
        )

        if result:
            _, score, idx = result
            if score > best_score:
                best_score = score
                best_match = original_choices[idx]

    if best_score >= threshold:
        return best_match, round(best_score, 2), "High Confidence"

    return None, 0.0, "No Match"

# ============================================================
# STREAMLIT UI
# ============================================================


uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")

    st.dataframe(df.head())

        # ===== Sidebar Controls =====
    st.sidebar.header("⚙ Match Configuration")

    source_col = st.sidebar.selectbox(
        "Select Source Column",
        df.columns,
        index=0
    )

    target_col = st.sidebar.selectbox(
        "Select Target Column",
        df.columns,
        index=1
    )

    threshold = st.sidebar.slider(
        "Match Threshold (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    if st.button("🚀 Run Matching"):

        source_names = df[source_col].dropna().tolist()
        target_names = df[target_col].dropna().tolist()

        target_clean = [
            strip_stopwords(clean_text(x))
            for x in target_names
        ]

        source_clean = [
            strip_stopwords(clean_text(x))
            for x in source_names
        ]

        # ============================================================
        # FUZZY MATCHING
        # ============================================================

        rows = []

        for i, name in enumerate(source_names):

            match, score, mtype = find_best_match(
                name,
                target_clean,
                target_names,
                threshold
            )

            rows.append({
                source_col: name,
                "Matched Name": match,
                "Score": score,
                "Match Type": mtype
            })

        out_df = pd.DataFrame(rows)

        # ============================================================
        # SIMULATED EXCEL VLOOKUP / XLOOKUP (EXACT MATCH)
        # ============================================================

        # ============================================================
# TRUE EXCEL VLOOKUP / XLOOKUP SIMULATION (EXACT MATCH)
# ============================================================

# Excel is case-insensitive but otherwise exact
        target_raw_set = set(str(x).strip().upper() for x in target_names)

        excel_matches = []

        for name in source_names:
            if str(name).strip().upper() in target_raw_set:
                excel_matches.append("Exact Match")
            else:
                excel_matches.append("No Match")

        out_df["Excel Match Type"] = excel_matches


        st.success("Matching Completed")
        st.dataframe(out_df)

        st.markdown("## 📊 Match Summary Dashboard")

        total_records = len(out_df)
        total_fuzzy_matches = (out_df["Match Type"] == "High Confidence").sum()
        total_excel_matches = (out_df["Excel Match Type"] == "Exact Match").sum()

        total_no_matches = (out_df["Match Type"] == "No Match").sum()
        fuzzy_match_rate = round((total_fuzzy_matches / total_records) * 100, 2) if total_records else 0
        excel_match_rate = round((total_excel_matches / total_records) * 100, 2) if total_records else 0


        st.warning("⚠️ **Important Note:** Matches are subject to manual scrutiny as the algorithm may occasionally produce false positives. Please review results carefully.")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("📄 Total Records", total_records)
        k2.metric("🤖 Fuzzy Matches", total_fuzzy_matches)
        k3.metric("📊 Excel Exact Matches", total_excel_matches)
        k4.metric("📈 Fuzzy Match Rate", f"{fuzzy_match_rate}%")

        st.info(f"📊 Excel Exact Match Rate (VLOOKUP/XLOOKUP) would be: {excel_match_rate}%")
        st.info(f"Excel Match is being calculated based on exact cleaned name (removing special characters and spaces) matches, simulating VLOOKUP/XLOOKUP behavior.")


        csv = out_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "matched_results.csv",
            "text/csv"
        )
