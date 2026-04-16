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
    .stApp {
        background: linear-gradient(135deg, #eef2f7 0%, #d9e4f5 100%);
        background-attachment: fixed;
    }
    .block-container {
        background-color: white;
        padding: 2rem 3rem 3rem 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
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
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #1f4e79;
        margin-bottom: 25px;
        font-size: 15px;
    }
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

# ===== Header =====
st.markdown('<div class="main-title">🔍 Fuzzy Logic Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Match company names intelligently using hybrid LCS + core-anchor scoring, '
    'token penalty, and rule-based validation.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
<b>📌 How to Use:</b><br>
1️⃣ Download the sample file below to understand the required format (2 columns).<br>
2️⃣ Upload your file (CSV or Excel).<br>
3️⃣ Select Source (Match FROM) and Target (Match TO) columns.<br>
4️⃣ Adjust the match threshold in the sidebar.<br>
5️⃣ Click <b>Run Matching</b> to generate results.<br><br>
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
        "BAJAJ FINANCE",
        "FIRSTCRY",
        "ABAR",
        "MAX POINT",
        "BATEEL GULF TRADING CO",
        "INTERNATIONAL APPLICATION COMPANY LTD",
        "CPPO",
    ],
    "Target_Name": [
        "TANLA PLATFORMS LIMITED",
        "HDFC BANK LIMITED",
        "SBI LIFE INSURANCE CO LTD",
        "BAJAJ HOUSING FINANCE",
        "FIRST CRY TRADING CO",
        "ABAR APP",
        "MAX POINT JEDDAH",
        "BATEEL ALKHALEJ FOR TRADING",
        "INTERNATIONAL RECRUITMENT COMPANY",
        "CPPO INTERNATIONAL",
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
# WORD LISTS
# ============================================================

# Legal / structural noise — stripped before scoring
STOPWORDS = {
    "THE", "LTD", "PVT", "CO", "AND", "LLP", "INC", "CORP", "CORPORATION",
    "COMPANY", "A", "AN", "OF", "FOR", "BY", "IN", "AT", "WITH", "LLC",
    "FZ", "FZE", "FZCO", "FZC", "WLL", "BSC", "JSC", "SARL", "SAS", "BV",
    "NV", "AG", "GMBH", "OY", "AB", "AS", "SA",
}

# Descriptor suffixes — describe the entity type but not its identity.
# Stripped so "Abar App" can match "Abar", "Max Point Jeddah" can match "Max Point".
DESCRIPTOR_SUFFIXES = {
    "APP", "APPS", "APPLICATION", "APPLICATIONS",
    "TRADING", "TRADE", "TRADERS", "TRADER",
    "SERVICES", "SERVICE", "SVC",
    "GROUP", "GRP",
    "SOLUTIONS", "SOLUTION",
    "SYSTEMS", "SYSTEM",
    "TECHNOLOGIES", "TECHNOLOGY", "TECH",
    "ENTERPRISES", "ENTERPRISE",
    "VENTURES", "VENTURE",
    "HOLDINGS", "HOLDING",
    "INTERNATIONAL", "INTL",
    "GLOBAL", "WORLDWIDE",
    "INDUSTRIES", "INDUSTRY",
    "MANAGEMENT", "MGMT",
    "INVESTMENT", "INVESTMENTS",
    "CONSULTANCY", "CONSULTING",
    "RESOURCES", "WORKS",
    "DISTRIBUTION", "DISTRIBUTORS",
}

# Geographic qualifiers — location suffixes that don't change core identity
GEO_WORDS = {
    "JEDDAH", "RIYADH", "DUBAI", "ABUDHABI", "ABU", "DHABI", "DOHA",
    "KUWAIT", "BAHRAIN", "MUSCAT", "OMAN", "CAIRO", "BEIRUT",
    "INDIA", "INDIAN", "GULF", "MENA", "ASIA", "EUROPE", "AFRICA",
    "MIDDLE", "EAST",
    "ALKHALEJ", "ALKHALEEJ", "ALKHALIJ",
}

# Always kept even if they appear in STOPWORDS
ESSENTIALS = {"LIFE", "BANK", "ENERGY", "ELECTRIC", "GENERAL", "INSURANCE", "FINANCE"}

# If a DISCRIMINATOR word is in one name but NOT the other → block the match.
# This prevents: Bajaj Finance ↔ Bajaj HOUSING Finance
#                International APPLICATION Co ↔ International RECRUITMENT Co
#                SBI LIFE ↔ SBI GENERAL
DISCRIMINATORS = {
    "NORTH", "SOUTH", "EAST", "WEST",
    "NORTHERN", "SOUTHERN", "EASTERN", "WESTERN", "CENTRAL",
    "UTTAR", "DAKSHIN",
}

REPLACEMENTS = {
    "PVT.":    "PVT",
    "PRIVATE": "PVT",
    "LIMITED": "LTD",
    "LTD.":    "LTD",
    "&":       "AND",
    "CO.":     "CO",
}

# ============================================================
# CLEANING
# ============================================================

def normalize_variants(s: str) -> str:
    for k, v in REPLACEMENTS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def clean_text(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).upper().strip()
    s = normalize_variants(s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = normalize_repeated_chars(s)   # ✅ ADD THIS LINE
    return s if s else None

def strip_noise(s: str) -> str | None:
    """Strip STOPWORDS + DESCRIPTOR_SUFFIXES + GEO_WORDS, keep ESSENTIALS."""
    if not s:
        return None
    noise = STOPWORDS | DESCRIPTOR_SUFFIXES | GEO_WORDS
    tokens = [t for t in s.split() if (t in ESSENTIALS) or (t not in noise)]
    return " ".join(tokens) if tokens else None


def collapse_spaces(s: str) -> str:
    """Remove all spaces: 'FIRST CRY' → 'FIRSTCRY'."""
    return s.replace(" ", "")

# ============================================================
# DISCRIMINATOR GUARD
# ============================================================

def has_discriminator_conflict(a: str, b: str) -> bool:
    """
    Block matches where a meaningful identity word is present in one name
    but absent in the other.
    """
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    for word in DISCRIMINATORS:
        if (word in a_tokens) != (word in b_tokens):
            return True
    return False

# ============================================================
# LCS HELPERS
# ============================================================

def lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def lcs_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    lcs = lcs_length(a, b)
    return round((2 * lcs / (len(a) + len(b))) * 100, 2)


def normalize_repeated_chars(s: str) -> str:
    """
    Collapse repeated characters:
    HADAYAA → HADAYA
    SOOOFT  → SOFT
    Keeps legitimate double letters intact.
    """
    return re.sub(r'(.)\1{2,}', r'\1', s)

def is_single_token_close_match(a: str, b: str) -> bool:
    """
    Allow single-word brands with very close edit distance
    HADAYA ↔ HADAYAA
    LULU ↔ LULUU
    """
    if " " in a or " " in b:
        return False
    return fuzz.ratio(a, b) >= 88

def strip_trailing_vowels(s: str) -> str:
    """
    Remove trailing vowel elongation:
    HAYAA → HAY
    HAYA  → HAY
    """
    return re.sub(r'[AEIOU]+$', '', s)

def is_phonetic_single_token_match(a: str, b: str) -> bool:
    """
    Match single-word names with trailing vowel elongation.
    HAYA ↔ HAYAA
    """
    if " " in a or " " in b:
        return False

    a_base = strip_trailing_vowels(a)
    b_base = strip_trailing_vowels(b)

    # Skeleton must match exactly
    if a_base == b_base and len(a_base) >= 3:
        return True

    return False


def is_phonetic_single_token_match(a: str, b: str) -> bool:
    """
    Match single-word names with trailing vowel elongation.
    HAYA ↔ HAYAA
    """
    if " " in a or " " in b:
        return False

    a_base = strip_trailing_vowels(a)
    b_base = strip_trailing_vowels(b)

    # Skeleton must match exactly
    if a_base == b_base and len(a_base) >= 3:
        return True

    return False




# ============================================================
# CORE HYBRID SCORER
# ============================================================

def score_pair(a: str, b: str) -> float:
    """
    Hybrid score combining:
    1. Space-collapsed LCS  → 'FIRSTCRY' vs 'FIRSTCRY' (from 'First Cry')
    2. Token LCS            → normal multi-word comparison
    3. Core-anchor (partial_ratio) → shorter name contained in longer
    4. Soft token penalty   → extra suffix tokens penalised lightly (4% each, max 20%)
    """
    if not a or not b:
        return 0.0

    # 1. Space-collapsed LCS
    a_col = collapse_spaces(a)
    b_col = collapse_spaces(b)
    collapsed_score = lcs_ratio(a_col, b_col)

    # 2. Token-level LCS
    token_score = lcs_ratio(a, b)

    base_score = max(collapsed_score, token_score)

    # 3. Core-anchor: how well does the shorter fit inside the longer?
    a_tokens = a.split()
    b_tokens = b.split()
    shorter_str = " ".join(a_tokens if len(a_tokens) <= len(b_tokens) else b_tokens)
    longer_str  = " ".join(b_tokens if len(a_tokens) <= len(b_tokens) else a_tokens)
    core_score  = fuzz.partial_ratio(shorter_str, longer_str)

    # Blend LCS and core-anchor equally
    blended = 0.50 * base_score + 0.50 * core_score

    # 4. Soft penalty for extra tokens in longer name
    shorter_set = set(shorter_str.split())
    longer_set  = set(longer_str.split())
    extra        = longer_set - shorter_set
    penalty      = max(0.80, 1.0 - len(extra) * 0.04)

    return round(min(100, blended * penalty), 2)


def smart_containment_score(a: str, b: str) -> float:
    shorter_len = min(len(a), len(b))
    longer_len  = max(len(a), len(b))
    coverage    = shorter_len / longer_len
    score       = coverage * 90 + min(shorter_len / 20, 1.0) * 8
    return round(min(98, score), 2)


def final_match_score(a: str, b: str, **kwargs) -> float:
    """Entry point for rapidfuzz.process.extractOne."""
    if not a or not b:
        return 0.0
    a_col = collapse_spaces(a)
    b_col = collapse_spaces(b)
    if len(a_col) > 4 and len(b_col) > 4 and (a_col in b_col or b_col in a_col):
        return smart_containment_score(a, b)
    return score_pair(a, b)

# ============================================================
# ACRONYM HELPERS
# ============================================================

def get_acronym(s: str) -> str:
    noise = STOPWORDS | DESCRIPTOR_SUFFIXES | GEO_WORDS
    words = [w for w in s.split() if (w in ESSENTIALS) or (w not in noise)]
    return "".join(w[0] for w in words if w)


def is_acronym_match(a: str, b: str) -> bool:
    a_acr = get_acronym(a)
    b_acr = get_acronym(b)

    # Acronyms must be meaningful
    if len(a_acr) < 3 or len(b_acr) < 3:
        return False

    if a_acr != b_acr:
        return False

    # Require at least one real core word overlap
    a_core = set(strip_noise(a).split())
    b_core = set(strip_noise(b).split())

    return bool(a_core & b_core)

# ============================================================
# NAME VARIANT EXPANSION
# ============================================================

def expand_name_variants(name: str) -> list:
    variants = [name]
    bracket_match = re.search(r"\((.+?)\)", name)
    if bracket_match:
        inside = bracket_match.group(1).strip()
        if len(inside.split()) > 1 or len(inside) > 3:
            outside = re.sub(r"\(.*?\)", "", name).strip()
            variants.extend([inside, outside])
    return list(set(variants))

# ============================================================
# MAIN MATCHING FUNCTION
# ============================================================

def find_best_match(
    main_name: str,
    cleaned_choices: list,
    original_choices: list,
    threshold: float = 60,
) -> tuple:
    if not main_name:
        return None, 0.0, "No Match"

    parts = []
    for segment in str(main_name).split("/"):
        parts.extend(expand_name_variants(segment.strip()))

    best_match = None
    best_score = 0.0
    best_idx   = None

    for part in parts:
        main_clean = strip_noise(clean_text(part))
        if not main_clean:
            continue

        local_cutoff = threshold
        if " " not in main_clean:
            local_cutoff = threshold - 10  # allow phonetic candidates

        result = process.extractOne(
            main_clean,
            cleaned_choices,
            scorer=final_match_score,
            score_cutoff=local_cutoff,
        )

        if result:
            _, score, idx = result
            if score > best_score:
                best_score = score
                best_match = original_choices[idx]
                best_idx   = idx

    # Post-match validation
    if best_match and best_score >= threshold and best_idx is not None:

        # Discriminator check on full cleaned text (pre-noise-strip)
        main_full  = clean_text(main_name) or ""
        match_full = clean_text(original_choices[best_idx]) or ""

        if has_discriminator_conflict(main_full, match_full):
            return None, 0.0, "No Match (Discriminator Conflict)"

        # Minimum shared token check
        # Minimum shared token / brand validation
        main_stripped  = strip_noise(main_full)  or ""
        match_stripped = cleaned_choices[best_idx] or ""

        def has_core_overlap(a: str, b: str) -> bool:
            a_tokens = set(a.split())
            b_tokens = set(b.split())

            # Direct token overlap
            if a_tokens & b_tokens:
                return True

            # Collapsed overlap (FIRSTCRY vs FIRST CRY)
            a_col = collapse_spaces(a)
            b_col = collapse_spaces(b)
            return a_col == b_col or a_col in b_col or b_col in a_col


        # ✅ APPLY THE LOGIC ✅
        if not has_core_overlap(main_stripped, match_stripped):

            # ✅ Phonetic single-token override (Haya ↔ Hayaa)
            if is_phonetic_single_token_match(main_stripped, match_stripped):
                return best_match, round(best_score, 2), "High Confidence"

            # ✅ Strong single-token similarity override
            if is_single_token_close_match(main_stripped, match_stripped):
                return best_match, round(best_score, 2), "High Confidence"

            # ❌ Otherwise reject
            if fuzz.token_set_ratio(main_stripped, match_stripped) < 60:
                return None, 0.0, "No Match"



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
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")

    st.dataframe(df.head())

    st.sidebar.header("⚙ Match Configuration")
    source_col = st.sidebar.selectbox("Select Source Column", df.columns, index=0)
    target_col = st.sidebar.selectbox("Select Target Column", df.columns, index=1)
    threshold  = st.sidebar.slider("Match Threshold (%)", min_value=0, max_value=100, value=80)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Scoring Method")
    st.sidebar.markdown(
        "**Hybrid LCS + Core Anchor**\n\n"
        "- Space-collapsed LCS handles *FirstCry* vs *First Cry*\n"
        "- Core-anchor forgives suffix words (*Jeddah*, *App*, *Alkhalej*)\n"
        "- Soft penalty (4%/token) keeps suffix-heavy names matchable\n"
        "- Discriminator guard blocks *Bajaj Finance* ↔ *Bajaj Housing Finance* "
        "and *Application Co* ↔ *Recruitment Co*"
    )

    if st.button("🚀 Run Matching"):

        source_names = df[source_col].dropna().tolist()
        target_names = df[target_col].dropna().tolist()

        target_clean = [strip_noise(clean_text(x)) or "" for x in target_names]

        rows = []
        with st.spinner("Running matching…"):
            for name in source_names:
                match, score, mtype = find_best_match(
                    name, target_clean, target_names, threshold
                )
                rows.append({
                    source_col:     name,
                    "Matched Name": match,
                    "Score":        score,
                    "Match Type":   mtype,
                })

        out_df = pd.DataFrame(rows)

        target_lookup_set = set(df[target_col].astype(str))
        out_df["Excel Match Type"] = out_df[source_col].apply(
            lambda x: "Exact Match" if str(x) in target_lookup_set else "No Match"
        )

        st.success("✅ Matching Completed")
        st.dataframe(out_df)

        st.markdown("## 📊 Match Summary Dashboard")

        total_records       = len(out_df)
        total_fuzzy_matches = (out_df["Match Type"] == "High Confidence").sum()
        total_excel_matches = (out_df["Excel Match Type"] == "Exact Match").sum()
        fuzzy_match_rate    = round((total_fuzzy_matches / total_records) * 100, 2) if total_records else 0
        excel_match_rate    = round((total_excel_matches / total_records) * 100, 2) if total_records else 0

        st.warning(
            "⚠️ **Important Note:** Matches are subject to manual scrutiny. "
            "Please review results carefully."
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📄 Total Records",      total_records)
        k2.metric("🤖 Fuzzy Matches",       total_fuzzy_matches)
        k3.metric("📊 Excel Exact Matches", total_excel_matches)
        k4.metric("📈 Fuzzy Match Rate",    f"{fuzzy_match_rate}%")

        st.info(f"📊 Excel Exact Match Rate (VLOOKUP/XLOOKUP): {excel_match_rate}%")

        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "matched_results.csv", "text/csv")
