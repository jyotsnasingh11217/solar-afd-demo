# app.py
# Solar-AFD Demo (Clean UI)
# - Pulls latest NWS AFD from a live link
# - Chunks into sections
# - Retrieves relevant evidence (TF-IDF)
# - (Optional) Uses an LLM to answer like an assistant, grounded only in evidence
#
# To enable LLM answers on Streamlit Cloud:
# 1) Add `openai` to requirements.txt
# 2) In Streamlit Cloud -> App -> Settings -> Secrets, add:
#    OPENAI_API_KEY = "your_key_here"

import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_URL = "https://forecast.weather.gov/product.php?site=TWC&issuedby=TWC&product=AFD"


# ----------------------------
# Fetch + parse AFD text
# ----------------------------
def fetch_afd_text(url: str) -> str:
    headers = {
        "User-Agent": "Solar-AFD-Demo/1.0 (contact: jyotsnasingh11217)"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    pre = soup.find("pre")
    if not pre:
        raise RuntimeError("Could not find AFD text (<pre>) on this page. The page layout may have changed.")
    return pre.get_text("\n").strip()


def split_into_sections(raw: str):
    raw = raw.replace("\r\n", "\n")

    # Match headers like ".SYNOPSIS..." ".DISCUSSION..." etc.
    pattern = re.compile(r"^\.(?P<name>[A-Z0-9 /]+)\.\.\.", re.MULTILINE)
    matches = list(pattern.finditer(raw))

    if not matches:
        return [("FULL_TEXT", raw)]

    sections = []
    for i, m in enumerate(matches):
        name = m.group("name").strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections.append((name, raw[start:end].strip()))
    return sections


# ----------------------------
# Lightweight extraction (signals)
# ----------------------------
def solar_extract(raw: str):
    t = raw.lower()

    def has_any(words):
        return [w for w in words if w in t]

    # Probabilities: grab ranges and percents.
    # NOTE: this is intentionally simple; it will capture humidity ranges too.
    probs = set()
    for m in re.finditer(r"\b(\d{1,3})\s*(?:-|to)\s*(\d{1,3})\s*(%|percent)\b", t):
        probs.add(f"{m.group(1)}-{m.group(2)}%")
    for m in re.finditer(r"\b(\d{1,3})\s*(%|percent)\b", t):
        probs.add(f"{m.group(1)}%")

    cloud_terms = [
        "cloud", "cloud cover", "mid and high level", "cirrus", "mostly clear", "overcast",
        "mid level", "high level"
    ]
    aerosol_terms = ["haze", "smoke", "dust", "aerosol"]
    precip_terms = ["rain", "showers", "precip", "qpf", "pops", "thunderstorms", "snow"]
    confidence_terms = ["confidence", "agreement", "uncertain", "murky", "likely", "possible"]

    timing = []
    for phrase in [
        "today", "tonight",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "christmas eve", "christmas day", "new year",
        "next week", "next weekend",
        "morning", "afternoon", "evening", "overnight"
    ]:
        if phrase in t:
            timing.append(phrase)

    # Temperature anomaly cue (optional)
    temp_anom = []
    m = re.search(r"(\d{1,2})\s*to\s*(\d{1,2})\s*degrees\s*above\s*normal", t)
    if m:
        temp_anom.append(f"+{m.group(1)} to +{m.group(2)} above normal")
    if "record" in t:
        temp_anom.append("record/near-record mentioned")

    # Simple heuristic "quick take"
    dni_risk = 0
    variability_risk = 0
    stability = 0

    if "mid and high level" in t or "cloud cover" in t or "cirrus" in t:
        dni_risk += 2
    if has_any(precip_terms):
        variability_risk += 2
    if has_any(aerosol_terms):
        dni_risk += 1
        variability_risk += 1
    if "dry" in t and not has_any(precip_terms):
        stability += 2

    def level(x):
        if x <= 0:
            return "low"
        if x <= 2:
            return "medium"
        return "high"

    return {
        "probabilities": sorted(probs),
        "cloud_signals": has_any(cloud_terms),
        "aerosol_signals": has_any(aerosol_terms),
        "precip_signals": has_any(precip_terms),
        "confidence_signals": has_any(confidence_terms),
        "timing_cues": sorted(set(timing)),
        "temperature_anomaly_cues": temp_anom,
        "solar_tags": {
            "dni_risk_level": level(dni_risk),
            "variability_risk_level": level(variability_risk),
            "overall_stability_level": level(stability),
        }
    }


# ----------------------------
# Retrieval (TF-IDF) = “RAG-style” evidence picker
# ----------------------------
class SimpleRAG:
    def __init__(self, sections):
        self.sections = sections
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([text for _, text in sections])

    def retrieve(self, query: str, k: int = 3):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.sections[i][0], self.sections[i][1]) for i in idxs]


# ----------------------------
# Optional LLM Answer (grounded in evidence only)
# ----------------------------
def llm_available() -> bool:
    return "OPENAI_API_KEY" in st.secrets and bool(st.secrets.get("OPENAI_API_KEY"))


def llm_answer(question: str, hits, source_url: str) -> str:
    """
    hits: list of tuples (score, section_name, section_text)
    """
    try:
        from openai import OpenAI
    except Exception:
        return "LLM mode is not available because the `openai` package is not installed."

    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

    evidence_blocks = []
    for _, name, text in hits:
        clean = re.sub(r"\s+", " ", text).strip()
        # Keep each evidence block bounded to reduce token use
        evidence_blocks.append(f"[{name}] {clean[:2000]}")

    prompt = f"""
You are a careful assistant. Answer the user's question using ONLY the evidence below.

Rules:
- If the answer is not stated in the evidence, say: "Not stated in this report."
- Do not guess.
- Keep the answer short: 3–6 bullets max.
- After the bullets, provide 2–4 supporting quotes with section tags (verbatim snippets).
- Finish with: Source: <url>

User question: {question}

EVIDENCE:
{chr(10).join(evidence_blocks)}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer only from provided evidence. No hallucinations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    if "Source:" not in text:
        text += f"\n\nSource: {source_url}"
    return text


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Solar-AFD Demo", layout="wide")

st.title("Solar-AFD Demo (Tucson)")
st.caption("Reads the latest NWS Area Forecast Discussion (AFD) and answers questions using evidence from the report (retrieval + optional LLM).")

with st.sidebar:
    st.header("Settings")
    url = st.text_input("AFD link", value=DEFAULT_URL)
    k = st.slider("Evidence chunks", 1, 5, 3)
    use_llm = st.toggle("LLM answer (grounded)", value=False, help="Requires OPENAI_API_KEY secret + openai in requirements.txt")
    st.markdown("---")
    st.caption("Tip: Best questions mention clouds, timing, confidence, rain chances, or uncertainty.")

# Session state init
if "afd_text" not in st.session_state:
    st.session_state.afd_text = ""
    st.session_state.sections = []
    st.session_state.rag = None
    st.session_state.source_url = ""
    st.session_state.info = None

colA, colB = st.columns([1, 1])
with colA:
    load = st.button("Load latest report", type="primary")
with colB:
    show_source_only = st.toggle("Hide technical details", value=True)

if load:
    try:
        raw = fetch_afd_text(url)
        sections = split_into_sections(raw)
        st.session_state.afd_text = raw
        st.session_state.sections = sections
        st.session_state.rag = SimpleRAG(sections)
        st.session_state.source_url = url
        st.session_state.info = solar_extract(raw)
        st.success("Loaded AFD successfully.")
    except Exception as e:
        st.error(f"Failed to load AFD: {e}")

if not st.session_state.afd_text:
    st.info("Click **Load latest report** to start.")
    st.stop()

# Clean tabs
tab_ask, tab_summary, tab_advanced = st.tabs(["Ask", "Solar summary", "Advanced"])

with tab_ask:
    st.subheader("Ask a question")
    q = st.text_input("Question", value="Will it rain today?")

    ask_btn = st.button("Answer", type="secondary")
    if ask_btn:
        hits = st.session_state.rag.retrieve(q, k=k)

        st.markdown("### Answer")
        if use_llm:
            if not llm_available():
                st.warning("LLM is not configured. Add OPENAI_API_KEY in Streamlit Secrets and `openai` in requirements.txt.")
                use_llm = False

        if use_llm:
            with st.spinner("Generating grounded answer..."):
                st.write(llm_answer(q, hits, st.session_state.source_url))
        else:
            # Evidence-first response (non-LLM)
            st.write("Top evidence from the report:")

            for score, name, text in hits:
                snippet = re.sub(r"\s+", " ", text).strip()
                snippet = snippet[:500] + ("..." if len(snippet) > 500 else "")
                st.markdown(f"**{name}** (score: {score:.3f})")
                st.write(snippet)

            st.caption("Source link")
            st.write(st.session_state.source_url)

        # Keep evidence optionally visible (clean UI)
        if not show_source_only:
            with st.expander("Evidence (full retrieved sections)", expanded=False):
                for score, name, text in hits:
                    st.markdown(f"**{name}** (score: {score:.3f})")
                    st.text(text)

with tab_summary:
    st.subheader("Solar quick take (auto-extracted)")
    info = st.session_state.info or {}

    tags = info.get("solar_tags", {})
    st.write(
        {
            "DNI risk": tags.get("dni_risk_level"),
            "Variability risk": tags.get("variability_risk_level"),
            "Overall stability": tags.get("overall_stability_level"),
        }
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What the report mentions**")
        st.write({
            "Cloud signals": info.get("cloud_signals", []),
            "Aerosol signals": info.get("aerosol_signals", []),
            "Precip signals": info.get("precip_signals", []),
        })
    with c2:
        st.markdown("**Timing + confidence**")
        st.write({
            "Timing cues": info.get("timing_cues", []),
            "Confidence signals": info.get("confidence_signals", []),
            "Probabilities (raw)": info.get("probabilities", []),
        })

    st.caption("Source link")
    st.write(st.session_state.source_url)

with tab_advanced:
    st.subheader("Advanced (transparency & debugging)")
    st.caption("This tab is optional for reviewers who want to see the raw text and the extracted structure.")

    with st.expander("Extracted signals (raw JSON)", expanded=False):
        st.json(st.session_state.info)

    with st.expander("Detected sections (preview)", expanded=False):
        for name, text in st.session_state.sections:
            st.markdown(f"**{name}**")
            preview = text[:1200] + ("..." if len(text) > 1200 else "")
            st.text(preview)

    with st.expander("Full raw AFD text", expanded=False):
        st.text(st.session_state.afd_text)
