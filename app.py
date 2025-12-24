import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Weather Report Interpretation (RAG)",
    layout="wide",
)

st.title("Accessible Interpretation of Weather Forecast Reports")
st.caption(
    "Evidence-grounded interpretation of National Weather Service forecast discussions "
    "using Retrieval-Augmented NLP. Not a forecast or advisory."
)

# ----------------------------
# Constants
# ----------------------------
AFD_URL = "https://forecast.weather.gov/product.php?site=TWC&issuedby=TWC&product=AFD"

# ----------------------------
# Utilities
# ----------------------------
def fetch_afd(url: str) -> str:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pre = soup.find("pre")
    return pre.get_text() if pre else ""

def split_sections(text: str):
    sections = {}
    current = "HEADER"
    buf = []
    for line in text.splitlines():
        if line.startswith(".") and line.endswith("..."):
            sections[current] = "\n".join(buf).strip()
            current = line.strip(".").replace("...", "")
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return sections

def retrieve_sections(question, sections, k=3):
    names = list(sections.keys())
    docs = [sections[n] for n in names]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs + [question])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()

    ranked = sorted(
        zip(sims, names, docs),
        key=lambda x: x[0],
        reverse=True
    )
    return ranked[:k]

def llm_available():
    return bool(st.secrets.get("GEMINI_API_KEY"))

def llm_answer(question, hits, source_url):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "LLM is not configured. Showing evidence only."

    evidence_blocks = []
    for _, name, text in hits:
        clean = re.sub(r"\s+", " ", text).strip()
        evidence_blocks.append(f"[{name}] {clean[:1200]}")

    prompt = f"""
You translate technical weather forecast discussions into accessible language.

Rules:
- Use ONLY the evidence below.
- Do NOT guess or add information.
- If something is not stated, write: "Not stated in this report."
- Do NOT give advice (no yes/no picnic decisions).
- Output format:

Outdoor Conditions Summary:
- Rain:
- Sky:
- Wind:
- Alerts:

Answer to the user's question (1–2 sentences).

Evidence:
- 2–4 short quotes with [SECTION] tags.

End with:
Source: {source_url}

User question:
{question}

EVIDENCE:
{chr(10).join(evidence_blocks)}
""".strip()

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return (
            f"LLM unavailable ({type(e).__name__}). Showing evidence instead.\n\n"
            + "\n".join(
                f"- [{name}] {re.sub(r'\\s+', ' ', text)[:250]}..."
                for _, name, text in hits
            )
            + f"\n\nSource: {source_url}"
        )

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Controls")
    use_llm = st.checkbox("Use LLM (Gemini)", value=True)
    st.markdown("---")
    st.write("Data source:")
    st.write("NWS Tucson Area Forecast Discussion")

# ----------------------------
# Main workflow
# ----------------------------
if st.button("Load latest Tucson AFD"):
    with st.spinner("Fetching report..."):
        raw_text = fetch_afd(AFD_URL)
        sections = split_sections(raw_text)
        st.session_state["sections"] = sections
        st.session_state["source_url"] = AFD_URL
    st.success("AFD loaded successfully.")

if "sections" in st.session_state:
    question = st.text_input(
        "Ask a question (example: Will it rain today? Is it windy? Are there any alerts?)"
    )

    if question:
        hits = retrieve_sections(question, st.session_state["sections"], k=3)

        st.subheader("Answer")
        if use_llm and llm_available():
            st.write(llm_answer(question, hits, st.session_state["source_url"]))
        else:
            st.write("Top evidence from the report:\n")
            for score, name, text in hits:
                st.markdown(f"**{name} (score {score:.3f})**")
                st.write(text[:800] + "...")
            st.markdown(f"**Source:** {st.session_state['source_url']}")

        with st.expander("Evidence (full retrieved sections)"):
            for _, name, text in hits:
                st.markdown(f"### {name}")
                st.text(text)
