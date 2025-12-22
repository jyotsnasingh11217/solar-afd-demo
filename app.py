import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Solar-AFD Demo", layout="wide")

DEFAULT_URL = "https://forecast.weather.gov/product.php?site=TWC&issuedby=TWC&product=AFD"

def fetch_afd_text(url: str) -> str:
    headers = {
        "User-Agent": "Solar-AFD-Demo/1.0 (contact: your_email@example.com)"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pre = soup.find("pre")
    if not pre:
        raise RuntimeError("Could not find the AFD text (<pre>) on that page.")
    return pre.get_text("\n").strip()

def split_into_sections(raw: str):
    raw = raw.replace("\r\n", "\n")
    pattern = re.compile(r"^\.(?P<name>[A-Z /]+)\.\.\.", re.MULTILINE)
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

class SimpleRAG:
    def __init__(self, sections):
        self.sections = sections
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([t for _, t in sections])

    def retrieve(self, query: str, k: int = 3):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.sections[i][0], self.sections[i][1]) for i in idxs]

def solar_extract(raw: str):
    t = raw.lower()
    def has_any(words): 
        return [w for w in words if w in t]

    probs = set()
    for m in re.finditer(r"\b(\d{1,3})\s*[-to]+\s*(\d{1,3})\s*(%|percent)\b", t):
        probs.add(f"{m.group(1)}-{m.group(2)}%")
    for m in re.finditer(r"\b(\d{1,3})\s*(%|percent)\b", t):
        probs.add(f"{m.group(1)}%")

    cloud = has_any(["cloud", "cirrus", "overcast", "mid and high level", "cloud cover", "mostly clear"])
    aerosol = has_any(["haze", "smoke", "dust", "aerosol"])
    precip = has_any(["rain", "showers", "precip", "qpf", "pops", "thunderstorms"])
    conf = has_any(["confidence", "agreement", "uncertain", "murky", "likely", "possible"])

    timing = []
    for phrase in ["today","tonight","tuesday","wednesday","thursday","friday","saturday","sunday",
                   "christmas eve","christmas day","next week","next weekend","morning","afternoon","evening"]:
        if phrase in t:
            timing.append(phrase)

    return {
        "probabilities": sorted(probs),
        "cloud_signals": cloud,
        "aerosol_signals": aerosol,
        "precip_signals": precip,
        "confidence_signals": conf,
        "timing_cues": sorted(set(timing))
    }

st.title("Solar-AFD Demo (Tucson)")
st.caption("Reads the latest NWS Area Forecast Discussion (AFD) and answers questions using retrieval + cited text. No weather prediction.")

url = st.text_input("AFD link", value=DEFAULT_URL)

col1, col2 = st.columns([1, 1])
load = col1.button("Load latest report")
show_raw = col2.checkbox("Show full raw AFD text", value=False)

if "afd_text" not in st.session_state:
    st.session_state.afd_text = ""
    st.session_state.sections = []
    st.session_state.rag = None
    st.session_state.source_url = ""

if load:
    try:
        raw = fetch_afd_text(url)
        sections = split_into_sections(raw)
        st.session_state.afd_text = raw
        st.session_state.sections = sections
        st.session_state.rag = SimpleRAG(sections)
        st.session_state.source_url = url
        st.success("Loaded AFD successfully.")
    except Exception as e:
        st.error(f"Failed to load AFD: {e}")

if st.session_state.afd_text:
    info = solar_extract(st.session_state.afd_text)

    st.subheader("Extracted solar-relevant signals (baseline)")
    st.write(info)

    if show_raw:
        st.subheader("Raw AFD text")
        st.text(st.session_state.afd_text)

    st.subheader("Ask a question")
    q = st.text_input("Question", value="When does cloud cover increase?")
    k = st.slider("How many evidence chunks to retrieve?", min_value=1, max_value=5, value=3)

    if st.button("Answer (with evidence)"):
        hits = st.session_state.rag.retrieve(q, k=k)
        st.markdown("### Evidence (retrieved sections)")
        for score, name, text in hits:
            snippet = re.sub(r"\s+", " ", text)[:700]
            st.markdown(f"**{name}** (score: {score:.3f})")
            st.write(snippet + ("..." if len(text) > 700 else ""))

        st.markdown("### Source link")
        st.write(st.session_state.source_url)
else:
    st.info("Click **Load latest report** to start.")
