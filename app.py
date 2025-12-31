import re
import io
import math
import json
import random
from collections import Counter, defaultdict

import streamlit as st

# =========================
# Optional deps
# =========================
PYPDF_OK = True
try:
    from pypdf import PdfReader
except Exception:
    PYPDF_OK = False

PPTX_OK = True
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except Exception:
    PPTX_OK = False


# =========================
# Basic Spanish stopwords (compact)
# =========================
STOPWORDS_ES = {
    "a","acá","ahí","al","algo","algunos","ante","antes","apenas","aquí","así","aun","aunque",
    "bajo","bien","cada","casi","como","con","contra","cual","cuales","cuando","cuanto","de",
    "del","desde","donde","dos","el","ella","ellas","ello","ellos","en","entre","era","es",
    "esa","esas","ese","eso","esos","esta","está","están","estas","este","esto","estos","fue",
    "ha","hace","hacia","han","hasta","hay","incluso","la","las","le","les","lo","los","más",
    "me","menos","mi","mis","mucho","muy","nada","ni","no","nos","nuestra","nuestro","o","os",
    "otra","otro","para","pero","poco","por","porque","que","qué","quien","quienes","se","sea",
    "ser","si","sí","sin","sobre","solo","su","sus","también","tan","tanto","te","tener","tiene",
    "todo","todos","tu","tus","un","una","uno","unos","y","ya","yo"
}


# =========================
# Text utilities
# =========================
def clean_text(t):
    t = (t or "").replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def normalize_token(w):
    w = w.lower()
    w = re.sub(r"^[^\wáéíóúüñ]+|[^\wáéíóúüñ]+$", "", w, flags=re.UNICODE)
    return w

def tokenize_words(text):
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text or "", flags=re.UNICODE)
    tokens = []
    for w in words:
        nw = normalize_token(w)
        if not nw:
            continue
        if nw in STOPWORDS_ES:
            continue
        if len(nw) <= 2:
            continue
        tokens.append(nw)
    return tokens

def split_sentences(text):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9])", text)
    if len(parts) <= 1:
        parts = re.split(r"\s{2,}|;\s+", text)
    return [p.strip() for p in parts if p and p.strip()]

def pdf_to_text(file_bytes):
    if not PYPDF_OK:
        raise RuntimeError("PDF no disponible: instala pypdf (ver requirements.txt).")
    bio = io.BytesIO(file_bytes)
    reader = PdfReader(bio)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n\n".join(pages))


# =========================
# Classic NLP scoring
# =========================
def tfidf_sentence_scores(sentences):
    if not sentences:
        return [], {}

    sent_tokens = [tokenize_words(s) for s in sentences]
    df = Counter()
    for toks in sent_tokens:
        for t in set(toks):
            df[t] += 1

    N = max(1, len(sentences))
    idf = {}
    for t, c in df.items():
        idf[t] = math.log((N + 1) / (c + 1)) + 1.0

    scores = []
    for toks in sent_tokens:
        if not toks:
            scores.append(0.0)
            continue
        tf = Counter(toks)
        score = 0.0
        for t, f in tf.items():
            score += (f / len(toks)) * idf.get(t, 0.0)
        scores.append(score)

    return scores, idf

def pick_top_sentences(sentences, max_sentences):
    if not sentences:
        return []
    scores, _ = tfidf_sentence_scores(sentences)
    idx = list(range(len(sentences)))
    idx.sort(key=lambda i: scores[i], reverse=True)
    chosen = sorted(idx[: max_sentences])
    return [sentences[i] for i in chosen]

def top_keywords(text, k=12):
    toks = tokenize_words(text)
    if not toks:
        return []
    c = Counter(toks)
    ranked = sorted(c.items(), key=lambda x: (x[1], len(x[0]) >= 4, len(x[0])), reverse=True)
    out = []
    for w, _ in ranked:
        if w not in out:
            out.append(w)
        if len(out) >= k:
            break
    return out

def find_def_sentence(term, sentences):
    patt = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    candidates = [s for s in sentences if patt.search(s)]
    if not candidates:
        return None
    cues = ("es ", "son ", "se define", "consiste", "significa", "se entiende")
    def_like = [s for s in candidates if any(c in s.lower() for c in cues)]
    return (def_like[0] if def_like else candidates[0]).strip()

def extract_procedure(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    proc = []
    step_like = re.compile(r"^(\d+[\)\.\-]|•|\-)\s+|^(primero|segundo|tercero|después|luego|finalmente)\b", re.IGNORECASE)
    for l in lines:
        if step_like.search(l):
            proc.append(l)

    if not proc:
        sents = split_sentences(text)
        for s in sents:
            if "paso" in s.lower() or "pasos" in s.lower():
                proc.append(s.strip())
                if len(proc) >= 6:
                    break
    return proc[:10]


# =========================
# Generators (heuristics)
# =========================
def generate_questions(text, n=5):
    text = clean_text(text)
    sents = split_sentences(text)
    kws = top_keywords(text, k=20)
    questions = []

    for term in kws:
        if len(questions) >= n:
            break
        questions.append(f"¿Cómo definirías {term} según el texto?")

    cues = ["porque", "por qué", "para", "objetivo", "consiste", "permite", "provoca", "evita"]
    for s in sents:
        if len(questions) >= n:
            break
        if any(c in s.lower() for c in cues):
            questions.append(f"Explica esta idea con tus palabras: “{s}”")

    return questions[:n]

def generate_notes(text, detail):
    text = clean_text(text)
    if not text:
        return "No hay contenido para generar apuntes."

    sentences = split_sentences(text)
    kws = top_keywords(text, k=12)

    n = 10
    if detail == "breve":
        n = 6
    elif detail == "exhaustivo":
        n = 16

    key_sents = pick_top_sentences(sentences, n)
    proc = extract_procedure(text)

    title = " ".join([k.capitalize() for k in kws[:3]]) if kws else "Apuntes"
    md = []
    md.append(f"# {title}\n")

    md.append("## Ideas clave")
    for s in key_sents:
        md.append(f"- {s}")

    if proc:
        md.append("\n## Procedimiento")
        for p in proc:
            md.append(f"- {p}")

    md.append("\n## Glosario (términos clave)")
    for term in kws[:10]:
        ds = find_def_sentence(term, sentences)
        if ds:
            md.append(f"- **{term}**: {ds}")
        else:
            md.append(f"- **{term}**: (no consta una definición explícita en el texto)")

    md.append("\n## Resumen en 10 líneas")
    sum_sents = pick_top_sentences(sentences, 10)
    for s in sum_sents[:10]:
        md.append(f"- {s}")

    md.append("\n## Preguntas de repaso (sin respuestas)")
    for q in generate_questions(text, n=5):
        md.append(f"- {q}")

    return "\n".join(md).strip()


# =========================
# PPTX generation
# =========================
def parse_md_sections(md_text):
    """
    Very tolerant Markdown parser:
    - "# " -> title
    - "## " -> section
    - "- " -> bullets
    """
    md_text = (md_text or "").replace("\r\n", "\n").strip()
    title = "Presentación"
    sections = []
    current = None

    for line in md_text.split("\n"):
        line = line.rstrip()
        if line.startswith("# "):
            title = line[2:].strip() or title
        elif line.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": line[3:].strip() or "Sección", "bullets": []}
        elif line.startswith("- "):
            if not current:
                current = {"title": "Contenido", "bullets": []}
            b = line[2:].strip()
            if b:
                # remove markdown bold ** **
                b = b.replace("**", "")
                current["bullets"].append(b)
        else:
            # ignore other lines
            pass

    if current:
        sections.append(current)

    if not sections:
        sections = [{"title": "Contenido", "bullets": [clean_text(md_text)[:500]]}]

    return title, sections

def add_bullets_to_slide(slide, bullets, font_size=22):
    body = slide.shapes.placeholders[1].text_frame
    body.clear()
    first = True
    for b in bullets:
        if first:
            p = body.paragraphs[0]
            first = False
        else:
            p = body.add_paragraph()
        p.text = b
        p.level = 0
        p.font.size = Pt(font_size)

def notes_to_pptx_bytes(notes_md, deck_title="Presentación", max_bullets_per_slide=6):
    if not PPTX_OK:
        raise RuntimeError("Falta python-pptx. Añádelo a requirements.txt")

    title, sections = parse_md_sections(notes_md)
    if deck_title:
        title = deck_title

    prs = Presentation()

    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = "Generado automáticamente desde apuntes"

    # Agenda
    agenda = prs.slides.add_slide(prs.slide_layouts[1])
    agenda.shapes.title.text = "Agenda"
    agenda_items = [s["title"] for s in sections][:10]
    add_bullets_to_slide(agenda, agenda_items, font_size=24)

    # Content slides (split long bullet lists)
    for sec in sections:
        bullets = sec.get("bullets", []) or ["(Sin puntos detectados en esta sección)"]
        chunks = [bullets[i:i + max_bullets_per_slide] for i in range(0, len(bullets), max_bullets_per_slide)]

        for j, chunk in enumerate(chunks):
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            t = sec["title"] if j == 0 else f"{sec['title']} (cont.)"
            slide.shapes.title.text = t
            add_bullets_to_slide(slide, chunk, font_size=22)

    # Export to bytes
    bio = io.BytesIO()
    prs.save(bio)
    bio.seek(0)
    return bio.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title="StudyWave (sin API) + PPTX", page_icon="🧠", layout="wide")
st.title("🧠 StudyWave — Apuntes + Presentación (.pptx) (SIN API)")

with st.sidebar:
    st.header("📥 Fuente")
    source = st.radio("Selecciona fuente", ["Texto", "PDF"], index=1)

    st.divider()
    st.header("🎯 Salidas")
    want_notes = st.checkbox("Generar apuntes", True)
    want_pptx = st.checkbox("Generar presentación (.pptx)", True)

    st.divider()
    detail = st.selectbox("Nivel de apuntes", ["breve", "medio", "exhaustivo"], index=1)
    max_bullets = st.slider("Máx. bullets por diapositiva", 4, 10, 6, 1)

    st.caption("Nota: La PPTX se genera a partir de los apuntes (Markdown).")

content = ""
doc_name = "apuntes"

if source == "Texto":
    content = st.text_area("Pega aquí tus apuntes", height=260, placeholder="Pega apuntes, temas, etc.")
else:
    if not PYPDF_OK:
        st.warning("Para leer PDFs necesitas `pypdf` (solo lector).")
    pdf = st.file_uploader("Sube un PDF", type=["pdf"])
    if pdf is not None:
        doc_name = re.sub(r"\.pdf$", "", pdf.name, flags=re.IGNORECASE) or "apuntes"
        try:
            b = pdf.read()
            if not PYPDF_OK:
                st.error("Falta `pypdf`. Añádelo a requirements.txt.")
                content = ""
            else:
                content = pdf_to_text(b)
                st.success(f"PDF cargado: {pdf.name} ({len(content):,} caracteres extraídos)")
        except Exception as e:
            st.error(f"Error leyendo PDF: {e}")

st.divider()
go = st.button("✨ Generar", type="primary", use_container_width=True)

if go:
    if not (content or "").strip():
        st.error("No hay contenido. Añade texto o sube un PDF.")
        st.stop()

    notes_md = ""
    if want_notes:
        with st.spinner("Generando apuntes..."):
            notes_md = generate_notes(content, detail)
        st.success("Apuntes listos ✅")
        st.markdown(notes_md)
        st.download_button(
            "⬇️ Descargar apuntes (.md)",
            data=notes_md.encode("utf-8"),
            file_name=f"{doc_name}_apuntes.md",
            mime="text/markdown",
        )
    else:
        # Si no genera apuntes, creamos una “pseudo-nota” con el texto original
        notes_md = f"# {doc_name}\n\n## Contenido\n- " + "\n- ".join(split_sentences(content)[:30])

    if want_pptx:
        if not PPTX_OK:
            st.warning("No puedo generar PPTX porque falta `python-pptx`. Añádelo a requirements.txt.")
        else:
            with st.spinner("Generando presentación (.pptx)..."):
                pptx_bytes = notes_to_pptx_bytes(
                    notes_md,
                    deck_title=doc_name,
                    max_bullets_per_slide=max_bullets
                )
            st.success("Presentación lista ✅")
            st.download_button(
                "⬇️ Descargar presentación (.pptx)",
                data=pptx_bytes,
                file_name=f"{doc_name}_presentacion.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
