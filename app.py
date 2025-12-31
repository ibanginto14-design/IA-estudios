import re
import io
import math
import json
import random
from collections import Counter, defaultdict
from datetime import datetime

import streamlit as st

# -----------------------------
# Optional deps (safe flags)
# -----------------------------
PYPDF_OK = True
try:
    from pypdf import PdfReader
except Exception:
    PYPDF_OK = False

PPTX_OK = True
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
except Exception:
    PPTX_OK = False


# -----------------------------
# Compact Spanish stopwords
# -----------------------------
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


# -----------------------------
# Text utilities
# -----------------------------
def clean_text(t):
    t = (t or "").replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def normalize_token(w):
    w = (w or "").lower()
    w = re.sub(r"^[^\wáéíóúüñ]+|[^\wáéíóúüñ]+$", "", w, flags=re.UNICODE)
    return w

def tokenize_words(text):
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text or "", flags=re.UNICODE)
    out = []
    for w in words:
        w = normalize_token(w)
        if not w:
            continue
        if w in STOPWORDS_ES:
            continue
        if len(w) <= 2:
            continue
        out.append(w)
    return out

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
        raise RuntimeError("Falta pypdf. Añádelo a requirements.txt.")
    bio = io.BytesIO(file_bytes)
    reader = PdfReader(bio)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n\n".join(pages))


# -----------------------------
# Classic NLP scoring (TF-IDF)
# -----------------------------
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
        idf[t] = math.log((N + 1) / (c + 1)) + 1.0  # smoothed

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

def pick_top_sentences(sentences, n):
    if not sentences:
        return []
    scores, _ = tfidf_sentence_scores(sentences)
    idx = list(range(len(sentences)))
    idx.sort(key=lambda i: scores[i], reverse=True)
    chosen = sorted(idx[:n])
    return [sentences[i] for i in chosen]

def top_keywords(text, k=14):
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
    cues = (" es ", " son ", "se define", "consiste", "significa", "se entiende")
    def_like = [s for s in candidates if any(c in (" " + s.lower() + " ") for c in cues)]
    return (def_like[0] if def_like else candidates[0]).strip()

def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# -----------------------------
# Notes generator (Markdown structured)
# -----------------------------
def generate_notes_md(text, detail):
    text = clean_text(text)
    if not text:
        return "# Apuntes\n\n(No hay contenido)"

    sents = split_sentences(text)
    kws = top_keywords(text, k=14)

    if detail == "breve":
        n = 7
    elif detail == "exhaustivo":
        n = 18
    else:
        n = 12

    key_sents = pick_top_sentences(sents, n)
    sum10 = pick_top_sentences(sents, 10)

    title = " ".join([k.capitalize() for k in kws[:3]]) if kws else "Apuntes"

    md = []
    md.append(f"# {title}")
    md.append("")
    md.append("## Ideas clave")
    for s in key_sents:
        md.append(f"- {s}")

    md.append("")
    md.append("## Glosario")
    for term in (kws[:10] if kws else []):
        ds = find_def_sentence(term, sents)
        if ds:
            md.append(f"- **{term}**: {ds}")
        else:
            md.append(f"- **{term}**: (no consta una definición explícita)")

    md.append("")
    md.append("## Resumen en 10 líneas")
    for s in sum10[:10]:
        md.append(f"- {s}")

    md.append("")
    md.append("## Preguntas de repaso")
    md += [f"- ¿Cómo explicarías **{k}** con tus palabras?" for k in (kws[:5] if kws else ["el tema"])]

    return "\n".join(md).strip()


# -----------------------------
# Flashcards generator (JSON + Anki TSV)
# -----------------------------
def generate_flashcards(text, n_cards):
    text = clean_text(text)
    sents = split_sentences(text)
    kws = top_keywords(text, k=max(20, n_cards))
    cards = []

    # 1) definitional cards
    for term in kws:
        if len(cards) >= n_cards:
            break
        ds = find_def_sentence(term, sents)
        if ds:
            cards.append({
                "front": f"¿Qué es {term}?",
                "back": ds,
                "tags": [term]
            })

    # 2) cloze cards
    if len(cards) < n_cards:
        pool = [s for s in sents if len(tokenize_words(s)) >= 7]
        random.shuffle(pool)
        for s in pool:
            if len(cards) >= n_cards:
                break
            toks = tokenize_words(s)
            present = [t for t in toks if t in kws]
            if not present:
                continue
            target = present[0]
            cloze = re.sub(rf"\b{re.escape(target)}\b", "_____", s, flags=re.IGNORECASE)
            cards.append({
                "front": f"Completa: {cloze}",
                "back": f"Palabra: **{target}**\n\nFrase original: {s}",
                "tags": [target]
            })

    return {"flashcards": cards[:n_cards]}

def flashcards_to_anki_tsv(fc_json):
    lines = []
    for c in fc_json.get("flashcards", []):
        front = (c.get("front", "") or "").replace("\t", " ").strip()
        back = (c.get("back", "") or "").replace("\t", " ").strip()
        tags = " ".join(c.get("tags", []) or [])
        # Anki: Front<TAB>Back<TAB>Tags (optional)
        lines.append(f"{front}\t{back}\t{tags}")
    return "\n".join(lines)


# -----------------------------
# Mindmap generator (Mermaid)
# -----------------------------
def generate_mindmap_mermaid(text, max_nodes=60):
    text = clean_text(text)
    if not text:
        return "```mermaid\nmindmap\n  root((Sin contenido))\n```"

    sents = split_sentences(text)
    kws = top_keywords(text, k=18)
    root = kws[0].capitalize() if kws else "Tema"

    # co-occurrence graph (keyword-keyword)
    co = defaultdict(Counter)
    for s in sents:
        toks = set(tokenize_words(s))
        present = [k for k in kws if k in toks]
        for a in present:
            for b in present:
                if a != b:
                    co[a][b] += 1

    # build mindmap
    lines = ["```mermaid", "mindmap", f"  root(({root}))"]
    node_count = 1

    top_branches = kws[1:7] if len(kws) > 1 else []
    for k in top_branches:
        if node_count >= max_nodes:
            break
        lines.append(f"    {k}")
        node_count += 1
        for sub, _ in co[k].most_common(3):
            if node_count >= max_nodes:
                break
            lines.append(f"      {sub}")
            node_count += 1

    lines.append("```")
    return "\n".join(lines)


# -----------------------------
# PPTX "pro" generator
# -----------------------------
def _rgb(hexstr):
    hexstr = hexstr.lstrip("#")
    return RGBColor(int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16))

def _add_top_bar(slide, color="#ff4b4b"):
    # bar across top
    left = Inches(0)
    top = Inches(0)
    width = Inches(13.333)  # widescreen default
    height = Inches(0.28)
    shape = slide.shapes.add_shape(1, left, top, width, height)  # 1 = MSO_SHAPE.RECTANGLE
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = _rgb(color)
    shape.line.fill.background()

def _set_title(slide, title, color="#111111"):
    title_shape = slide.shapes.title
    title_shape.text = title
    for p in title_shape.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(38)
            r.font.bold = True
            r.font.color.rgb = _rgb(color)

def _add_footer(slide, text_left="", text_right="", color="#666666"):
    # small footer text
    left = Inches(0.6)
    top = Inches(7.05)
    width = Inches(12.2)
    height = Inches(0.3)
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text_left
    p.font.size = Pt(10)
    p.font.color.rgb = _rgb(color)

    # right aligned
    tx2 = slide.shapes.add_textbox(Inches(10.6), top, Inches(2.6), height)
    tf2 = tx2.text_frame
    tf2.clear()
    p2 = tf2.paragraphs[0]
    p2.text = text_right
    p2.font.size = Pt(10)
    p2.font.color.rgb = _rgb(color)
    p2.alignment = 2  # right

def _add_bullets(slide, bullets, font_size=22):
    # layout 1: Title and Content
    body = slide.shapes.placeholders[1].text_frame
    body.clear()
    for i, b in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = b
        p.level = 0
        p.font.size = Pt(font_size)

def _parse_notes_to_sections(notes_md):
    notes_md = (notes_md or "").replace("\r\n", "\n").strip()
    title = "Presentación"
    sections = []
    current = None

    for line in notes_md.split("\n"):
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
            b = line[2:].strip().replace("**", "")
            if b:
                current["bullets"].append(b)

    if current:
        sections.append(current)

    if not sections:
        sections = [{"title": "Contenido", "bullets": [clean_text(notes_md)[:500]]}]

    return title, sections

def notes_to_pptx_bytes(notes_md, deck_title, max_bullets_per_slide=6, accent="#ff4b4b"):
    if not PPTX_OK:
        raise RuntimeError("Falta python-pptx. Añádelo a requirements.txt.")

    title, sections = _parse_notes_to_sections(notes_md)
    if deck_title:
        title = deck_title

    prs = Presentation()
    # Force widescreen 13.333 x 7.5 (default in many installs already)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 1) Cover
    slide = prs.slides.add_slide(prs.slide_layouts[0])  # title slide
    _add_top_bar(slide, accent)
    _set_title(slide, title)
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = "Resumen automático • Flashcards • Mindmap"
    _add_footer(slide, text_left="StudyWave (sin API)", text_right=date_str)

    # 2) Agenda
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _add_top_bar(slide, accent)
    slide.shapes.title.text = "Agenda"
    agenda_items = [s["title"] for s in sections][:12]
    _add_bullets(slide, agenda_items, font_size=24)
    _add_footer(slide, text_left="Agenda", text_right="2", color="#777777")

    # 3+) Sections
    slide_no = 3
    for sec in sections:
        bullets = sec.get("bullets", []) or ["(Sin puntos detectados en esta sección)"]
        bullet_chunks = chunk_list(bullets, max_bullets_per_slide)

        for j, chunk in enumerate(bullet_chunks):
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            _add_top_bar(slide, accent)
            slide.shapes.title.text = sec["title"] if j == 0 else f"{sec['title']} (cont.)"
            _add_bullets(slide, chunk, font_size=22)
            _add_footer(slide, text_left=sec["title"], text_right=str(slide_no), color="#777777")
            slide_no += 1

    bio = io.BytesIO()
    prs.save(bio)
    bio.seek(0)
    return bio.getvalue()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="StudyWave PRO (sin API)", page_icon="🧠", layout="wide")
st.title("🧠 StudyWave PRO — Apuntes + PPTX + Flashcards + Mindmap (SIN API)")

with st.sidebar:
    st.subheader("Estado")
    st.write(f"pypdf (PDF): {'✅' if PYPDF_OK else '❌'}")
    st.write(f"python-pptx (PPTX): {'✅' if PPTX_OK else '❌'}")
    if not PYPDF_OK:
        st.caption("Para PDF: añade pypdf a requirements.txt")
    if not PPTX_OK:
        st.caption("Para PPTX: añade python-pptx a requirements.txt")

    st.divider()
    source = st.radio("Fuente", ["PDF", "Texto"], index=0)
    detail = st.selectbox("Nivel de apuntes", ["breve", "medio", "exhaustivo"], index=1)

    st.divider()
    st.subheader("Salidas")
    want_notes = st.checkbox("Apuntes", True)
    want_pptx = st.checkbox("PowerPoint PRO (.pptx)", True)
    want_flash = st.checkbox("Flashcards / Fichas", True)
    want_mindmap = st.checkbox("Mapa mental (Mermaid)", True)

    st.divider()
    accent = st.color_picker("Color acento PPT", "#ff4b4b")
    max_bullets = st.slider("Bullets por diapositiva", 4, 10, 6, 1)
    n_flash = st.slider("Nº flashcards", 5, 60, 20, 1)

content = ""
doc_name = "apuntes"

if source == "Texto":
    content = st.text_area("Pega tus apuntes", height=260, placeholder="Pega aquí tus apuntes o texto...")
else:
    pdf = st.file_uploader("Sube un PDF", type=["pdf"])
    if pdf is not None:
        doc_name = re.sub(r"\.pdf$", "", pdf.name, flags=re.IGNORECASE) or "apuntes"
        try:
            if not PYPDF_OK:
                st.error("No puedo leer PDF porque falta pypdf (requirements.txt).")
                content = ""
            else:
                content = pdf_to_text(pdf.read())
                st.success(f"Texto extraído: {len(content):,} caracteres")
                if len(content.strip()) < 300:
                    st.warning("He extraído muy poco texto. Si el PDF es escaneado (imagen), sin OCR no se puede leer.")
        except Exception as e:
            st.error("Error leyendo el PDF:")
            st.exception(e)
            content = ""

st.divider()
go = st.button("✨ Generar materiales", type="primary", use_container_width=True)

if go:
    try:
        if not clean_text(content):
            st.error("No hay texto para procesar.")
            st.stop()

        # 1) Notes
        notes_md = generate_notes_md(content, detail)

        colA, colB = st.columns([1, 1], gap="large")

        with colA:
            if want_notes:
                st.subheader("📚 Apuntes")
                st.markdown(notes_md)
                st.download_button(
                    "⬇️ Descargar apuntes (.md)",
                    data=notes_md.encode("utf-8"),
                    file_name=f"{doc_name}_apuntes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        # 2) Flashcards
        with colB:
            if want_flash:
                st.subheader("🃏 Flashcards / Fichas")
                fc = generate_flashcards(content, n_flash)
                st.caption("Formato JSON + export Anki (TSV).")
                st.json(fc)

                st.download_button(
                    "⬇️ Descargar flashcards (.json)",
                    data=json.dumps(fc, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{doc_name}_flashcards.json",
                    mime="application/json",
                    use_container_width=True,
                )
                anki_tsv = flashcards_to_anki_tsv(fc)
                st.download_button(
                    "⬇️ Descargar para Anki (.tsv)",
                    data=anki_tsv.encode("utf-8"),
                    file_name=f"{doc_name}_anki.tsv",
                    mime="text/tab-separated-values",
                    use_container_width=True,
                )

        st.divider()

        # 3) Mindmap
        if want_mindmap:
            st.subheader("🧩 Mapa mental (Mermaid)")
            mm = generate_mindmap_mermaid(content, max_nodes=60)
            st.code(mm, language="markdown")
            st.download_button(
                "⬇️ Descargar mindmap (.md)",
                data=mm.encode("utf-8"),
                file_name=f"{doc_name}_mindmap.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # 4) PPTX
        if want_pptx:
            st.subheader("📊 PowerPoint PRO")
            if not PPTX_OK:
                st.error("No puedo generar PPTX porque falta python-pptx (requirements.txt).")
            else:
                pptx_bytes = notes_to_pptx_bytes(
                    notes_md=notes_md,
                    deck_title=doc_name,
                    max_bullets_per_slide=max_bullets,
                    accent=accent
                )
                st.success(f"PPTX generado ✅ ({len(pptx_bytes)/1024:.1f} KB)")
                st.download_button(
                    "⬇️ Descargar PowerPoint (.pptx)",
                    data=pptx_bytes,
                    file_name=f"{doc_name}_presentacion_pro.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )

        st.success("Todo listo ✅")

    except Exception as e:
        st.error("La app encontró un error, pero lo he capturado para que no se caiga. Detalle:")
        st.exception(e)
