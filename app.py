from __future__ import annotations

import re
import io
import math
import json
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any

import streamlit as st

# =========================
# Optional PDF support
# =========================
PYPDF_OK = True
try:
    from pypdf import PdfReader
except Exception:
    PYPDF_OK = False


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
def clean_text(t: str) -> str:
    t = (t or "").replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def normalize_token(w: str) -> str:
    w = w.lower()
    w = re.sub(r"^[^\wáéíóúüñ]+|[^\wáéíóúüñ]+$", "", w, flags=re.UNICODE)
    return w

def tokenize_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text, flags=re.UNICODE)
    tokens: List[str] = []
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

def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter robust to PDFs
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9])", text)
    if len(parts) <= 1:
        parts = re.split(r"\s{2,}|;\s+", text)
    return [p.strip() for p in parts if p and p.strip()]

def pdf_to_text(file_bytes: bytes) -> str:
    if not PYPDF_OK:
        raise RuntimeError("PDF no disponible: instala pypdf (ver requirements.txt).")
    bio = io.BytesIO(file_bytes)
    reader = PdfReader(bio)
    pages: List[str] = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n\n".join(pages))


# =========================
# Classic NLP scoring
# =========================
def tfidf_sentence_scores(sentences: List[str]) -> Tuple[List[float], Dict[str, float]]:
    if not sentences:
        return [], {}

    sent_tokens = [tokenize_words(s) for s in sentences]
    df = Counter()
    for toks in sent_tokens:
        for t in set(toks):
            df[t] += 1

    N = max(1, len(sentences))
    idf: Dict[str, float] = {}
    for t, c in df.items():
        idf[t] = math.log((N + 1) / (c + 1)) + 1.0  # smoothed

    scores: List[float] = []
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

def pick_top_sentences(sentences: List[str], max_sentences: int) -> List[str]:
    if not sentences:
        return []
    scores, _ = tfidf_sentence_scores(sentences)
    idx = list(range(len(sentences)))
    idx.sort(key=lambda i: scores[i], reverse=True)
    chosen = sorted(idx[: max_sentences])
    return [sentences[i] for i in chosen]

def top_keywords(text: str, k: int = 12) -> List[str]:
    toks = tokenize_words(text)
    if not toks:
        return []
    c = Counter(toks)
    ranked = sorted(c.items(), key=lambda x: (x[1], len(x[0]) >= 4, len(x[0])), reverse=True)
    out: List[str] = []
    for w, _ in ranked:
        if w not in out:
            out.append(w)
        if len(out) >= k:
            break
    return out

def find_def_sentence(term: str, sentences: List[str]) -> Optional[str]:
    patt = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    candidates = [s for s in sentences if patt.search(s)]
    if not candidates:
        return None
    cues = ("es ", "son ", "se define", "consiste", "significa", "se entiende")
    def_like = [s for s in candidates if any(c in s.lower() for c in cues)]
    return (def_like[0] if def_like else candidates[0]).strip()

def extract_procedure(text: str) -> List[str]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    proc: List[str] = []
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
def generate_questions(text: str, n: int = 5) -> List[str]:
    text = clean_text(text)
    sents = split_sentences(text)
    kws = top_keywords(text, k=20)
    questions: List[str] = []

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

def generate_notes(text: str, detail: str) -> str:
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
    md: List[str] = []
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

def generate_flashcards(text: str, n: int) -> Dict[str, Any]:
    text = clean_text(text)
    sentences = split_sentences(text)
    kws = top_keywords(text, k=max(12, n))
    cards: List[Dict[str, Any]] = []

    for term in kws:
        if len(cards) >= n:
            break
        ds = find_def_sentence(term, sentences)
        if ds:
            cards.append({"front": f"¿Qué es {term}?", "back": ds, "tags": [term]})

    if len(cards) < n:
        pool = [s for s in sentences if len(tokenize_words(s)) >= 6]
        random.shuffle(pool)
        for s in pool:
            if len(cards) >= n:
                break
            toks = tokenize_words(s)
            present = [t for t in toks if t in kws]
            if not present:
                continue
            target = present[0]
            cloze = re.sub(rf"\b{re.escape(target)}\b", "_____", s, flags=re.IGNORECASE)
            cards.append({
                "front": f"Completa: {cloze}",
                "back": f"La palabra clave era **{target}**. Contexto: {s}",
                "tags": [target]
            })

    return {"flashcards": cards[:n]}

def generate_quiz(text: str, n: int, difficulty: str) -> Dict[str, Any]:
    text = clean_text(text)
    sentences = split_sentences(text)
    kws = top_keywords(text, k=40)
    if not sentences or not kws:
        return {"quiz": []}

    candidates = [s for s in sentences if 8 <= len(tokenize_words(s)) <= 28]
    if not candidates:
        candidates = sentences[:]

    random.shuffle(candidates)
    quiz: List[Dict[str, Any]] = []
    used = set()

    for s in candidates:
        if len(quiz) >= n:
            break
        toks = tokenize_words(s)
        present = [t for t in toks if t in kws]
        if not present:
            continue

        target = present[0]
        key = (target, s[:60])
        if key in used:
            continue
        used.add(key)

        distractors = [k for k in kws if k != target]
        random.shuffle(distractors)
        if difficulty == "difícil":
            distractors.sort(key=lambda x: abs(len(x) - len(target)))

        opts = [target] + distractors[:3]
        random.shuffle(opts)

        cloze = re.sub(rf"\b{re.escape(target)}\b", "_____", s, flags=re.IGNORECASE)
        quiz.append({
            "question": f"Completa la frase: {cloze}",
            "options": opts,
            "answer_index": opts.index(target),
            "explanation": f"En el texto aparece: {s}"
        })

    return {"quiz": quiz[:n]}

def generate_mindmap_mermaid(text: str) -> str:
    text = clean_text(text)
    if not text:
        return "```mermaid\nmindmap\n  root((Sin contenido))\n```"

    kws = top_keywords(text, k=18)
    root = kws[0].capitalize() if kws else "Tema"
    sents = split_sentences(text)

    co = defaultdict(Counter)
    for s in sents:
        toks = set(tokenize_words(s))
        present = [k for k in kws if k in toks]
        for a in present:
            for b in present:
                if a != b:
                    co[a][b] += 1

    lines = ["```mermaid", "mindmap", f"  root(({root}))"]
    for k in kws[1:7]:
        lines.append(f"    {k}")
        for sub, _ in co[k].most_common(2):
            lines.append(f"      {sub}")
    lines.append("```")
    return "\n".join(lines)


# =========================
# UI
# =========================
st.set_page_config(page_title="StudyWave (sin API)", page_icon="🧠", layout="wide")
st.title("🧠 StudyWave — Apuntes + Flashcards + Quiz + Mindmap (SIN API)")

st.info(
    "Esta versión NO usa ChatGPT ni modelos en la nube. "
    "Funciona sin API y sin descargar modelos, usando NLP clásica (resumen extractivo + heurísticas)."
)

with st.sidebar:
    st.header("📥 Fuente")
    source = st.radio("Selecciona fuente", ["Texto", "PDF"], index=1)

    st.divider()
    st.header("🎯 Salidas")
    want_notes = st.checkbox("Generar apuntes", True)
    want_flashcards = st.checkbox("Generar flashcards", True)
    want_quiz = st.checkbox("Generar quiz", True)
    want_mindmap = st.checkbox("Generar mindmap (Mermaid)", True)

    st.divider()
    detail = st.selectbox("Nivel de apuntes", ["breve", "medio", "exhaustivo"], index=1)
    n_flash = st.slider("Nº flashcards", 5, 60, 20, 1)
    n_quiz = st.slider("Nº preguntas quiz", 5, 40, 12, 1)
    quiz_diff = st.selectbox("Dificultad quiz", ["fácil", "media", "difícil"], index=1)

content = ""

if source == "Texto":
    content = st.text_area("Pega aquí tu texto", height=260, placeholder="Pega apuntes, artículos, etc.")
else:
    if not PYPDF_OK:
        st.warning("Para leer PDFs necesitas `pypdf` (no es un modelo, solo lector de PDF).")
    pdf = st.file_uploader("Sube un PDF", type=["pdf"])
    if pdf is not None:
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
go = st.button("✨ Generar materiales", type="primary", use_container_width=True)

if go:
    if not content.strip():
        st.error("No hay contenido. Añade texto o sube un PDF.")
        st.stop()

    results: Dict[str, Any] = {}

    if want_notes:
        with st.spinner("Generando apuntes (NLP clásica)..."):
            results["notes_md"] = generate_notes(content, detail)

    if want_flashcards:
        with st.spinner("Generando flashcards..."):
            results["flashcards_json"] = generate_flashcards(content, n_flash)

    if want_quiz:
        with st.spinner("Generando quiz..."):
            results["quiz_json"] = generate_quiz(content, n_quiz, quiz_diff)

    if want_mindmap:
        with st.spinner("Generando mindmap..."):
            results["mindmap_md"] = generate_mindmap_mermaid(content)

    st.success("Listo ✅")

    tabs = []
    if want_notes: tabs.append("📚 Apuntes")
    if want_flashcards: tabs.append("🃏 Flashcards")
    if want_quiz: tabs.append("📝 Quiz")
    if want_mindmap: tabs.append("🧩 Mindmap")

    tab_objs = st.tabs(tabs) if tabs else []
    ti = 0

    if want_notes:
        with tab_objs[ti]:
            st.markdown(results["notes_md"])
            st.download_button(
                "⬇️ Descargar apuntes (.md)",
                data=results["notes_md"].encode("utf-8"),
                file_name="apuntes.md",
                mime="text/markdown",
            )
        ti += 1

    if want_flashcards:
        with tab_objs[ti]:
            st.subheader(f"Flashcards ({len(results['flashcards_json'].get('flashcards', []))})")
            st.json(results["flashcards_json"])
            st.download_button(
                "⬇️ Descargar flashcards (.json)",
                data=json.dumps(results["flashcards_json"], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="flashcards.json",
                mime="application/json",
            )
        ti += 1

    if want_quiz:
        with tab_objs[ti]:
            q = results["quiz_json"].get("quiz", [])
            st.subheader(f"Quiz ({len(q)})")

            score = 0
            answered = 0
            for i, item in enumerate(q, start=1):
                st.markdown(f"**{i}. {item.get('question','')}**")
                opts = item.get("options", [])
                if len(opts) != 4:
                    st.warning("Pregunta inválida (opciones).")
                    continue

                choice = st.radio(
                    label=f"Respuesta {i}",
                    options=list(range(4)),
                    format_func=lambda idx: opts[idx],
                    key=f"quiz_{i}",
                )
                answered += 1
                if choice == int(item.get("answer_index", -1)):
                    score += 1

                with st.expander("Ver explicación"):
                    st.write(item.get("explanation", ""))
                st.divider()

            if answered:
                st.info(f"Puntuación: **{score}/{answered}**")

            st.download_button(
                "⬇️ Descargar quiz (.json)",
                data=json.dumps(results["quiz_json"], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="quiz.json",
                mime="application/json",
            )
        ti += 1

    if want_mindmap:
        with tab_objs[ti]:
            st.subheader("Mindmap (Mermaid)")
            st.code(results["mindmap_md"], language="markdown")
            st.download_button(
                "⬇️ Descargar mindmap (.md)",
                data=results["mindmap_md"].encode("utf-8"),
                file_name="mindmap.md",
                mime="text/markdown",
            )
