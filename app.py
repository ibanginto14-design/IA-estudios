# app.py
import os
import re
import json
import math
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

# -----------------------------
# Optional deps (we handle missing gracefully)
# -----------------------------
try:
    from pypdf import PdfReader
    PYPDF_OK = True
except Exception:
    PYPDF_OK = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YT_OK = True
except Exception:
    YT_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

# OpenAI client (recommended path)
OPENAI_OK = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_OK = False

# Local whisper (optional)
WHISPER_OK = True
try:
    import whisper  # pip install -U openai-whisper
except Exception:
    WHISPER_OK = False


# -----------------------------
# Helpers
# -----------------------------
def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Simple char-based chunking with paragraph boundaries."""
    text = _clean_text(text)
    if len(text) <= max_chars:
        return [text]

    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            # if single paragraph too large, hard split
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def _extract_youtube_id(url: str) -> Optional[str]:
    # Supports: https://www.youtube.com/watch?v=ID, https://youtu.be/ID, embed, shorts
    patterns = [
        r"v=([A-Za-z0-9_-]{6,})",
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"embed/([A-Za-z0-9_-]{6,})",
        r"shorts/([A-Za-z0-9_-]{6,})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _pdf_to_text(file_bytes: bytes) -> str:
    if not PYPDF_OK:
        raise RuntimeError("Falta dependencia: pypdf. Instala: pip install pypdf")
    reader = PdfReader(file_bytes)
    out = []
    for i, page in enumerate(reader.pages):
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return _clean_text("\n\n".join(out))


# -----------------------------
# LLM Provider
# -----------------------------
@dataclass
class LLMConfig:
    provider: str  # "openai"
    model: str
    temperature: float = 0.2


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if cfg.provider == "openai":
            if not OPENAI_OK:
                raise RuntimeError("No se pudo importar openai. Instala: pip install openai")
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno.")
            self.client = OpenAI(api_key=api_key)
        else:
            raise RuntimeError("Provider no soportado.")

    def chat(self, system: str, user: str) -> str:
        if self.cfg.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        raise RuntimeError("Provider no soportado.")


# -----------------------------
# Prompts
# -----------------------------
SYS_STUDY = """Eres un asistente experto en tomar apuntes.
Tu objetivo es convertir contenido bruto en materiales de estudio claros, estructurados y útiles.
No inventes datos: si algo no aparece en el texto, indícalo como 'no especificado' o 'no consta'.
Usa español neutro y Markdown limpio.
"""

def prompt_notes(text: str, level: str) -> str:
    return f"""
Convierte el siguiente contenido en APUNTES para estudiar.

Requisitos:
- Nivel de detalle: {level} (breve / medio / exhaustivo)
- Estructura con títulos, viñetas, y una sección final de 'Resumen en 10 líneas'
- Añade un glosario corto de términos clave (máx 10)
- Si el texto incluye pasos/procesos, crea un apartado 'Procedimiento'
- Incluye 5 preguntas de repaso al final (sin respuestas)

TEXTO:
\"\"\"{text}\"\"\"
""".strip()

def prompt_flashcards(text: str, n: int) -> str:
    return f"""
Crea {n} FLASHCARDS a partir del texto.
Devuélvelas en JSON estricto con esta forma:
{{
  "flashcards":[
    {{"front":"...", "back":"...", "tags":["...","..."]}}
  ]
}}
- 'front' debe ser una pregunta o concepto
- 'back' debe ser la respuesta explicada en 1-3 frases
- tags: 1-3 etiquetas cortas
TEXTO:
\"\"\"{text}\"\"\"
""".strip()

def prompt_quiz(text: str, n: int, difficulty: str) -> str:
    return f"""
Crea un QUIZ tipo test de {n} preguntas (dificultad: {difficulty}) a partir del texto.
Devuelve JSON estricto:
{{
  "quiz":[
    {{
      "question":"...",
      "options":["A","B","C","D"],
      "answer_index": 0,
      "explanation":"..."
    }}
  ]
}}
Reglas:
- 4 opciones por pregunta
- Solo 1 correcta
- Explicación breve basada en el texto
TEXTO:
\"\"\"{text}\"\"\"
""".strip()

def prompt_mindmap(text: str) -> str:
    return f"""
Genera un mapa mental en formato Mermaid (mindmap) basado en el texto.
Devuelve SOLO el bloque mermaid, sin texto adicional.
Debe ser legible y no excesivamente largo (máx 60 nodos).
TEXTO:
\"\"\"{text}\"\"\"
""".strip()


# -----------------------------
# Transcription
# -----------------------------
def transcribe_audio_openai(audio_path: str, model: str = "whisper-1") -> str:
    if not OPENAI_OK:
        raise RuntimeError("No se pudo importar openai.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        # OpenAI API: audio transcriptions
        # Note: model names may vary by account; whisper-1 is a common baseline.
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
        )
    # resp.text is typical
    return _clean_text(getattr(resp, "text", str(resp)))


def transcribe_audio_local(audio_path: str, model_size: str = "base") -> str:
    if not WHISPER_OK:
        raise RuntimeError("No tienes whisper local instalado. pip install -U openai-whisper")
    m = whisper.load_model(model_size)
    r = m.transcribe(audio_path)
    return _clean_text(r.get("text", ""))


# -----------------------------
# Study pipeline
# -----------------------------
def summarize_chunks(llm: LLM, chunks: List[str], detail_level: str) -> str:
    """Summarize each chunk into notes, then merge into final cohesive notes."""
    if len(chunks) == 1:
        return llm.chat(SYS_STUDY, prompt_notes(chunks[0], detail_level))

    partial_notes = []
    for i, ch in enumerate(chunks, start=1):
        user = prompt_notes(ch, detail_level="medio")
        partial = llm.chat(SYS_STUDY, f"[Parte {i}/{len(chunks)}]\n\n{user}")
        partial_notes.append(partial)

    merged = "\n\n---\n\n".join(partial_notes)
    final = llm.chat(
        SYS_STUDY,
        f"""
Une y depura estos apuntes parciales en unos APUNTES FINALES coherentes (sin repeticiones).
- Mantén estructura clara
- Mejora el orden
- Conserva precisión

APUNTES PARCIALES:
\"\"\"{merged}\"\"\"
""".strip(),
    )
    return final


def safe_parse_json(s: str) -> Dict[str, Any]:
    # Try to extract JSON if model wrapped it
    s = s.strip()
    # Remove markdown fences if present
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        # last resort: find first {...} block
        m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ThetaWave-like Study AI (Streamlit)", page_icon="🧠", layout="wide")

st.title("🧠 StudyWave — Apuntes + Flashcards + Quiz + Mindmap (tipo ThetaWave)")

with st.sidebar:
    st.header("⚙️ Configuración")
    provider = st.selectbox("Proveedor LLM", ["openai"], index=0)
    model = st.text_input("Modelo (OpenAI)", value="gpt-4o-mini")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("📥 Fuente de contenido")
    source = st.radio("Selecciona fuente", ["Texto", "PDF", "Audio", "YouTube"], index=0)

    st.divider()
    st.subheader("🎯 Salidas")
    want_notes = st.checkbox("Generar apuntes", value=True)
    want_flashcards = st.checkbox("Generar flashcards", value=True)
    want_quiz = st.checkbox("Generar quiz", value=True)
    want_mindmap = st.checkbox("Generar mindmap (Mermaid)", value=True)

    detail = st.selectbox("Nivel de apuntes", ["breve", "medio", "exhaustivo"], index=1)
    n_flash = st.slider("Nº flashcards", 5, 60, 20, 1)
    n_quiz = st.slider("Nº preguntas quiz", 5, 40, 12, 1)
    quiz_diff = st.selectbox("Dificultad quiz", ["fácil", "media", "difícil"], index=1)

    st.divider()
    st.caption("Tip: si el texto es enorme, la app lo trocea y fusiona resultados.")


# Collect input content
content = ""
meta = {}

if source == "Texto":
    content = st.text_area("Pega aquí tu texto", height=260, placeholder="Apuntes, artículos, etc.")
elif source == "PDF":
    if not PYPDF_OK:
        st.warning("Para PDF necesitas: `pip install pypdf`")
    pdf = st.file_uploader("Sube un PDF", type=["pdf"])
    if pdf is not None:
        try:
            # pypdf can read from bytes via stream; easiest: pass file-like
            content = _pdf_to_text(pdf)
            meta["pdf_name"] = pdf.name
            st.success(f"PDF cargado: {pdf.name} ({len(content):,} caracteres extraídos)")
        except Exception as e:
            st.error(f"No se pudo extraer texto del PDF: {e}")
elif source == "Audio":
    audio = st.file_uploader("Sube un audio", type=["mp3", "wav", "m4a", "aac", "ogg", "flac"])
    st.caption("Transcripción: OpenAI (recomendado) o Whisper local (si lo tienes instalado).")
    stt_mode = st.radio("Modo de transcripción", ["OpenAI", "Whisper local"], index=0, horizontal=True)
    if audio is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio.name}") as tmp:
            tmp.write(audio.read())
            audio_path = tmp.name

        if st.button("🎙️ Transcribir audio", type="primary"):
            try:
                with st.spinner("Transcribiendo..."):
                    if stt_mode == "OpenAI":
                        content = transcribe_audio_openai(audio_path, model="whisper-1")
                    else:
                        content = transcribe_audio_local(audio_path, model_size="base")
                st.success(f"Transcripción lista ({len(content):,} caracteres)")
                st.text_area("Transcripción", value=content, height=220)
            except Exception as e:
                st.error(f"Error transcribiendo: {e}")
elif source == "YouTube":
    if not YT_OK:
        st.warning("Para transcripción YouTube necesitas: `pip install youtube-transcript-api`")
    url = st.text_input("URL de YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if url and st.button("📼 Obtener transcript", type="primary"):
        try:
            vid = _extract_youtube_id(url)
            if not vid:
                raise RuntimeError("No pude detectar el ID del vídeo. Prueba con otro enlace.")
            with st.spinner("Descargando transcript..."):
                transcript = YouTubeTranscriptApi.get_transcript(vid)
            content = _clean_text(" ".join([x.get("text", "") for x in transcript]))
            meta["youtube_id"] = vid
            st.success(f"Transcript listo ({len(content):,} caracteres)")
            st.text_area("Transcript", value=content, height=220)
        except Exception as e:
            st.error(f"Error obteniendo transcript: {e}")

# Main action
st.divider()
colA, colB = st.columns([1, 2])
with colA:
    go = st.button("✨ Generar materiales", type="primary", use_container_width=True)
with colB:
    st.caption("Si falla el JSON de flashcards/quiz, baja la temperatura o reduce el tamaño del texto.")

if go:
    if not content.strip():
        st.error("No hay contenido. Añade texto / sube un PDF / transcribe audio / añade un YouTube.")
        st.stop()

    # Init LLM
    try:
        llm = LLM(LLMConfig(provider=provider, model=model, temperature=temperature))
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Chunking
    chunks = _chunk_text(content, max_chars=12000)

    results = {}

    # Notes
    if want_notes:
        with st.spinner("Generando apuntes..."):
            results["notes_md"] = summarize_chunks(llm, chunks, detail)

    # For structured JSON tasks, use a merged text (possibly shortened)
    # If many chunks, we first build a condensed 'study context' to keep JSON generation stable.
    study_context = content
    if len(chunks) > 1:
        with st.spinner("Condensando contenido para flashcards/quiz/mindmap..."):
            condensed_parts = []
            for i, ch in enumerate(chunks, start=1):
                part_sum = llm.chat(
                    SYS_STUDY,
                    f"""
Resume esta parte en puntos clave (máx 15 viñetas), sin inventar.
PARTE {i}/{len(chunks)}:
\"\"\"{ch}\"\"\"
""".strip(),
                )
                condensed_parts.append(part_sum)
            study_context = _clean_text("\n\n".join(condensed_parts))

    # Flashcards
    if want_flashcards:
        with st.spinner("Generando flashcards..."):
            raw = llm.chat(SYS_STUDY, prompt_flashcards(study_context, n_flash))
            try:
                results["flashcards_json"] = safe_parse_json(raw)
            except Exception:
                # retry with stricter instruction
                raw2 = llm.chat(
                    SYS_STUDY,
                    "Devuelve SOLO JSON válido. Sin markdown. " + prompt_flashcards(study_context, n_flash)
                )
                results["flashcards_json"] = safe_parse_json(raw2)

    # Quiz
    if want_quiz:
        with st.spinner("Generando quiz..."):
            raw = llm.chat(SYS_STUDY, prompt_quiz(study_context, n_quiz, quiz_diff))
            try:
                results["quiz_json"] = safe_parse_json(raw)
            except Exception:
                raw2 = llm.chat(
                    SYS_STUDY,
                    "Devuelve SOLO JSON válido. Sin markdown. " + prompt_quiz(study_context, n_quiz, quiz_diff)
                )
                results["quiz_json"] = safe_parse_json(raw2)

    # Mindmap (Mermaid)
    if want_mindmap:
        with st.spinner("Generando mindmap (Mermaid)..."):
            mm = llm.chat(SYS_STUDY, prompt_mindmap(study_context))
            # ensure it is a mermaid fenced block
            if "```" not in mm:
                mm = "```mermaid\n" + mm.strip() + "\n```"
            results["mindmap_md"] = mm

    # -----------------------------
    # Render results
    # -----------------------------
    st.success("Listo ✅")
    tab_labels = []
    if want_notes: tab_labels.append("📚 Apuntes")
    if want_flashcards: tab_labels.append("🃏 Flashcards")
    if want_quiz: tab_labels.append("📝 Quiz")
    if want_mindmap: tab_labels.append("🧩 Mindmap")
    tabs = st.tabs(tab_labels) if tab_labels else []

    t = 0
    if want_notes:
        with tabs[t]:
            st.markdown(results["notes_md"])
            st.download_button(
                "⬇️ Descargar apuntes (.md)",
                data=results["notes_md"].encode("utf-8"),
                file_name="apuntes.md",
                mime="text/markdown",
            )
        t += 1

    if want_flashcards:
        with tabs[t]:
            fc = results.get("flashcards_json", {}).get("flashcards", [])
            st.subheader(f"Flashcards ({len(fc)})")
            if PANDAS_OK and fc:
                df = pd.DataFrame(fc)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Descargar flashcards (.csv)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="flashcards.csv",
                    mime="text/csv",
                )
            else:
                st.json(results.get("flashcards_json", {}))
            st.download_button(
                "⬇️ Descargar flashcards (.json)",
                data=json.dumps(results.get("flashcards_json", {}), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="flashcards.json",
                mime="application/json",
            )
        t += 1

    if want_quiz:
        with tabs[t]:
            q = results.get("quiz_json", {}).get("quiz", [])
            st.subheader(f"Quiz ({len(q)})")
            # interactive render
            score = 0
            answered = 0
            for i, item in enumerate(q, start=1):
                st.markdown(f"**{i}. {item.get('question','')}**")
                opts = item.get("options", [])
                if not opts or len(opts) != 4:
                    st.warning("Pregunta con opciones inválidas. Revisa el JSON.")
                    continue
                choice = st.radio(
                    label=f"Respuesta {i}",
                    options=list(range(4)),
                    format_func=lambda idx: opts[idx],
                    key=f"quiz_{i}",
                    horizontal=False,
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
                data=json.dumps(results.get("quiz_json", {}), ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="quiz.json",
                mime="application/json",
            )
        t += 1

    if want_mindmap:
        with tabs[t]:
            st.subheader("Mindmap en Mermaid")
            st.markdown("Pégalo en un visor Mermaid (o en Markdown compatible con Mermaid).")
            st.code(results["mindmap_md"], language="markdown")
            st.download_button(
                "⬇️ Descargar mindmap (.md)",
                data=results["mindmap_md"].encode("utf-8"),
                file_name="mindmap.md",
                mime="text/markdown",
            )

st.divider()
with st.expander("📦 Requisitos recomendados (pip)"):
    st.markdown(
        """
- streamlit
- openai
- pypdf
- youtube-transcript-api
- pandas

Opcional:
- openai-whisper (si quieres transcripción local)
        """.strip()
    )
