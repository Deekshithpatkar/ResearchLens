import re
from typing import List

NOISE_PATTERNS = [
    r"\b(reg\.?\s*no|guide|supervisor|department|university|institute|submitted|msc|mtech|phd|author|student)\b",
    r"^page\s*\d+$",
    r"^\d+$",
]

SECTION_RE = re.compile(
    r"^(\d+(\.\d+)*)?\s*(abstract|introduction|background|related work|related-work|method|methods|approach|experiment|experiments|results?|discussion|conclusion|limitations|future work|references)\b",
    re.I,
)


def preprocess_text_lines(text: str) -> List[str]:
    """Clean extracted PDF text line by line while preserving section boundaries."""
    if not text:
        return []

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        if any(re.search(pattern, line, re.I) for pattern in NOISE_PATTERNS):
            continue

        if len(line.split()) <= 2 and not re.search(r"[A-Za-z]", line):
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def preprocess_text(text: str) -> str:
    """Clean extracted PDF text by removing common header/footer noise."""
    cleaned_lines = preprocess_text_lines(text)
    cleaned_text = " ".join(cleaned_lines)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def normalize_heading(line: str) -> str:
    cleaned = re.sub(r"\s+", " ", line).strip()
    cleaned = re.sub(r"^\d+(\.\d+)*\s*", "", cleaned)
    if not cleaned:
        return ""
    if cleaned.lower() == "abstract":
        return "Abstract"
    if cleaned.lower() == "introduction":
        return "Introduction"
    return cleaned


def is_section_heading(line: str) -> bool:
    return bool(SECTION_RE.match(line.strip()))


def chunk_text(text, chunk_size=220, overlap=30) -> List[str]:
    """Chunk cleaned text into semantic-ish blocks while preserving section structure when present."""
    lines = preprocess_text_lines(text)
    if not lines:
        return []

    sections = []
    current_heading = None
    current_lines = []

    for line in lines:
        if is_section_heading(line):
            if current_lines or current_heading is not None:
                sections.append((current_heading, " ".join(current_lines).strip()))
            current_heading = normalize_heading(line)
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_heading is not None:
        sections.append((current_heading, " ".join(current_lines).strip()))

    if sections and any(section_text for _, section_text in sections):
        chunks = []
        for heading, section_text in sections:
            if not section_text:
                continue
            section_sentences = re.split(r"(?<=[.!?])\s+", section_text)
            section_sentences = [s.strip() for s in section_sentences if s.strip()]
            if not section_sentences:
                continue

            combined = []
            combined_len = 0
            for sentence in section_sentences:
                words = sentence.split()
                if not words:
                    continue
                if combined_len + len(words) <= max(chunk_size // 2, 80):
                    combined.append(sentence)
                    combined_len += len(words)
                else:
                    if combined:
                        chunk_value = f"{heading}: {' '.join(combined).strip()}" if heading else " ".join(combined).strip()
                        chunks.append(chunk_value)
                    combined = [sentence]
                    combined_len = len(words)

            if combined:
                chunk_value = f"{heading}: {' '.join(combined).strip()}" if heading else " ".join(combined).strip()
                chunks.append(chunk_value)

        if chunks:
            return chunks

    cleaned_text = preprocess_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if current_len + len(words) <= chunk_size:
            current_chunk.append(sentence)
            current_len += len(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks