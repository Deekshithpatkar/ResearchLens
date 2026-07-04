from pathlib import Path
from backend.pdf_utils import extract_text_from_pdf
from backend.text_chunker import chunk_text, preprocess_text
pdf_path = Path('data/Attention is all you need.pdf')
print('exists', pdf_path.exists())
text = extract_text_from_pdf(str(pdf_path))
print('text len', len(text))
print(text[:5000])
print('---preprocessed sample---')
print(preprocess_text(text)[:4000])
print('---chunks---')
chunks = chunk_text(text, chunk_size=120)
for i, c in enumerate(chunks[:15]):
    print(i, '---')
    print(c[:900])
    print()
