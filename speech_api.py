# speech_api.py
from gtts import gTTS
from pathlib import Path

def tts_long_text(
    text: str,
    base_filename: str = "speech",
    max_chunk_length: int = 800,
    max_chunks: int = 5,
    lang: str = "en"
):
    """
    Convert long text into multiple audio files using gTTS (no API key needed).

    Returns:
        (True, [Path, ...]) on success
        (False, error_message) on failure
    """
    # 1. Split text into word-bound chunks
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_length, text_len)
        if end < text_len:
            # backtrack to last space to avoid cutting a word
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    # 2. Enforce chunk limit
    if len(chunks) > max_chunks:
        return False, (
            f"Text too long for TTS (needs {len(chunks)} chunks, limit is {max_chunks})."
        )

    # 3. Generate and save audio
    audio_paths = []
    for i, chunk in enumerate(chunks):
        try:
            tts = gTTS(text=chunk, lang=lang)
            out_path = Path(__file__).parent / f"{base_filename}_{i}.mp3"
            tts.save(str(out_path))
            audio_paths.append(out_path)
        except Exception as e:
            return False, f"TTS failed at chunk {i+1}: {e}"

    return True, audio_paths
