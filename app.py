from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np

app = Flask(__name__)

# Camelot map
camelot_map = {
    "c major": "8B",  "a minor": "8A",
    "g major": "9B",  "e minor": "9A",
    "d major": "10B", "b minor": "10A",
    "a major": "11B", "f# minor": "11A",
    "e major": "12B", "c# minor": "12A",
    "b major": "1B",  "g# minor": "1A",
    "f# major": "2B", "d# minor": "2A",
    "c# major": "3B", "a# minor": "3A",
    "g# major": "4B", "f minor": "4A",
    "d# major": "5B", "c minor": "5A",
    "a# major": "6B", "g minor": "6A",
    "f major": "7B",  "d minor": "7A"
}

# Krumhansl-Schmuckler profiles (standard)
major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key(audio_path):
    # FAST load (downsample to 22050hz)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Analyze only first 60 seconds (FAST)
    max_duration = 60
    y = y[:sr * max_duration]

    # Harmonic separation (best for key detection)
    y_harmonic = librosa.effects.harmonic(y)

    # Chroma extraction (still accurate but faster now)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Normalize chroma
    chroma_mean = chroma_mean / np.sum(chroma_mean)


    best_score = -999
    best_key = None
    best_mode = None

    for i in range(12):
        major_rot = np.roll(major_profile, i)
        minor_rot = np.roll(minor_profile, i)

        major_score = np.corrcoef(chroma_mean, major_rot)[0, 1]
        minor_score = np.corrcoef(chroma_mean, minor_rot)[0, 1]

        # Slight preference for major
        major_score += 0.02

        if major_score > best_score:
            best_score = major_score
            best_key = notes[i]
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_key = notes[i]
            best_mode = "minor"

    final_key = f"{best_key} {best_mode}".lower()

    # ------------------------------
    # FIX 1: G Major vs D Major
    # ------------------------------
    if final_key == "d major":
        C = chroma_mean[0]   # C
        Cs = chroma_mean[1]  # C#

        # If C natural stronger than C#, D major is impossible
        if C > Cs * 1.4:
            final_key = "g major"

    # ------------------------------
    # FIX 2: G# Major vs C Minor confusion
    # ------------------------------
    if final_key == "g# major":
        C = chroma_mean[0]   # C
        Eb = chroma_mean[3]  # D#
        G = chroma_mean[7]   # G

        # If C and Eb are strong, likely C minor
        if C > 0.07 and Eb > 0.05:
            final_key = "c minor"

    # ------------------------------
    # FIX 3: If D# Major appears, prefer C Minor if tonic strong
    # ------------------------------
    if final_key == "d# major":
        C = chroma_mean[0]
        Eb = chroma_mean[3]
        G = chroma_mean[7]

        if C > 0.08 and Eb > 0.06:
            final_key = "c minor"

    # Camelot
    camelot = camelot_map.get(final_key, "N/A")

    # Convert final key to Title Case
    final_key_title = final_key.title()

    return final_key_title, camelot, float(best_score), chroma_mean.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = "uploaded_audio.wav"
    file.save(filepath)

    key, camelot, score, chroma = detect_key(filepath)

    # Top 5 notes
    top_notes = sorted(
        [(notes[i], chroma[i]) for i in range(12)],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    top_notes_text = ", ".join([f"{n} ({v*100:.1f}%)" for n, v in top_notes])

    return jsonify({
        "key": key,
        "camelot": camelot,
        "confidence": round(score * 100, 2),
        "top_notes": top_notes_text
    })


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

