# 🎹 AI Meeting Summarizer

An intelligent tool that converts audio meetings into clean, structured summaries with sentiment insights. It helps teams capture meeting highlights, action items, and emotional tone — all automated using speech recognition and AI.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PranavVaish/AI_Meeting_summarizer.git
cd AI_Meeting_summarizer
```

### 2. Install Dependencies

Make sure you’re using **Python 3.8+**.

```bash
pip install -r requirements.txt
```

### 3. Set API Keys (for OpenAI)

Create a `.env` file in the root directory and add:

```ini
OPENAI_API_KEY=your_openai_key_here
```

Or export it directly in the terminal:

```bash
export OPENAI_API_KEY="your_openai_key_here"
```

### 4. Run the Application

#### For Flask:

```bash
python app.py
```

#### For Streamlit:

```bash
streamlit run app.py
```

---

## 🧐 How It Works

1. **Upload Audio**\
   Drop your `.wav` or `.mp3` file into the UI or place it in the `audio/` folder.

2. **Speech-to-Text**\
   The system transcribes the audio using `speech_recognition` or `Whisper`.

3. **Summarization**\
   It sends the transcript to OpenAI (or other LLMs) to generate bullet-point summaries.

4. **Sentiment Analysis**\
   Uses `VADER` or `TextBlob` to assess the emotional tone of the meeting.

5. **Outputs**\
   All results are saved in:

   - `/transcripts`
   - `/summaries`
   - `/sentiment_results`

---

## 📸 Screenshots

📂 [Click here to view sample outputs](https://drive.google.com/drive/folders/1Sv5XDUT5-_jNQICXA7LiUjCHpwBChXbE?usp=drive_link)

---

## ✨ Sample Output

### 📄 Summary

```
- Team discussed sprint progress and major blockers.
- Positive feedback on recent UI changes.
- Feature B will be delayed due to integration bugs.
- Action Items:
  • Fix login issue by Friday.
  • Schedule next client demo.
```

### 😊 Sentiment

```
Overall tone: Neutral → Positive

Emotion Peaks:
  • 00:01:23 → Negative spike (frustration on bug)
  • 00:02:45 → Positive spike (client feedback)
```

---

## 🛠️ Troubleshooting

| Issue                     | Fix                                                 |
| ------------------------- | --------------------------------------------------- |
| 🔐 API key error          | Ensure your OpenAI key is correctly added to `.env` |
| 🎧 Audio not transcribing | Use high-quality `.wav` or `.mp3` audio files       |
| 📦 Dependency error       | Re-run `pip install -r requirements.txt`            |

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** your feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m "Add my feature"
   ```
4. **Push** to the branch:
   ```bash
   git push origin feat/my-feature
   ```
5. **Open a Pull Request**

---

## 📜 License

This project is licensed under the **MIT License**.\
Feel free to fork, improve, and share it!

---

## 🙋 Contact

Built with ❤️ by **Pranav Vaish**\
📧 [LinkedIn Profile](https://www.linkedin.com/in/pranavvaish20)
