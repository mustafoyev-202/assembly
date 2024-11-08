# Speech-to-Text with Speaker Diarization and Summary 🎤💬✨

Welcome to **Speech-to-Text with Speaker Diarization and Summary**, an intelligent Streamlit app that seamlessly converts audio files into text, identifies speakers, and provides insightful summaries. Powered by AssemblyAI and Lemur AI, this app is perfect for those who want to unlock the value in their audio content quickly and efficiently. 

## 🚀 Features

### 1. **Speech-to-Text Transcription**
   - Automatically transcribes uploaded audio files into text.
   - Supports popular audio formats: `.wav`, `.mp3`, `.m4a`.
   - Ensures high accuracy with AssemblyAI’s robust transcription engine.

### 2. **Speaker Diarization**
   - Identifies and labels different speakers in a conversation.
   - Displays transcription alongside speaker labels, providing clear, structured dialogue.

### 3. **Automated Summarization**
   - Generates a concise summary of the conversation using Lemur AI.
   - Categorizes topics discussed (e.g., business, personal, technical).
   - Enables quick insights without combing through lengthy conversations.

### 4. **Interactive and User-Friendly Interface**
   - Easy-to-use Streamlit interface for seamless interaction.
   - Option to view the full transcript or just the summary.
   - Instant feedback and error handling for a smooth experience.

---

## 🛠️ Installation and Setup

### Prerequisites
Make sure you have the following installed:
- **Python 3.10 or higher**
- **pip** (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/mustafoyev-202/assembly.git
cd speech-to-text-diarization
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Your Environment Variables
Create a `.env` file in the project root and add your AssemblyAI API key:
```
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

> **Note:** You can obtain your API key by signing up for [AssemblyAI](https://www.assemblyai.com/).

### Step 4: Run the App
```bash
streamlit run app.py
```

---

## 📦 Project Structure

```plaintext
speech-to-text-diarization/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Required Python dependencies
├── .env                   # Environment variables for API keys
└── README.md              # Project documentation
```

---

## 🧠 How It Works

1. **Upload an Audio File**  
   Upload an audio file in `.wav`, `.mp3`, or `.m4a` format.

2. **Transcription with Diarization**  
   The app uses AssemblyAI to transcribe your audio and identify distinct speakers.

3. **Summary Generation**  
   Lemur AI analyzes the transcript to generate a concise summary and categorize the discussion topics.

4. **Review Transcription and Summary**  
   View the speaker-attributed transcript and optional full conversation summary.

---

## 💡 Use Cases

- **Business Meetings**: Quickly capture and summarize key points from meetings.
- **Podcast Transcription**: Easily transcribe and summarize podcast episodes.
- **Interviews and Focus Groups**: Identify speakers and summarize insights.
- **Lectures and Workshops**: Obtain a summarized record of educational content.

---

## 🌐 Technologies Used

- **[Streamlit](https://streamlit.io/):** Interactive web app framework.
- **[AssemblyAI](https://www.assemblyai.com/):** Speech-to-text and speaker diarization API.
- **Lemur AI** (via AssemblyAI): Advanced language models for text summarization.

---

## 🛡️ Security

- Ensure your API keys are kept private by using a `.env` file.
- Do not commit your `.env` file to version control.

---

## 📝 Future Enhancements

- **Real-time Transcription:** Enable real-time transcription for live audio streams.
- **Multi-Language Support:** Extend transcription capabilities to multiple languages.
- **Sentiment Analysis:** Provide emotional insights from the conversation.
- **Export Options:** Allow users to download transcripts and summaries as text or PDF files.

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-branch-name`.
5. Submit a pull request.

---

## 📧 Contact

For any questions or feedback, feel free to reach out:

- **Email:** baxtiyormustafoyev2006@gmail.com  
- **GitHub:** mustafoyev-202

---

## 🏆 Acknowledgments

A big thanks to:
- **AssemblyAI** for their powerful transcription and speaker diarization capabilities.
- **Streamlit** for making it easy to build interactive web apps.
- **Lemur AI** for enhancing the value of transcripts with automated summaries.

---

Ready to transform your audio content into actionable insights? **Let’s get started!** 🚀