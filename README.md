# 📱 Real-Time WhatsApp Chat Summarizer

A powerful NLP tool that takes raw WhatsApp chat exports (`_chat.txt`) and generates concise, abstractive summaries using a fine-tuned Google Pegasus model. This project is deployed on Hugging Face Spaces.

🔗 **Live Demo:** [WhatsApp Chat Summarizer on Hugging Face](https://huggingface.co/spaces/AishaniS/WhatsAppChatSummarizerModel)

---

## 🛠 Tech Stack

- **Language:** Python 3.12+
- **Deep Learning Framework:** PyTorch
- **NLP Library:** Hugging Face Transformers
- **Frontend/UI:** Streamlit
- **Data Processing:** Pandas, Regex
- **Evaluation Metrics:** ROUGE Score

---

## 🤖 Model Details

The core of this application uses a fine-tuned version of the Google Pegasus model, specifically optimized for abstractive summarization of conversations.

- **Base Model:** `google/pegasus-cnn_dailymail`
- **Fine-Tuning Dataset:** SAMSum Corpus (`knkarthick/samsum`)
  - **Why SAMSum?** Unlike news datasets (CNN/DailyMail), SAMSum contains real-life messenger-like conversations, making it ideal for summarizing informal chat logs.
- **Training Configuration:**
  - Batch Size: 1 (with gradient accumulation steps = 16)
  - Epochs: 1
  - Optimizer: AdamW
  - Hardware: Trained on NVIDIA T4 GPU (Google Colab)

---

## 📂 Project Structure

The repository relies on a modular pipeline designed for reproducibility:

- **`app.py`**: The main entry point for the Streamlit application. It handles file uploads, regex cleaning, chunking long chats, and displaying results.
- **`text_summarizer.ipynb`**: The training notebook used to fine-tune the Pegasus model on the SAMSum dataset.
- **`module_2_preprocessing.py`**: Handles tokenization and cleaning of raw text data.
- **`module_3_model.py`**: Loads the model architecture and displays configuration details (layers, vocab size).
- **`module_4_evaluation.py`**: Runs inference on test data and calculates ROUGE metrics to validate summary quality.
- **`requirements.txt`**: List of dependencies required to run the project.

---

## 🚀 Installation & Local Usage

To run this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/whatsapp-summarizer.git
cd whatsapp-summarizer
```

### 2. Install Dependencies

It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Upload Data

- Export a chat from WhatsApp (Settings > Export Chat > Without Media).
- Upload the `_chat.txt` file to the local Streamlit interface.

---

## ☁️ Deployment on Hugging Face

This project is deployed as a Streamlit Space on Hugging Face.

1. **Model Storage:** The fine-tuned model (`AishaniS/text_summarizer`) is hosted on the Hugging Face Hub, allowing `app.py` to pull it dynamically using `AutoModelForSeq2SeqLM.from_pretrained()`.

2. **Caching:** The app uses `@st.cache_resource` to load the heavy model only once, ensuring faster performance for subsequent users.

3. **Chunking Strategy:** To handle WhatsApp chats longer than the model's context window (1024 tokens), the app splits text into chunks of 2000 characters and summarizes them iteratively.

---

## 📊 Performance Metrics

The model's performance is evaluated using ROUGE scores:

- **ROUGE-1:** Measures unigram overlap
- **ROUGE-2:** Measures bigram overlap
- **ROUGE-L:** Measures longest common subsequence

---

## 🔍 How It Works

1. **Upload:** User uploads a WhatsApp `_chat.txt` file
2. **Preprocessing:** Regex removes timestamps, system messages, and metadata
3. **Chunking:** Long conversations are split into manageable chunks
4. **Summarization:** Each chunk is passed through the fine-tuned Pegasus model
5. **Output:** Concatenated summaries are displayed in the UI

---

## 📝 Example Usage

**Input (WhatsApp Chat):**
```
John: Hey, are we still meeting at 5?
Sarah: Yes! Don't forget to bring the documents.
John: Got it. See you there!
```

**Output (Summary):**
```
John and Sarah confirm their 5 PM meeting. John will bring the documents.
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@aishani-s20](https://github.com/aishani-s20)
- Hugging Face: [@AishaniS](https://huggingface.co/AishaniS)
- Email: aishani1020@gmail.com

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for hosting the model and providing the Transformers library
- [SAMSum Dataset](https://huggingface.co/datasets/knkarthick/samsum) for conversational training data
- [Google Pegasus](https://github.com/google-research/pegasus) for the base model architecture