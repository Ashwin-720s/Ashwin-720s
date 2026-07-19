# CBSE Class 11 Study Bot — Full Guide (Unsloth + Google Colab edition)

A complete walkthrough for collecting data and **fine-tuning a real model with Unsloth**, using your i5/16GB PC for data prep and a **free Google Colab GPU** for the actual training.

---

## 0. Why Colab, not your PC

Unsloth is a library that makes fine-tuning LLMs faster and lighter — but it still **requires an NVIDIA GPU**. It does not support CPU-only training at all (confirmed from Unsloth's own docs: CPU is supported for chat/inference only, never training). Your i5 with no GPU simply cannot run it locally.

The fix: **Google Colab gives everyone a free NVIDIA T4 GPU** in the browser, no installation, no cost. Unsloth publishes official beginner-friendly Colab notebooks built exactly for this. So the plan is:

| Where | What happens |
|---|---|
| **Your PC (i5, CPU)** | Collect CBSE textbook data, auto-generate Q&A training pairs |
| **Google Colab (free T4 GPU)** | Fine-tune Llama 3.2 (3B) with Unsloth on your Q&A data |
| **Your PC again** | Download the fine-tuned model, run it locally with Ollama for your live demo |

This means you get to say "I fine-tuned a language model" truthfully — because you did — while never needing to buy or rent a GPU.

---

## 1. Getting the data (CBSE Class 11 textbooks)

1. Go to the official NCERT site: **https://ncert.nic.in/textbook.php**
2. Select **Class 11**, then your subject (Physics, Chemistry, Biology, or Maths) — free, legal, and the actual textbooks CBSE schools use.
3. Download PDFs for **2-3 chapters per subject**. Don't try to cover a whole book — for a 1-week project, focus beats coverage.
4. Rename each file as: `Subject_ChapterNumber_ChapterName.pdf`
   - Example: `Physics_04_Thermodynamics.pdf`
5. Put all PDFs into a folder called `raw_pdfs/` next to the Python scripts.

---

## 2. Setting up your computer

### Install Python
Get Python 3.10 or 3.11 from https://www.python.org/downloads/ (tick "Add to PATH" on Windows).

### Create a virtual environment and install packages
```bash
cd path\to\your\project\folder
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Install Ollama (to run your fine-tuned model locally later)
1. Download from **https://ollama.com** and install.
2. Pull the base model (used for auto-generating Q&A pairs, and as a fallback):
   ```bash
   ollama pull llama3.2:3b
   ```
3. Leave Ollama running in the background — it starts a local server automatically.

You do **not** need a Google account with billing, or anything paid — the free Colab tier is enough for a 3B model on a small dataset.

---

## 3. Step-by-step pipeline

### Step 1 — Extract and chunk textbook text
```bash
python 1_prepare_data.py
```
Reads your PDFs, splits them into clean text chunks, labels each with subject/chapter (guessed from the filename). Produces `data/chunks.csv`. Skim this file afterward to make sure labels look right.

### Step 2 — Auto-generate Q&A training pairs
```bash
python 2_generate_qa_pairs.py
```
Fine-tuning needs **question -> answer** pairs, not raw paragraphs. This script uses your local Ollama (llama3.2:3b) to read each chunk and write 2 realistic exam-style Q&A pairs from it. This step doesn't train anything — it just uses an existing model to help build your dataset, so it's fine on CPU (expect it to take a while; a few chunks per minute).

Output: `data/training_data.json`, in this format:
```json
[
  {"instruction": "What is the first law of thermodynamics?", "input": "", "output": "..."},
  {"instruction": "...", "input": "", "output": "..."}
]
```

**Aim for at least 100-200 pairs total** for a project this size — check the count the script prints at the end. If it's low, add more chapters and re-run Steps 1-2.
   
**Quality check (important — do this, it's 15 minutes well spent):** open `data/training_data.json` and skim 15-20 pairs. Delete any that are vague, wrong, or too easy. Bad training examples teach the model bad habits — better to fine-tune on 120 solid pairs than 200 sloppy ones.

### Step 3 — Fine-tune with Unsloth on Google Colab (free GPU)

1. Open Unsloth's official free notebook: **https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb**
   (This is Unsloth's maintained Alpaca-format fine-tuning notebook — the same template works for Llama 3.2 3B, just swap the model name as below. Using their maintained notebook instead of a custom one means you're less likely to hit broken dependencies.)
2. In Colab, go to **Runtime -> Change runtime type -> T4 GPU**, save. This is what gives you the free GPU.
3. Click **Runtime -> Run all** once first, top to bottom, without changes — this confirms the notebook works before you touch anything.
4. Now make these edits and re-run from that cell onward:
   - **Model name cell**: change the model string to `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` (smaller and faster than the default 8B — better for a live demo).
   - **Dataset cell**: this is the main change. Replace the line that loads the default Alpaca dataset with code that loads your file instead:
     ```python
     from datasets import Dataset
     import json

     with open("training_data.json") as f:
         data = json.load(f)
     dataset = Dataset.from_list(data)
     dataset = dataset.map(formatting_prompts_func, batched=True)
     ```
     Before this, use Colab's file upload button (folder icon on the left sidebar) to upload your `data/training_data.json`, or run in a cell:
     ```python
     from google.colab import files
     uploaded = files.upload()   # pick training_data.json from your PC
     ```
   - **Training args cell**: for a small dataset (~100-300 examples), 2-3 epochs is enough — look for `num_train_epochs` or `max_steps` and set `num_train_epochs = 3`. Too many epochs on a small dataset causes the model to just memorize instead of generalizing.
5. Run the training cell. For a 3B model on a few hundred examples, expect roughly **10-25 minutes** on the free T4.
6. Once training finishes, run the notebook's **GGUF export cell** (it's already in the notebook, usually near the end, something like `model.save_pretrained_gguf(...)`). This converts your fine-tuned model into a format Ollama can run.
7. Download the resulting `.gguf` file to your PC (Colab's file browser on the left -> right-click -> Download). It'll be a few GB.

**If Colab disconnects or times out:** free Colab sessions can be limited to a few hours and may disconnect if idle. Keep the tab active, don't close it mid-training, and if it disconnects, just re-run from the top — your uploaded dataset stays in that session only, so re-upload it if you have to restart.

### Step 4 — Bring your fine-tuned model back to your PC

1. Put the downloaded `.gguf` file in your project folder, and rename it to match the `Modelfile` (or edit the `Modelfile`'s `FROM` line to match your actual filename).
2. Import it into Ollama:
   ```bash
   ollama create cbse-study-bot -f Modelfile
   ```
3. Test it directly:
   ```bash
   ollama run cbse-study-bot
   ```
   Ask it a few questions from your chapters and see how it does.

### Step 5 — Build the retrieval layer (optional but recommended)
```bash
python 3_build_retrieval_index.py
```
This embeds your textbook chunks so the app can pull in the actual source text alongside your fine-tuned model's answer — good for showing judges the model isn't making things up, and a nice safety net if the fine-tuned model gets something wrong live.

### Step 6 — Run the chatbot demo
```bash
streamlit run 4_app.py
```
This talks to your fine-tuned `cbse-study-bot` model first, and automatically falls back to the base `llama3.2:3b` if the fine-tuned one isn't imported yet (handy while you're still setting up).

---

## 4. What to say to judges

Be precise about what you actually did — it's a stronger story than an inflated one:

- *"I collected CBSE Class 11 textbook content and generated a Q&A training dataset from it."*
- *"I fine-tuned Llama 3.2 (3B parameters) on that dataset using Unsloth, which makes fine-tuning fast enough to run on a free cloud GPU in about 20 minutes instead of needing my own expensive hardware."*
- *"The fine-tuned model now runs entirely on my own laptop, offline, through Ollama — no internet or ongoing cost needed for the demo."*
- If you also built the retrieval layer: *"I paired it with a retrieval system that pulls the actual textbook passage, so the answer is grounded in real content rather than the model just generating something plausible-sounding."*

If asked "did you train it from scratch": be upfront that Llama 3.2 was pretrained by Meta, and what you did is **fine-tune** it — adjust an already-capable model to specialize on your subject matter. That's exactly what fine-tuning is for, and it's the standard, correct way to build something like this — claiming otherwise would actually undersell how the technique works.

---

## 5. One-week schedule

| Day | Task |
|---|---|
| **Day 1** | Download PDFs, set up Python + Ollama, run Step 1 (data prep) |
| **Day 2** | Run Step 2 (Q&A generation), then hand-review and clean the dataset |
| **Day 3** | Set up Colab, run the Unsloth notebook once unmodified to confirm it works, then start adapting it to your dataset |
| **Day 4** | Fine-tune for real, export GGUF, download, import into Ollama (Steps 3-4) |
| **Day 5** | Build retrieval layer + Streamlit app (Steps 5-6), test with 15-20 real questions |
| **Day 6** | Polish UI, prepare poster (pipeline diagram, example Q&A, what you fine-tuned vs. what was pretrained) |
| **Day 7 (Fri)** | Final rehearsal. Export a short screen-recording of the demo working as backup in case of WiFi/hardware issues |

Given the added Colab step, **don't leave Step 3 (actual fine-tuning) until the last two days** — Colab GPU availability and session limits can be unpredictable, and you want buffer time if a run needs to be redone.

---

## 6. Troubleshooting

- **"Colab says no GPU available / stuck in queue"** — free-tier GPU access can be rate-limited at busy times. Try again in an hour, or check if you're accidentally on a CPU runtime (Runtime -> Change runtime type).
- **"Training loss isn't going down"** — usually means the dataset is too small or too repetitive. Add more chapters/Q&A pairs.
- **"Fine-tuned model gives weird/broken answers"** — likely too many training epochs on too little data (overfitting). Reduce `num_train_epochs` to 2 and re-train, or add more data.
- **`ollama create` fails** — double check the `.gguf` filename in `Modelfile` exactly matches your downloaded file, and that the file is in the same folder as `Modelfile`.
- **"Ollama not found" error in the Streamlit app** — make sure `ollama serve` is running (opening the Ollama app does this automatically).

---

## 7. Video tutorials

- **Unsloth's own fine-tuning walkthrough** (Llama fine-tune + export to Ollama, closely matches Step 3-4 above):
  https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
- **Ollama basics for beginners**:
  https://www.youtube.com/watch?v=tVTwZRhxw9w
- **TF-IDF text classification** (only needed if you build the optional bonus classifier):
  https://www.youtube.com/watch?v=i74DVqMsRWY
