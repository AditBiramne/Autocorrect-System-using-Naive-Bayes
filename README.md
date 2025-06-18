# 🔠 Autocorrect-System-using-Naive-Bayes

A simple and lightweight **autocorrect system** built using the **Naive Bayes approach** and **edit distance**. It simulates how intelligent text correction can be achieved without large datasets or deep learning — ideal for understanding core NLP concepts.

---

📌 Overview

This system corrects misspelled words by calculating **posterior probabilities** of candidate words using:

- **P(C)**: Prior probability (based on word frequency)
- **P(W|C)**: Likelihood (based on edit distance between input and candidates)
- **P(C|W)**: Posterior probability (product of the above two)

---

🔍 Example

```python
misspelled_word = "helo"
suggestions = autocorrect_naive_bayes(misspelled_word, word_counts, total_words)
```

✅ Returns top suggestions like:

```
hello (Probability: 0.1)
world (Probability: 0.05)
data (Probability: 0.04)
```

---

📦 Dependencies

- Python 3
- editdistance
- collections (Python standard library)

🛠 Install with:

```bash
pip install editdistance
```

---

📁 How It Works

### 1️⃣ Sample Corpus (Unigram)

A basic vocabulary is hardcoded:

```python
corpus = [
  "hello", "world", "python", "programming",
  "machine", "learning", "artificial", "intelligence",
  "data", "science"
]
```

### 2️⃣ Word Frequency

Frequencies are counted using:

```python
from collections import Counter
word_counts = Counter(corpus)
total_words = sum(word_counts.values())
```

### 3️⃣ Prior Probability: P(C)

```python
prior = word_counts[candidate] / total_words
```

### 4️⃣ Likelihood: P(W|C)

```python
import editdistance
likelihood = 1 / (editdistance.eval(input_word, candidate) + 1)
```

### 5️⃣ Posterior: P(C|W)

```python
posterior = prior * likelihood
```

---

🚀 How to Run

1. ✅ Install the required package:

```bash
pip install editdistance
```

2. ✅ Save the script as `autocorrect.py`

3. ✅ Run using:

```bash
python autocorrect.py
```

4. ✅ Output:

```
Suggestions for 'helo':
hello (Probability: 0.1)
world (Probability: 0.05)
data (Probability: 0.04)
...
```

---

🌟 Highlights

- No external dataset required — simple unigram model
- Lightweight and easy to understand
- Demonstrates key ML concepts: prior, likelihood, and posterior
- Great for learning **Naive Bayes** without overwhelming complexity

---

⚙️ Future Improvements

- 📚 Use a **large real-world corpus** (e.g., from Wikipedia, books, or subtitles)
- 🧠 Incorporate **bigram** or **context-based probabilities**
- 🔤 Improve likelihood with **realistic typo models**
- 🏁 Add ranking metrics like Levenshtein weight or keyboard distance
- 📲 Convert to an **interactive web app** using Flask or Streamlit

---

📚 References

- [EditDistance PyPI](https://pypi.org/project/editdistance/)
- [Naive Bayes Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

