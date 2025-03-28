# Movie & TV Series Identification using Generative AI and RAG

## Overview
This project aims to develop a system that identifies the name of a movie or TV series using a given dialogue. Inspired by Shazam, which identifies songs based on audio snippets, this system leverages Generative AI and Retrieval-Augmented Generation (RAG) to accurately map dialogues to their respective media sources.

## Features
- **Subtitle Extraction**: Extracts subtitle files from a database and cleans them.
- **Data Processing**: Prepares and structures subtitles for efficient searching.
- **Vector Embedding & Storage**: Converts subtitle text into vector embeddings for fast similarity searches.
- **Retrieval-Augmented Generation (RAG)**: Uses embeddings and generative AI to retrieve relevant content.
- **Query Processing**: Matches a given dialogue to the closest subtitle entries.
- **Metadata Extraction**: Extracts season, episode, and movie information.
- **ASR Integration**: Converts speech to text for direct audio-to-movie identification.

## Tech Stack
- **Python**: Core programming language.
- **LangChain**: Manages document loading, embeddings, and similarity searches.
- **Sentence-Transformers**: Generates embeddings for subtitle texts.
- **ChromaDB**: Stores vectorized subtitle embeddings.
- **Google Generative AI (Gemini API)**: Processes queries and enhances retrieval.
- **Whisper (OpenAI)**: Performs automatic speech recognition (ASR) on audio files.
- **SQLite & Pandas**: Handles structured subtitle data.
- **Zipfile & IO Modules**: Manages compressed subtitle archives.

## Installation
```sh
# Clone the repository
git clone https://github.com/yourgithubusername/movie-dialogue-rag.git
cd movie-dialogue-rag

# Install required dependencies
pip install -r requirements.txt
```

## Usage
### 1. Extract Subtitle Data
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("path/to/your/database.db")
df = pd.read_sql_query("SELECT * FROM zipfiles", conn)
```

### 2. Process Subtitles
```python
import zipfile, io

def decode_method(binary_data):
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            subtitle_content = zip_file.read(zip_file.namelist()[0])
    return subtitle_content.decode('latin-1')

df['file_content'] = df['content'].apply(decode_method)
df.to_csv("decoded_subtitles.csv", index=False)
```

### 3. Create Vector Embeddings & Store
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
videosubtitles_db = Chroma(collection_name="vector_database", embedding_function=embeddings_model, persist_directory="./chroma_db")

# Add documents to the vector database
videosubtitles_db.add_documents(df[['file_content']])
```

### 4. Query the System
```python
query = "I'll be back"
docs_chroma = videosubtitles_db.similarity_search_with_score(query, k=3)
```

### 5. Identify the Movie/Series
```python
from langchain_google_genai import GoogleGenerativeAI

model = GoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key="YOUR_GOOGLE_API_KEY")
output = model.invoke({"context": docs_chroma, "metadata": "Subtitle Metadata"})
print(output)
```

### 6. Speech-to-Text Integration
```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")
audio_path = "path/to/audio/file"
transcription = pipe(audio_path)["text"]
```

## Future Enhancements
- Improve ASR accuracy using fine-tuned Whisper models.
- Integrate an API for real-time queries.

## Contributing
Feel free to submit issues, pull requests, or suggestions to improve the system.

## License
This project is licensed under the MIT License.

---
Developed by **Mydukuri Radha Komalidevi**  
Email: **mrkomalidevi03@gmail.com**  
GitHub: [komali03](https://github.com/komali03)

