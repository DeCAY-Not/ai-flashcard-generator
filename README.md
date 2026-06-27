# AI-Powered GATE Flashcard Generator

This project is an automated study tool designed to streamline GATE 2027 preparation. It uses Retrieval-Augmented Generation (RAG) to transform technical study materials (PDFs) into structured, Q&A-style flashcards ready for import into AnkiDroid.

## Technology Stack
- Python 3.10+
- pdfplumber: Used to programmatically extract raw text from PDF documents.
- LangChain Text Splitters: Utilized to divide large documents into overlapping, context-aware chunks, preventing data loss at segment boundaries.
- Google GenAI SDK: Powers the embedding process (converting text to mathematical vectors) and the generative intelligence (Gemini 2.5 Flash) that formats study data.
- ChromaDB: A local vector database that indexes your study material for instant, semantic retrieval.
- Dotenv: Manages API security by keeping sensitive credentials outside of the codebase.

## Step-by-Step Workflow
1. Ingestion: The script reads the specified PDF page-by-page, extracts all text, and cleans it by removing empty lines and excess whitespace.
2. Chunking: The cleaned text is broken into 500-character segments with a 50-character overlap, ensuring that important concepts or formulas are never cut off between chunks.
3. Embedding: Each chunk is sent to Gemini to be converted into a high-dimensional vector representing its semantic meaning.
4. Storage: These vectors and the original text chunks are stored in a persistent ChromaDB database on your local machine, allowing for instant reuse across different study sessions.
5. Retrieval: When you input a topic, the script embeds your query and searches ChromaDB for the 5 most relevant chunks of text from your PDF.
6. Generation: These chunks are sent to Gemini 2.5 Flash with a strict system instruction to return a JSON array containing 'Question' and 'Answer' pairs.
7. Export: The JSON output is parsed by Python and appended to a CSV file in the project root, which can be imported directly into AnkiDroid.

## How to Run
1. Ensure your virtual environment is active (.venv).
2. Place your target PDF in the PDF/ folder (named DS.pdf).
3. Ensure your GEMINI_API_KEY is defined in a .env file.
4. Run the generator: python main.py
5. Follow the terminal prompts to enter study topics and generate cards.
6. Once finished, import the gate_flashcards.csv file into Anki Desktop, then sync your account.