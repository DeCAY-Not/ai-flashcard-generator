import pdfplumber
import os
import chromadb
import json
import csv
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = genai.Client()

# --- SETUP PERSISTENT DATABASE ---
# This saves the vectors to your hard drive so you don't rebuild it every time!
db = chromadb.PersistentClient(path="./chroma_storage")
collection_name = "gate_ds_notes"

# Check if we already built this database
existing_collections = [c.name for c in db.list_collections()]

if collection_name not in existing_collections:
    print("Database not found. Building it for the first time...")
    
    pdf_path = "PDF/DS.pdf"
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                raw_text += page.extract_text() + "\n"

    cleaned_text = "\n".join([line.strip() for line in raw_text.split("\n") if line.strip()])
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(cleaned_text)
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0: print(f"  Embedding chunk {i}/{len(chunks)}...")
        response = client.models.embed_content(model="gemini-embedding-001", contents=chunk)
        embeddings.append(response.embeddings[0].values)

    collection = db.create_collection(name=collection_name)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks
    )
    print("✅ Database built and saved to disk!")
else:
    print("✅ Existing database found! Skipping extraction and loading instantly.")
    collection = db.get_collection(name=collection_name)

# --- PHASE 3 & 4: THE GENERATION LOOP ---
print("\n--- RAG Flashcard Generator Active ---")
print("Type 'quit' when you are done.")

csv_filename = "gate_flashcards.csv"

# Create the CSV with headers if it doesn't exist yet
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerow(["Question", "Answer"])

# Keep asking for topics until you type 'quit'
while True:
    query = input("\nWhat topic do you want flashcards for? (or 'quit'): ")
    if query.lower() == 'quit':
        break

    print("Searching document and generating...")
    
    # Retrieve
    q_vec = client.models.embed_content(model="gemini-embedding-001", contents=query).embeddings[0].values
    results = collection.query(query_embeddings=[q_vec], n_results=5)
    context = "\n\n---\n\n".join(results["documents"][0])

    # Generate
    sys_instruct = "You are a GATE CSE study assistant. Using ONLY the context, generate flashcards. Return strictly valid JSON containing a flat array of objects with keys 'Question' and 'Answer'."
    prompt = f"CONTEXT:\n{context}\n\nUSER QUERY:\n{query}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        
        flashcards = json.loads(response.text)
        
        # Append to CSV (mode='a') so we don't overwrite previous cards
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for card in flashcards:
                writer.writerow([card.get("Question", ""), card.get("Answer", "")])
                
        print(f"✅ Added {len(flashcards)} new cards to {csv_filename}!")
        
    except Exception as e:
        print(f"❌ Error generating cards: {e}")

print("Session closed. Good luck with the GATE prep!")