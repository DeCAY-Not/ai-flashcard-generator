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
db = chromadb.PersistentClient(path="./chroma_storage")
collection_name = "gate_cse_master"

existing_collections = [c.name for c in db.list_collections()]

if collection_name not in existing_collections:
    print("Master database not found. Scanning PDF folder...")
    
    raw_text = ""
    pdf_folder = "PDF"
    
    # 1. NEW: Loop through EVERY file in the PDF folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Reading {filename}...")
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        raw_text += page.extract_text() + "\n"

    print("Cleaning and chunking data...")
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
    print("✅ Master database built and saved to disk!")
else:
    print("✅ Existing master database found! Loading instantly.")
    collection = db.get_collection(name=collection_name)

# --- PHASE 3 & 4: THE GENERATION LOOP ---
print("\n--- Advanced GATE Flashcard Generator Active ---")
print("Type 'quit' when you are done.")

csv_filename = "gate_master_flashcards.csv"

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        csv.writer(file).writerow(["Question", "Answer"])

while True:
    query = input("\nWhat topic do you want flashcards for? (or 'quit'): ")
    if query.lower() == 'quit':
        break

    print("Searching documents and generating exam-level questions...")
    
    q_vec = client.models.embed_content(model="gemini-embedding-001", contents=query).embeddings[0].values
    
    # 2. NEW: Retrieve more chunks (8 instead of 5) for broader context
    results = collection.query(query_embeddings=[q_vec], n_results=8)
    context = "\n\n---\n\n".join(results["documents"][0])

    # 3. NEW: Aggressive Prompt Engineering for GATE-level rigor
    sys_instruct = """You are a strict examiner writing flashcards for a highly competitive computer science exam. 
Using ONLY the provided context, generate exactly 5 to 7 advanced study flashcards.
Do not ask simple definition questions. Instead, focus on:
- Time and space complexities.
- Mathematical formulas and their specific use cases.
- Edge cases, exceptions, and conceptual traps.
- "What happens if..." scenario questions.

Return strictly valid JSON containing a flat array of objects with keys 'Question' and 'Answer'. Make the answers highly detailed but concise."""

    prompt = f"CONTEXT:\n{context}\n\nUSER QUERY:\n{query}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.2, # Slightly higher temperature to allow for more complex question formulation
                response_mime_type="application/json",
            ),
        )
        
        flashcards = json.loads(response.text)
        
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for card in flashcards:
                writer.writerow([card.get("Question", ""), card.get("Answer", "")])
                
        print(f"✅ Generated {len(flashcards)} advanced cards and saved to {csv_filename}!")
        
    except Exception as e:
        print(f"❌ Error generating cards: {e}")

print("Session closed. Time to review in Anki!")