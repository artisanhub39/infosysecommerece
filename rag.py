import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings  

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Load dataset
csv_file = "ecommerce_purchases_1000.csv"  # Updated CSV variable
df = pd.read_csv(csv_file)
if df.empty:
    raise ValueError("Dataset is empty! Run collect_data.py first.")  # Added check
print(" Loading dataset...")
print(df.head())

# 3. Convert dataset into documents (use "Product" column as content)
print(" Creating documents...")
loader = DataFrameLoader(df, page_content_column="Product")
documents = loader.load()  # Changed variable name from 'docs' to 'documents'

# 4. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitter.split_documents(documents)  # Changed variable name
print(f" Created {len(documents)} chunks")

# 5. Create FAISS vectorstore
embeddings = FakeEmbeddings(size=32)  # testing only
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# 6. Groq client
client = Groq(api_key=api_key)

print("\n Ask your question (type 'exit' to quit):")
while True:
    query = input("> ")
    if query.lower() == "exit":
        break

    # Retrieve relevant docs
    retrieved_docs = retriever.get_relevant_documents(query)  # Changed variable name
    context = "\n".join([d.page_content for d in retrieved_docs])
    print(f"Retrieved {len(retrieved_docs)} documents for context.")  # Added print for clarity

    # Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  
        messages=[
            {"role": "system", "content": "You are an e-commerce product analyst."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
    )

    print(" Answer:", response.choices[0].message.content)
