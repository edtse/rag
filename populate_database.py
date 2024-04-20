import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
import hashlib


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    
    documents = []
    
    document_loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)    
    documents.extend(document_loader.load())

    document_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFDirectoryLoader)    
    documents.extend(document_loader.load())

    document_loader = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)    
    documents.extend(document_loader.load())

    # Walk through directory recursively
    # for root, dirs, files in os.walk(DATA_PATH):
    #     # Filter and load PDF documents
    #     pdf_files = [file for file in files if file.endswith('.pdf')]
    #     for pdf_dir in dirs:
    #         full_path = os.path.join(root, pdf_dir)
    #         document_loader = PyPDFDirectoryLoader(full_path)
    #         documents.extend(document_loader.load())

        # # Filter and load Markdown documents
        # md_files = [file for file in files if file.endswith('.md') or file.endswith('.MD')]
        # for md_file in md_files:
        #     full_path = os.path.join(root, md_file)
        #     document_loader = UnstructuredMarkdownLoader(full_path)
        #     documents.extend(document_loader.load())

        # Filter and load Markdown documents
        # word_files = [file for file in files if file.endswith('.doc') or file.endswith('.docx')]
        # for word_file in word_files:
        #     full_path = os.path.join(root, md_file)
        #     document_loader = AzureAIDocumentIntelligenceLoader(full_path)
        #     documents.extend(document_loader.load())

    return documents

def calculate_sha1(filepath):
    sha1 = hashlib.sha1()
    try:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(65536)  # Read in chunks of 64KB
                if not data:
                    break
                sha1.update(data)
    except FileNotFoundError:
        print("File not found")
        return None
    return sha1.hexdigest()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # Collect IDs from the current chunks.
    current_ids = {chunk.metadata["id"] for chunk in chunks_with_ids}
    # Find and delete documents no longer present in the new set.
    obsolete_ids = existing_ids - current_ids
    if obsolete_ids:
        print(f"👉 Deleted documents: {len(obsolete_ids)}")
        db.delete(ids=list(obsolete_ids))
        db.persist()
        
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        sha1_hash = calculate_sha1(source)  # Calculate if not provided
        current_page_id = f"{source}:{page}:{sha1_hash}"        

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
