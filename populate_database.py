import argparse
import os
import glob
import shutil
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import UnstructuredFileLoader
import pickle
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from img2table.ocr import TesseractOCR
from img2table.document import PDF

# Stored at local paths.
CHROMA_PATH = "chroma"
DATA_PATH = "./data/"
PARSING_DATA_PATH = "./parsing_data/"
CHUNKS_PATH = "processed_chunks.pkl"
PARSED_FILES_LIST = "parsed_files.json"

# LlamaParse API key from .env file.
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

def main():
    """
    Main function that runs the Chroma database population process.

    To add files with text paragraphs, change the extract_tables_from_pdf() call with load_documents() and run 'python populate_database.py' without the reset flag.

    Run the populate_database.py script with the '--reset" flag to fully clear the database before adding files in the data path.
    """
    load_dotenv()
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    os.makedirs('llama_parsed', exist_ok=True)
    os.makedirs(PARSING_DATA_PATH, exist_ok=True)

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Update data store.
    documents = extract_tables_from_pdf(DATA_PATH) # switch this with the load_documents() function for PDF files with text paragraphs.
    chunks = split_documents(documents)
    save_chunks(chunks)
    add_to_chroma(chunks)

def extract_tables_from_pdf(directory_path):
    """
    Extract ONLY tables from a PDF document using img2table and return structured data - use for PDFs with only tables.

    If the PDF contains text paragraphs, this function will not extract it. Instead, use the load_documents() functionLlamaParse.
    
    Returns:
        List[dict]: A dictionary of extracted tables keyed by file name.
    """
    parsed_files = load_parsed_files()
    files_to_parse = []

    for file in os.listdir(DATA_PATH):
        if file not in parsed_files:
            files_to_parse.append(file)
            shutil.copy(os.path.join(DATA_PATH, file), PARSING_DATA_PATH)

    if not files_to_parse:
        print("No new files to parse.")
        return load_parsed_documents()
    if (len(files_to_parse) == 1):
        print(f"Parsing {len(files_to_parse)} new file...")
    else:
        print(f"Parsing {len(files_to_parse)} new files...")
    ocr = TesseractOCR(n_threads=1, lang="eng")
    all_extracted_tables = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            doc = PDF(pdf_path)
            extracted_tables = doc.extract_tables(ocr=ocr, implicit_rows=False, borderless_tables=False, min_confidence=50)
            
            all_extracted_tables.append({filename: extracted_tables})
            
            with open('llama_parsed/output.md', 'a') as f:
                for key, value_list in extracted_tables.items():
                    for value in value_list:
                        table_df = value.df
                        f.write(f"Table {key} from {filename}:\n")
                        f.write(table_df.to_string())
                        f.write("\n\n")
    
    parsed_files.extend(files_to_parse)
    save_parsed_files(parsed_files)

    for file in os.listdir(PARSING_DATA_PATH):
        os.remove(os.path.join(PARSING_DATA_PATH, file))
    return load_parsed_documents()

def load_parsed_documents():
    """
    Load all parsed documents from the 'output.md' file.

    Returns:
        List[Documents]: A list containing the parsed content.
    """
    loader = UnstructuredFileLoader('llama_parsed/output.md')
    documents = loader.load()
    print("Loaded parsed documents")
    return documents

def split_documents(documents: list[Document]):
    """
    Split the input documents into smaller chunks for processing.

    Args:
        documents (List[Document]): A list of documents to be split.

    Returns:
        List[Document]: A list containing the split chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"]
    )
    return text_splitter.split_documents(documents)

def save_chunks(chunks: list[Document]):
    """
    Save processed document chunks to a pickle file.

    Args:
        chunks (List[Document]): A list containing the processed chunks.
    """
    with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
    print(f"✅ Saved {len(chunks)} chunks to {CHUNKS_PATH}")

def load_chunks():
    """
    Load previously saved document chunks from the pickle file.

    Returns:
        None: If no saved chunks are found

        or

        List[Document]: A list containing the processed chunks.
    """
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
        return chunks
    else:
        print("❌ No saved chunks found.")
        return None

def load_parsed_files():
    """
    Load the list of previously parsed files.

    Returns:
        List[str]: A list of filenames that have already been parsed.
    """
    if os.path.exists(PARSED_FILES_LIST):
        with open(PARSED_FILES_LIST, 'r') as f:
            return json.load(f)
    return []

def load_documents():
    """
    Load and parse new documents from the data directory using LlamaParse - use for files with text / paragraph.

    This function uses LlamaParse to process PDF files into a structured format.

    Returns:
        List[Documents]: A list containing the parsed content.
    """
    parsed_files = load_parsed_files()
    files_to_parse = []

    for file in os.listdir(DATA_PATH):
        if file not in parsed_files:
            files_to_parse.append(file)
            shutil.copy(os.path.join(DATA_PATH, file), PARSING_DATA_PATH)

    if not files_to_parse:
        print("No new files to parse.")
        return load_parsed_documents()
    if (len(files_to_parse) == 1):
        print(f"Parsing {len(files_to_parse)} new file...")
    else:
        print(f"Parsing {len(files_to_parse)} new files...")
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",
        parsing_instruction="""
            Using the format Column Name : Field Value, convert all of the fields with headers of one row into one single line of text. 
            Example:
                S.No: 1; State / UT: Assam; Location: Guwahati, Employment Exchange; Address: District Employment Exchange, Guwahati AK Azad Road, Rehabari, Guwahati-8; Type Of Center: Employment Exchange.
                S.No: 10; JSS NAME: JSS Ongole; STATE: Andhra Pradesh; DISTRICT: Prakasham; ADDRESS: H.No.3-119/1, Satyanarayanapuram, 2nd Lane, Ongole, MARKAPUR-523002, PRAKASAM (Andhra Pradesh); EMAIL: jss.ongole@gmail.com; MOBILE: 8333046955; COURSES THEY ARE OFFERING: Handicrafts & Carpets; PIN CODE: 523002; Type of Training Centre: JSS.
            Use semicolons to separate different fields.
            """,
        skip_diagonal_text=True
    )
    print("loaded parser")
    file_extractor = {".pdf": parser}
    llama_documents = SimpleDirectoryReader(input_dir=PARSING_DATA_PATH, file_extractor=file_extractor).load_data()
    print("created llama_parse documents")
    with open('llama_parsed/output.md', 'a') as f: 
        for doc in llama_documents:
            f.write(doc.text + '\n')
    
    parsed_files.extend(files_to_parse)
    save_parsed_files(parsed_files)

    for file in os.listdir(PARSING_DATA_PATH):
        os.remove(os.path.join(PARSING_DATA_PATH, file))

    return load_parsed_documents()

def save_parsed_files(parsed_files):
    """
    Save the list of parsed files.

    Args:
        parsed_files (List[str]): A list of filenames that have been parsed.
    """
    with open(PARSED_FILES_LIST, 'w') as f:
        json.dump(parsed_files, f)

def add_to_chroma(chunks: list[Document]):
    """
    Add new document chunks to the Chroma vector store.

    This function only adds new documents that don't already exist in the database using chunk IDs.

    Args:
        chunks (List[Document]): A list of document chunks to be added to the database.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # only add documents that don't exist in the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for each document chunk.

    Page Source : Page Number : Chunk Index 
    "data/Model_Career_Centres.pdf:6:2"

    Args:
        chunks (List[Document]): A list of document chunks objects to assign IDs to.

    Returns:
        List[Document]: The input list of document chunk objects with added 'id' metadata.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def get_embedding_function():
    """
    Get the FastEmbed embedding function for document embeddings.

    Returns:
        FastEmbedEmbeddings: An instance of FastEmbedEmbeddings for creating document embeddings.
    """
    embeddings = FastEmbedEmbeddings()
    return embeddings

def clear_database():
    """
    Clear the existing database, chunks, and parsed files.

    This function removes all previously processed local input (except for files in the /data directory).
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)
    output_path = 'llama_parsed/output.md'
    if os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write('')
    if os.path.exists(PARSED_FILES_LIST):
        os.remove(PARSED_FILES_LIST)
    if os.path.exists(PARSING_DATA_PATH):
        shutil.rmtree(PARSING_DATA_PATH)
        os.makedirs(PARSING_DATA_PATH)
    print("✨ Cleared database, chunks, parsed files, and output.md")

if __name__ == "__main__":
    main()