import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables and API key
load_dotenv()
# my_groq_api_key = "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"
my_groq_api_key = st.secrets["my_groq_api_key"]

if not my_groq_api_key:
    raise ValueError("Groq API key not found. Please set it in .env or as an environment variable.")

# Define directories
INPUT_DIR = "input_files"
DESIGN_SPEC_DIR = "design_spec"
OUTPUT_DIR = "output_files"
FAISS_INDEX_PATH = "faiss_index"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to read files from a directory
def read_files(directory, allowed_extensions={".html", ".css", ".js"}):
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in allowed_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        docs.append(Document(page_content=content, metadata={"filename": file}))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return docs

# Initialize embeddings for FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load or create FAISS index
def load_or_create_faiss_index(docs):
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

# Read design spec files and create FAISS index
design_spec_docs = read_files(DESIGN_SPEC_DIR)
vectorstore = load_or_create_faiss_index(design_spec_docs)

def retrieve_design_standard(input_content):
    docs = vectorstore.similarity_search(input_content, k=3)
    return "\n".join(doc.page_content for doc in docs)

# Initialize the LLM
llm = ChatGroq(groq_api_key=my_groq_api_key, model_name="gemma2-9b-it")

# Optimized system prompt for component generation
system_prompt = (
    "You are a UI assistant that generates HTML components strictly following the design standard.\n\n"
    "**Rules:**\n"
    "- Use only the classes and styles defined in the design standard (no external styles).\n"
    "- Provide only the <body> content (omit <html>, <head>).\n"
    "- Keep output minimal while maintaining the correct structure.\n\n"
    "**Relevant Design Standard Styles:**\n{design_standard_retrieved}\n\n"
    "**User Request:**\n{input}"
)

# System prompt for full website transformation
common_system_prompt = """You are a UI transformation expert.
Transform the provided Penpot-generated HTML into a single, self-contained HTML file that complies with Bootstrap 5 standards.
Ensure that the transformed HTML:
- Uses modern HTML5 best practices and a clean, semantic structure.
- Includes a single local reference to Bootstrap 5 CSS in the <head>:
  <link rel="stylesheet" href="bootstrap.min.css">
- Embeds only minimal custom CSS inside a <style> tag.
- Removes extraneous or redundant inline styles.
- Produces a visually structured and responsive layout.
- Adheres to the design standards provided.
"""

ui_specific_instructions = """
Ensure the following elements are present in the transformed output:
- A navigation bar with links labeled "Home", "Contact", "Service", and "About".
- A header that displays the text "Welcome To UiAutomation".
- A section titled "Convert design to code".
- An interactive area containing a "Try It Now" checkbox and two buttons labeled "Learn More" and "Get Now".
"""

final_system_prompt_template = """{common_prompt}

### Design Standard:
{design_standard}

### Additional UI Specifications:
{ui_instructions}

### Original Penpot HTML (truncated):
{input}

### Output (Single Transformed HTML File):
"""

# ✅ Generate Bootstrap components
def generate_components(generate_component_command):
    retrieved_docs = vectorstore.similarity_search(generate_component_command, k=2)

    # Extract relevant class names and styles to minimize token usage
    design_standard_retrieved = "\n".join([
        "\n".join([line for line in doc.page_content.split("\n") if "." in line][:5])  # Extract first 5 class definitions
        for doc in retrieved_docs
    ])

    # Format prompt
    formatted_prompt = system_prompt.format(
        design_standard_retrieved=design_standard_retrieved,
        input=generate_component_command
    )

    # Stream output efficiently
    #response_placeholder = st.empty()
    response = ""

    for chunk in llm.stream(formatted_prompt):
        response += chunk.content
        #response_placeholder.markdown(response, unsafe_allow_html=True)  # Update dynamically

    # Clean up response (remove markdown formatting)
    cleaned_response = response.replace("```html", "").replace("```", "").strip()

    # Construct final HTML output
    final_html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <link rel='stylesheet' href='bootstrap.min.css'>
        <script src='bootstrap.bundle.min.js'></script>
        <title>Generated UI Component</title>
    </head>
    <body>
        {cleaned_response}
    </body>
    </html>
    """

    # Display generated component
    st.subheader("Generated Bootstrap Component")
    st.components.v1.html(final_html, height=500, scrolling=True)

    # Save output with "generate_" prefix
    output_file = os.path.join(OUTPUT_DIR, f"generate_{len(os.listdir(OUTPUT_DIR))}.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)

# ✅ Transform full website
def transform_new_website(input_snippet, filename):
    design_standard = retrieve_design_standard(input_snippet)
    
    final_system_prompt = final_system_prompt_template.format(
        common_prompt=common_system_prompt,
        design_standard=design_standard,
        ui_instructions=ui_specific_instructions,
        input=input_snippet
    )
    
    transformed_response = "".join(chunk.content for chunk in llm.stream(final_system_prompt)).strip()
    
    match = re.search(r"```(?:html)?\n(.*?)```", transformed_response, re.DOTALL | re.IGNORECASE)
    if match:
        transformed_response = match.group(1).strip()
    
    output_file_path = os.path.join(OUTPUT_DIR, f"transformed_{filename}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transformed_response)
    
    # Display transformed output
    st.subheader("Transformed Bootstrap Output")
    st.components.v1.html(transformed_response, height=500, scrolling=True)

# ✅ Streamlit UI Part

st.title("Transform to Bootstrap design")
with st.sidebar:
    option = st.radio("Select an option:", ["Generate Components", "Create Website"])

if option == "Generate Components":
    generate_component_command = st.text_area("Enter HTML/CSS or request a UI component:", "Generate different types of buttons")
    if st.button("Generate"):
        if generate_component_command.strip():  # Fixed indentation
            generate_components(generate_component_command)  

if option == "Create Website":
    st.subheader("Upload your files")
    uploaded_files = st.file_uploader("Upload HTML, CSS, or JS files", type=["html", "css", "js"], accept_multiple_files=True)
    if uploaded_files:
        docs = []
        for file in uploaded_files:
            try:
                content = file.read().decode("utf-8")
                docs.append(Document(page_content=content, metadata={"filename": file.name}))
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
        if docs:
            st.success(f"Successfully uploaded {len(docs)} files!")
            if st.button("Transform with LLM"):
                with st.spinner("Processing files..."):
                    for doc in docs:
                        try:
                            input_snippet = doc.page_content[:4000]
                            transform_new_website(input_snippet, doc.metadata["filename"])
                        except Exception as e:
                            st.error(f"Error processing {doc.metadata['filename']}: {e}")