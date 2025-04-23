import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter , HTMLSemanticPreservingSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup, Tag
import requests

# Load environment variables and API key
load_dotenv()
my_groq_api_key = os.getenv("GROQ_API_KEY")
if not my_groq_api_key:
    raise ValueError("Groq API key not found. Please set it in .env or as an environment variable.")

# Define directories
INPUT_DIR = "input_files"
DESIGN_SPEC_DIR = "design_spec"
OUTPUT_DIR = "output_files"
FAISS_INDEX_PATH = "faiss_index"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_or_create_faiss_index(docs):
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

design_spec_docs = read_files(DESIGN_SPEC_DIR)
vectorstore = load_or_create_faiss_index(design_spec_docs)

def retrieve_design_standard(input_content):
    docs = vectorstore.similarity_search(input_content, k=3)
    return "\n".join(doc.page_content for doc in docs)

def load_bootstrap_css():
    bootstrap_css_path = os.path.join(DESIGN_SPEC_DIR, "bootstrap.min.css")
    if os.path.exists(bootstrap_css_path):
        with open(bootstrap_css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_bootstrap_js():
    bootstrap_js_path = os.path.join(DESIGN_SPEC_DIR, "bootstrap.bundle.min.js")
    if os.path.exists(bootstrap_js_path):
        with open(bootstrap_js_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_ui_instructions(filename):
    instruction_file = os.path.join(DESIGN_SPEC_DIR, f"{os.path.splitext(filename)[0]}.txt")
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return "No specific UI instructions found. Proceed with standard transformation."


llm = ChatGroq(groq_api_key=my_groq_api_key, model_name="gemma2-9b-it")

unified_prompt_template = """You are a UI transformation assistant.
Generate HTML strictly following the provided design standard.

### Rules:
- Use only the classes and styles defined in the design standard (no external styles).
- Provide only the <body> content (omit <html>, <head> unless specified).
- Keep output minimal while maintaining the correct structure.
- Ensure responsiveness and Bootstrap 5 compliance.

### Design Standard:
{design_standard}

### UI-Specific Instructions:
{ui_instructions}

### User Request:
{input}

### Output:
"""

def generate_components(generate_component_command):
    retrieved_docs = vectorstore.similarity_search(generate_component_command, k=2)
    design_standard_retrieved = "\n".join([
        "\n".join([line for line in doc.page_content.split("\n") if "." in line][:5])
        for doc in retrieved_docs
    ])

    formatted_prompt = unified_prompt_template.format(
        design_standard=design_standard_retrieved,
        ui_instructions="Component should be simple, structured, and reusable.",
        input=generate_component_command
    )

    response = ""
    for chunk in llm.stream(formatted_prompt):
        response += chunk.content

    cleaned_response = response.replace("```html", "").replace("```", "").strip()
    bootstrap_css = load_bootstrap_css()

    final_html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
       <style>
            {bootstrap_css}
        </style>
        <title>Generated UI Component</title>
    </head>
    <body>
        {cleaned_response}
    </body>
    </html>
    """

    output_file = os.path.join(OUTPUT_DIR, f"generate_{len(os.listdir(OUTPUT_DIR))}.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)

    st.subheader("Generated Bootstrap Output")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Live Preview")
        st.components.v1.html(final_html, height=500 , scrolling=True)

    with col2:
        st.markdown("### Generated HTML")
        st.text_area("Generated HTML", final_html, height=500)


def fetch_html_as_document(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Document(page_content=response.text, metadata={"filename": url})
    except Exception as e:
        st.error(f"Failed to fetch HTML: {e}")
        return None

def extract_body_only(doc: Document):
    soup = BeautifulSoup(doc.page_content, "html.parser")
    body = soup.body
    if body:
        return Document(page_content=str(body), metadata=doc.metadata)
        #return body
    else:
        st.warning("No <body> found in the page.")
        return None


def save_chunks_to_file(chunks, filename="semantic_chunks_debug.txt"):
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n\n--- Chunk {i+1} ---\n")
            f.write(str(chunk))


def extract_body_chunks(html_content, max_chunk_size=3000):
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.body

    if body is None:
        return []

    chunks = []
    current_chunk = ""
    
    def add_chunk(chunk):
        clean_chunk = chunk.strip()
        if clean_chunk:
            chunks.append(clean_chunk)

    def is_split_point(tag: Tag):
        return tag.name in ['section', 'article', 'div'] and (
            'container' in tag.get('class', []) or
            'row' in tag.get('class', []) or
            'main' in tag.get('class', []) or
            'content' in tag.get('class', []) or
            'contact' in tag.get('id', '') or
            'about' in tag.get('id', '') or
            'footer' in tag.get('id', '')
        )

    for element in body.children:
        if isinstance(element, Tag):
            element_str = str(element)

            # If this element is a semantic split point
            if is_split_point(element) or len(current_chunk) + len(element_str) > max_chunk_size:
                add_chunk(current_chunk)
                current_chunk = element_str
            else:
                current_chunk += element_str

    # Add remaining content
    add_chunk(current_chunk)
    return chunks

def extract_body_chunks_1(body_content):
    if not body_content:
        return []

    soup = BeautifulSoup(body_content, "html.parser")
    body = soup.body

    chunks = []

    for element in body.children:
        if element.name in ["div", "header", "img", "footer"]:
            html_chunk = str(element)
            chunks.append(html_chunk)

    return chunks

def strip_outer_tags(html_block):
    # Removes <html>, <head>, and <body> wrappers if present
    html_block = re.sub(r"<\/?(html|head|body)[^>]*>", "", html_block, flags=re.IGNORECASE)
    return html_block.strip()

def transform_new_website_chunk(body_doc, filename):

    ui_specific_instructions = load_ui_instructions("ui_instruction_set")

    use_RecursiveCharacterTextSplitter = 0

    if use_RecursiveCharacterTextSplitter:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        final_chunks = text_splitter.split_text(str(body_doc.page_content))
    else:
        final_chunks = extract_body_chunks(str(body_doc.page_content))
    
    save_chunks_to_file(final_chunks)
    transformed_chunks = []
    for chunk in final_chunks:
        #input_content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        design_standard = retrieve_design_standard(chunk)
        formatted_prompt = unified_prompt_template.format(
            design_standard=design_standard,
            ui_instructions=ui_specific_instructions,
            input=chunk
        )

        response = "".join(c.content for c in llm.stream(formatted_prompt)).strip()
        match = re.search(r"```(?:html)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            response = match.group(1).strip()

        response = strip_outer_tags(response)
        transformed_chunks.append(response)
 
    #transformed_chunks = deduplicate_navbars(transformed_chunks)
    transformed_response = "\n".join(transformed_chunks)
    #transformed_response = remove_redundant_containers(transformed_response)
    bootstrap_css = load_bootstrap_css()
    bootstrap_js = load_bootstrap_js()

    final_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>{bootstrap_css}</style>
        <script>{bootstrap_js}</script>
        <title>Transformed Website</title>
    </head>
    <body>
        {transformed_response}
    </body>
    </html>
    """

    output_file_path = os.path.join(OUTPUT_DIR, f"transformed_{filename}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    output_file_path = os.path.join(OUTPUT_DIR, f"body_transformed_{filename}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transformed_response)

    st.subheader("Transformed Bootstrap Output")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Live Preview")
        st.components.v1.html(final_html, height=500, scrolling=True)

    with col2:
        st.markdown("### Generated HTML")
        st.text_area("Generated HTML", final_html, height=500)

# Streamlit UI

st.title("Transform to Bootstrap design")
with st.sidebar:
    option = st.radio("Select an option:", ["Generate Components", "Create Website"])

if option == "Generate Components":
    st.subheader("Generate UI Components")

    if "component_requests" not in st.session_state:
        st.session_state.component_requests = []

    if st.button("Add Component"):
        st.session_state.component_requests.append({"type": None})

    for i, component in enumerate(st.session_state.component_requests):
        with st.expander(f"Component {i+1}"):
            component_type = st.selectbox(
                "Select Component Type",
                ["Button", "Radio Button", "Dropdown"],
                key=f"component_type_{i}"
            )
            st.session_state.component_requests[i]["type"] = component_type

            if component_type == "Button":
                button_type = st.selectbox(
                    "Select Button Type",
                    ["Primary", "Secondary", "Success", "Danger",
                     "Warning", "Info", "Light", "Dark", "Link"],
                    key=f"button_type_{i}"
                )
                st.session_state.component_requests[i]["details"] = {"button_type": button_type}

            elif component_type == "Radio Button":
                num_radio = st.number_input("Number of Radio Buttons", min_value=1, max_value=10, value=2, key=f"num_radio_{i}")
                radio_labels = [st.text_input(f"Radio Button {j+1} Label", key=f"radio_label_{i}_{j}") for j in range(num_radio)]
                st.session_state.component_requests[i]["details"] = {"num_radio": num_radio, "radio_labels": radio_labels}

            elif component_type == "Dropdown":
                num_dropdown = st.number_input("Number of Dropdown Options", min_value=1, max_value=10, value=2, key=f"num_dropdown_{i}")
                dropdown_labels = [st.text_input(f"Dropdown Option {j+1} Label", key=f"dropdown_label_{i}_{j}") for j in range(num_dropdown)]
                st.session_state.component_requests[i]["details"] = {"num_dropdown": num_dropdown, "dropdown_labels": dropdown_labels}

    if st.button("Generate Components"):
        prompts = []
        for component in st.session_state.component_requests:
            if component["type"] == "Button":
                prompts.append(f"Generate a {component['details']['button_type'].lower()} button using Bootstrap 5.")
            elif component["type"] == "Radio Button":
                labels = ", ".join(component["details"]["radio_labels"])
                prompts.append(f"Generate a radio button group with {component['details']['num_radio']} options: {labels} using Bootstrap 5.")
            elif component["type"] == "Dropdown":
                labels = ", ".join(component["details"]["dropdown_labels"])
                prompts.append(f"Generate a dropdown with {component['details']['num_dropdown']} options: {labels} using Bootstrap 5.")
        if prompts:
            final_prompt = "\n".join(prompts)
            generate_components(final_prompt)

if option == "Create Website":
    st.subheader("Choose how to transform a web page")

    browse_option = st.selectbox("Browse Mode", ["Browse by URL", "Browse by File"])

    if browse_option == "Browse by URL":
        url_input = st.text_input("Enter a full webpage URL (e.g., https://example.com)")
        if url_input:
            if st.button("Transform from URL"):
                with st.spinner("Fetching and transforming the webpage..."):
                    try:
                        doc = fetch_html_as_document(url_input)
                        body_doc = extract_body_only(doc)
                        if body_doc:
                            transform_new_website_chunk(body_doc, filename=url_input.split("/")[-1] or "url_page.html")
                    except Exception as e:
                        st.error(f"Error processing URL: {e}")

    elif browse_option == "Browse by File":
        uploaded_files = st.file_uploader("Upload HTML, CSS, or JS files", type=["html", "css", "js"], accept_multiple_files=True)
        if uploaded_files:
            docs = []
            for file in uploaded_files:
                try:
                    content = file.read().decode("utf-8")
                    doc = Document(page_content=content, metadata={"filename": file.name})
                    body_doc = extract_body_only(doc)
                    if body_doc:
                        docs.append(body_doc)
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")
            if docs:
                st.success(f"Successfully extracted body content from {len(docs)} file(s)!")
                if st.button("Transform from File(s)"):
                    with st.spinner("Processing..."):
                        for doc in docs:
                            try:
                                transform_new_website_chunk(doc, filename=doc.metadata["filename"])
                            except Exception as e:
                                st.error(f"Error processing {doc.metadata['filename']}: {e}")