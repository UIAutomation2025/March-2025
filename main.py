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

    #cleaned_response = response.replace("```html", "").replace("```", "").strip()
    bootstrap_css = load_bootstrap_css()
    cleaned_response = ""

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

headers_to_split_on = [
    ("body", "body"),
]


def extract_semantic_blocks(html_str):
    soup = BeautifulSoup(html_str, "html.parser")
    semantic_blocks = []

    # High-level semantic sections
    primary_tags = ["header", "nav", "main", "section", "article", "footer", "form"]
    for tag in primary_tags:
        semantic_blocks.extend(soup.find_all(tag))

    # Containers with meaningful IDs
    id_keywords = ["band", "tour", "contact", "about", "services", "gallery"]
    for div in soup.find_all("div", id=True):
        if any(keyword in div["id"].lower() for keyword in id_keywords):
            semantic_blocks.append(div)

    # W3.CSS class-based containers
    class_keywords = ["w3-top", "w3-container", "w3-content", "w3-padding", "w3-row", "w3-center"]
    for div in soup.find_all("div", class_=True):
        if any(cls for cls in div["class"] if any(kw in cls for kw in class_keywords)):
            semantic_blocks.append(div)

    # Wrap semantic elements in their parent if the parent is a meaningful container
    final_blocks = []
    seen = set()
    for block in semantic_blocks:
        # Get the highest parent that’s a <div> or <section> containing this block
        parent = block
        while parent.parent and parent.parent.name in ["div", "section"]:
            parent = parent.parent
        html_str = str(parent)
        if html_str not in seen:
            seen.add(html_str)
            final_blocks.append(html_str)

    return final_blocks


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


def strip_outer_tags(html_block):
    # Removes <html>, <head>, and <body> wrappers if present
    html_block = re.sub(r"<\/?(html|head|body)[^>]*>", "", html_block, flags=re.IGNORECASE)
    return html_block.strip()

def deduplicate_navbar(chunks):
    seen_nav = False
    deduped_chunks = []
    for chunk in chunks:
        if "<nav" in chunk.page_content.lower():
            if not seen_nav:
                deduped_chunks.append(chunk)
                seen_nav = True
            # Skip additional navbars
        else:
            deduped_chunks.append(chunk)
    return deduped_chunks

def save_chunks_to_file(chunks, filename="semantic_chunks_debug.txt"):
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"\n\n--- Chunk {i+1} ---\n")
            f.write(str(chunk))

# Fix overly nested containers
def remove_redundant_containers(html):
    return re.sub(r'(<div class="container">\s*){2,}', r'<div class="container">', html)

def deduplicate_navbars(transformed_chunks):
    seen = False
    deduped = []
    for chunk in transformed_chunks:
        if "<nav" in chunk and not seen:
            deduped.append(chunk)
            seen = True
        elif "<nav" not in chunk:
            deduped.append(chunk)
    return deduped




# Tags that usually represent logical "chunks"
LOGICAL_BLOCK_TAGS = {'header', 'nav', 'main', 'section', 'article', 'aside', 'footer', 'div'}

def split_logically_ordered(body_tag: Tag):
    logical_chunks = []
    current_chunk = []

    def flush_chunk():
        if current_chunk:
            combined_html = "".join(str(el) for el in current_chunk)
            logical_chunks.append(combined_html)
            current_chunk.clear()

    for child in body_tag.children:
        if isinstance(child, Tag):
            if child.name in LOGICAL_BLOCK_TAGS:
                flush_chunk()
                logical_chunks.append(str(child))
            else:
                current_chunk.append(child)
        else:
            current_chunk.append(child)  # could be NavigableString

    flush_chunk()
    return logical_chunks

def extract_body_chunks_new(body):
    """
    Extracts top-level chunks from the body using start and end tags of each direct child.
    This method does not rely on specific tag names and works generically.

    Args:
        body (Tag): A BeautifulSoup <body> tag.

    Returns:
        List[Tuple[int, str]]: List of (chunk_id, html_string).
    """
    if not body:
        return []

    chunks = []
    chunk_id = 0

    for element in body.children:
        if isinstance(element, Tag):
            html_chunk = str(element)
            chunks.append((chunk_id, html_chunk))
            chunk_id += 1

    return chunks

def extract_body_chunks(body):
    
    if not body:
        return []

    chunks = []
    chunk_id = 0

    for element in body.children:
        if element.name in ["div", "header", "img", "footer","script"]:
            html_chunk = str(element)
            chunks.append((chunk_id, html_chunk))
            chunk_id += 1

    return chunks


def transform_new_website_chunk(body_doc, filename):

    if isinstance(body_doc, str):
        body_doc = BeautifulSoup(body_doc, 'html.parser')

    ui_specific_instructions = load_ui_instructions("ui_instruction_set")
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    final_chunks = text_splitter.split_text(str(body_doc.page_content))

    #semantic_blocks = extract_semantic_blocks(body_doc.page_content)
    #final_chunks = [Document(page_content=block) for block in semantic_blocks]

    #final_chunks = extract_body_chunks(body_doc)
    #final_chunks = [Document(page_content=block) for block in semantic_blocks]


    save_chunks_to_file(final_chunks)


    # semantic_splitter = HTMLSemanticPreservingSplitter(headers_to_split_on=headers_to_split_on)
    # semantic_chunks = semantic_splitter.split_text(input_snippet)

    
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=30)
    # for chunk in semantic_chunks:
    #     final_chunks.extend(text_splitter.split_text(chunk.page_content))

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
