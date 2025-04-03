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

def load_bootstrap_css():
    """Load Bootstrap CSS from design_spec directory."""
    bootstrap_css_path = os.path.join(DESIGN_SPEC_DIR, "bootstrap.min.css")
    if os.path.exists(bootstrap_css_path):
        with open(bootstrap_css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_bootstrap_js():
    """Load Bootstrap JS from design_spec directory."""
    bootstrap_js_path = os.path.join(DESIGN_SPEC_DIR, "bootstrap.bundle.min.js")
    if os.path.exists(bootstrap_js_path):
        with open(bootstrap_js_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_ui_instructions(filename):
    """Loads UI-specific instructions from a text file matching the input filename."""
    instruction_file = os.path.join(INPUT_DIR, f"{os.path.splitext(filename)[0]}.txt")
    
    if os.path.exists(instruction_file):
        with open(instruction_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return "No specific UI instructions found. Proceed with standard transformation."

# Initialize the LLM
llm = ChatGroq(groq_api_key=my_groq_api_key, model_name="llama-3.3-70b-versatile")


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

    
# ✅ Generate Bootstrap components
def generate_components(generate_component_command):
    retrieved_docs = vectorstore.similarity_search(generate_component_command, k=2)

    # Extract relevant class names and styles to minimize token usage
    design_standard_retrieved = "\n".join([
        "\n".join([line for line in doc.page_content.split("\n") if "." in line][:5])  # Extract first 5 class definitions
        for doc in retrieved_docs
    ])

    # Use the unified prompt for component generation
    formatted_prompt = unified_prompt_template.format(
        design_standard=design_standard_retrieved,
        ui_instructions="Component should be simple, structured, and reusable.",
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


    bootstrap_css = load_bootstrap_css()
    # Construct final HTML output
    final_html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
       <style>
            {bootstrap_css}  /* Embedded Bootstrap CSS */
        </style>
        <title>Generated UI Component</title>
    </head>
    <body>
        {cleaned_response}
    </body>
    </html>
    """

    # Save output with "generate_" prefix
    output_file = os.path.join(OUTPUT_DIR, f"generate_{len(os.listdir(OUTPUT_DIR))}.html")
    print(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)

    # Display transformed output
    st.subheader("Generated Bootstrap Output")
    st.components.v1.html(final_html, height=200, scrolling=True)
    st.text_area("Generated HTML", final_html, height=600)
   


# ✅ Transform full website
def transform_new_website(input_snippet, filename):
    design_standard = retrieve_design_standard(input_snippet)
    # Load UI-specific instructions for the given file
    ui_specific_instructions = load_ui_instructions(filename)
    
    # Use the unified prompt for full transformation
    formatted_prompt = unified_prompt_template.format(
        design_standard=design_standard,
        ui_instructions=ui_specific_instructions,
        input=input_snippet
    )
    
    # Get transformed HTML from LLM
    transformed_response = "".join(chunk.content for chunk in llm.stream(formatted_prompt)).strip()
    
    # Extract HTML content from markdown block if present
    match = re.search(r"```(?:html)?\n(.*?)```", transformed_response, re.DOTALL | re.IGNORECASE)
    if match:
        transformed_response = match.group(1).strip()

    # Load Bootstrap CSS and embed it inside the final HTML
    bootstrap_css = load_bootstrap_css()
    bootstrap_js = load_bootstrap_js()
    
    final_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            {bootstrap_css}  /* Embedded Bootstrap CSS */
        </style>
        <script>
            {bootstrap_js}  /* Embedded Bootstrap JS */
        </script>
        <title>Transformed Website</title>
    </head>
    <body>
        {transformed_response}
    </body>
    </html>
    """

    # Save the output file
    output_file_path = os.path.join(OUTPUT_DIR, f"transformed_{filename}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(final_html)
  
     # Display transformed output
    st.subheader("Transformed Bootstrap Output")
    st.components.v1.html(final_html, height=500, scrolling=True)
    st.text_area("Generated HTML", final_html, height=600)
   


# ✅ Streamlit UI Part

st.title("Transform to Bootstrap design")
with st.sidebar:
    option = st.radio("Select an option:", ["Generate Components", "Create Website"])

if option == "Generate Components":
    st.subheader("Generate UI Components")

    # Initialize session state for dynamic component requests
    if "component_requests" not in st.session_state:
        st.session_state.component_requests = []

    # Function to add a new component
    if st.button("Add Component"):
        st.session_state.component_requests.append({"type": None})  # Placeholder for new component

    # UI for selecting components
    for i, component in enumerate(st.session_state.component_requests):
        with st.expander(f"Component {i+1}"):
            # Select component type
            component_type = st.selectbox(
                "Select Component Type",
                ["Button", "Radio Button", "Dropdown"],
                key=f"component_type_{i}"
            )
            
            # Store selected type
            st.session_state.component_requests[i]["type"] = component_type

            if component_type == "Button":
                button_type = st.selectbox(
                    "Select Button Type",
                    [
                        "Primary", "Secondary", "Success", "Danger",
                        "Warning", "Info", "Light", "Dark", "Link"
                    ],
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

    # Generate all components at once
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

        # Send all prompts at once
        if prompts:
            final_prompt = "\n".join(prompts)
            generate_components(final_prompt)  

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