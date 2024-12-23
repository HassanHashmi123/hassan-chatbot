import os
import pickle
import streamlit as st
from secret import gemini_api_key
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import faiss

# Set the Google Gemini API key
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Initialize Google Gemini LLM with specific configurations
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Specify the model to use
    temperature=0.8    
)

# Sidebar Contents
with st.sidebar:
    st.title('🤖 LLM Chat App')
    st.markdown('''
    ### About
    This is Google Gemini powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Google Gemini](https://cloud.google.com/ai)
    ''')
    add_vertical_space(5)
    st.write('Made By:')
    st.write('[Hassan Mujeeb Hashmi](https://your-link.com)')

# Title
st.title("Chat with your PDF using")
st.title("Hassan Chat!")

# File uploader
pdf = st.file_uploader("Upload your PDF document:", type="pdf")

if pdf is not None:
    # Read the pdf file and store it in a variable
    pdf_reader = PdfReader(pdf)
    
    # Combining all the pages' text of the pdf file into a single text
    doc = ""
    for page in pdf_reader.pages:
        doc += page.extract_text()
    
    # LangChain chunking of the large document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )    
    chunks = text_splitter.split_text(text=doc)
    
    # 1. Take the name of the PDF uploaded
    st.write(pdf.name)
    file_name = pdf.name[:-4]
    
    # Paths for saving the FAISS index and document mappings
    index_file_path = f"{file_name}_faiss.index"
    metadata_file_path = f"{file_name}_faiss_meta.pkl"
    
    # Check if the FAISS index file and metadata exist
    if os.path.exists(index_file_path) and os.path.exists(metadata_file_path):
        # Load the FAISS index
        st.write("Loading FAISS index and metadata from disk...")
        index = faiss.read_index(index_file_path)
        
        # Load the index-to-docstore mapping and the docstore (chunks)
        with open(metadata_file_path, "rb") as f:
            data = pickle.load(f)
            index_to_docstore_id = data['index_to_docstore_id']
            docstore = data['docstore']  # These are your chunks
        
        # Recreate the FAISS vector store with the loaded components
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        VectorStores = FAISS(embeddings.embed_query, index, docstore, index_to_docstore_id)
        st.write("FAISS index and metadata loaded successfully.")
    else:
        # Create embeddings using Google Gemini and store them in FAISS vector store
        st.write("Creating FAISS index and embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        VectorStores = FAISS.from_texts(chunks, embeddings)
        
        # Save the FAISS index to disk
        faiss.write_index(VectorStores.index, index_file_path)
        
        # Save the index-to-docstore-id mapping and docstore (metadata) to disk
        data = {
            'index_to_docstore_id': VectorStores.index_to_docstore_id,
            'docstore': VectorStores.docstore
        }
        with open(metadata_file_path, "wb") as f:
            pickle.dump(data, f)
        
        st.write("FAISS index and metadata saved successfully.")
    
    # Get the user's query
    query = st.text_input("Ask your question")
    st.write(query)
        
    if query:
        # Search for the most relevant documents from FAISS
        docs = VectorStores.similarity_search(query=query, k=30)
        st.write(docs)
        
        # Create a custom prompt template for the question-answering task
        prompt_template = """
        You are a helpful assistant. Given the following context, answer the questions in a friendly manner. You will have to explain each and every single point in very simplistic details but very cautiously so that the user will understand completely with very basic examples.

        Context:
        {context}

        Question:
        {question}
        """
        
        # Create a PromptTemplate with two input variables: context and question
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )
        
        # Combine the document content into a single context string
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create the LLMChain with the custom prompt
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain with the extracted context and user query
        response = chain.run({
            "context": context,
            "question": query
        })
        
        # Display the response
        response_length = len(response)
        st.write(response_length)
        st.write(response)
        print(response)
