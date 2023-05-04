"""
Python script with streamlit for frontend, example using: langchain, pypdf & memory
# Author: Diego Maye
# Date: Abril 27, 2023
"""

import re
from io import BytesIO
from typing import List

# Modules to Import
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader


@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Convierte una cadena o lista de cadenas en una lista de documentos
    con metadatos."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache_data
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("Se encuentra Indexando..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings Listo!.", icon="✅")
    return index


st.title("🤖 SwitchGPT personalizado con origen de datos pdf 🧠 ")
st.markdown(
    """ 
        ####  🗨️ Prueba conversacional con archivos PDF usando `Conversational Buffer Memory`  
        > *powered by [switch]('https://switchsoftware.us/') *
        ----
        """
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Subir archivo PDF**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    # pages
    if pages:
        with st.expander("Contenido de las Paginas", expanded=False):
            page_sel = st.number_input(
                label="Seleccionar Pagina", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]
        api = st.secrets["OPENAI_API_KEY"]
        if api:
            # embeddings = OpenAIEmbeddings(openai_api_key=api)
            # # Indexing
            # # Save in a Vector DB
            # with st.spinner("It's indexing..."):
            #     index = FAISS.from_documents(pages, embeddings)
            index = test_embed()
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=api),
                chain_type="stuff",
                retriever=index.as_retriever(),
            )

            # Our tool
            tools = [
                Tool(
                    name="Empresa Switch Sistema de Preguntas y Respuestas",
                    func=qa.run,
                    description="Útil para cuando necesites responder dudas sobre los aspectos planteados. La entrada puede ser una pregunta parcial o completamente formada.",
                )
            ]

            prefix = """Ten una conversación con un ser humano y responde las siguientes preguntas lo mejor que puedas según el contexto y la memoria disponible.
                         Tienes acceso a una única herramienta:"""
            suffix = """Comenzar!"

            {chat_history}
            Pregunta: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history"
                )

            # Chain
            # Zero shot agent
            # Agen Executor
            llm_chain = LLMChain(
                llm=OpenAI(
                    temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                ),
                prompt=prompt,
            )
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            )

            query = st.text_input(
                "**En que estas pensando?**",
                placeholder="Preguntame cualquier cosa sobre {}".format(name_of_file),
            )

            if query:
                with st.spinner(
                    "Generando respuestas para tu cunsulta Query : `{}` ".format(query)
                ):
                    res = agent_chain.run(query)
                    st.info(res, icon="🤖")

            with st.expander("History/Memory"):
                st.session_state.memory
