import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Asistente Operativo Multimodal", page_icon="🤖", layout="centered")

# --- CARGA ULTRA-LIGERA DE LA BASE DE DATOS YA PROCESADA ---
@st.cache_resource
def cargar_agente_ligero():
    # Apuntamos a la carpeta vector_db que subiste desde tu compu
    ruta_db = "./vector_db" 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(ruta_db):
        return None
        
    vector_store = Chroma(persist_directory=ruta_db, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Cargar solo el LLM de texto (ocupa muy poca memoria)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_llm = AutoModelForCausalLM.from_pretrained(model_id)
    
    pipe = pipeline("text-generation", model=model_llm, tokenizer=tokenizer, max_new_tokens=256, temperature=0.2)
    llm = HuggingFacePipeline(pipeline=pipe)

    system_prompt = (
        "Eres un asistente operativo experto. Responde en español basándote ÚNICAMENTE en el contexto provisto. "
        "Si la información no está en el contexto, di amablemente que no la encuentras en los manuales.\n\n"
        "Contexto:\n{context}\n\nPregunta: {input}\nRespuesta:"
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# --- INTERFAZ ---
st.title("🤖 Asistente Operativo Corporativo")
st.markdown("---")

agente = cargar_agente_ligero()

if agente is None:
    st.error("❌ Error: No se encontró la carpeta 'vector_db' en el repositorio. Asegúrate de subirla desde tu computadora.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! He procesado las guías operativas (incluyendo sus imágenes). ¿Qué duda deseas resolver?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.write(message["content"])

    if pregunta_usuario := st.chat_input("Escribe tu duda aquí..."):
        with st.chat_message("user"): st.write(pregunta_usuario)
        st.session_state.messages.append({"role": "user", "content": pregunta_usuario})

        with st.chat_message("assistant"):
            with st.spinner("Analizando manuales..."):
                respuesta = agente.invoke(pregunta_usuario)
                st.write(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})