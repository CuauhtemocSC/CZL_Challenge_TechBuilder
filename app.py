import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Configuración de la página web
st.set_page_config(page_title="Asistente Operativo IA", page_icon="🤖", layout="centered")

# Usamos la caché de Streamlit para cargar el modelo e índices una sola vez y ahorrar memoria
@st.cache_resource
def cargar_agente():
    ruta_db = "./vector_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(ruta_db):
        return None
        
    vector_store = Chroma(persist_directory=ruta_db, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Modelo ultra-ligero ideal para correr localmente
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256, 
        temperature=0.2
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    system_prompt = (
        "Eres un asistente operativo experto. Responde la pregunta del usuario "
        "basándote ÚNICAMENTE en el contexto provisto. Si no sabes la respuesta o no está "
        "en el contexto, di amablemente que no tienes esa información en los manuales.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {input}\n"
        "Respuesta:"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- INTERFAZ DE USUARIO ---
st.title("🤖 Asistente de Guías Operativas")
st.subheader("Consulta tus manuales y resuelve dudas en segundos")
st.markdown("---")

# Inicializar el agente RAG
with st.spinner("Cargando el motor de IA y base de datos..."):
    agente = cargar_agente()

if agente is None:
    st.error("❌ No se encontró la base de datos vectorial en `./vector_db`. Por favor, ejecuta primero `python ingestar.py` en tu terminal.")
else:
    # Inicializar el historial de chat en la sesión si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! He leído tus guías operativas. ¿En qué proceso te puedo ayudar hoy?"}
        ]

    # Mostrar los mensajes anteriores del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Capturar la entrada del usuario
    if pregunta_usuario := st.chat_input("Escribe tu duda sobre los manuales aquí..."):
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.write(pregunta_usuario)
        st.session_state.messages.append({"role": "user", "content": pregunta_usuario})

        # Generar respuesta del agente
        with st.chat_message("assistant"):
            with st.spinner("Buscando en manuales..."):
                respuesta_completa = agente.invoke(pregunta_usuario)
                st.write(respuesta_completa)
        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})