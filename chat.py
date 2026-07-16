import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def inicializar_agente():
    print("🤖 Cargando base de datos vectorial y modelo de lenguaje...")
    
    # 1. Conectar a la base de datos vectorial
    ruta_db = "./vector_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(ruta_db):
        print("❌ Error: No se encuentra la base de datos. Ejecuta primero 'ingestar.py'.")
        return None
        
    vector_store = Chroma(persist_directory=ruta_db, embedding_function=embeddings)
    # Configurar el buscador para traer los 3 fragmentos más relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Configurar el LLM Gratuito (Modelo ultra-ligero)
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

    # 3. Diseñar el Prompt
    system_prompt = (
        "Eres un asistente operativo experto. Responde la pregunta del usuario "
        "basándote ÚNICAMENTE en el contexto provisto. Si no sabes la respuesta o no está "
        "en el contexto, di amablemente que no tienes esa información.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {input}\n"
        "Respuesta:"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)

    # Función auxiliar para formatear los documentos encontrados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. Construir la cadena moderna (LCEL) sin usar langchain.chains
    # Esto une el flujo: Busca contexto -> Junta con prompt -> Envía al LLM -> Muestra texto limpio
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Agente listo para chatear.")
    return rag_chain

def iniciar_chat():
    agente = inicializar_agente()
    if not agente:
        return

    print("\n💬 ¡Hola! Soy tu asistente de guías operativas. Escribe 'salir' para terminar.")
    print("-" * 60)
    
    while True:
        pregunta = input("\nTú: ")
        if pregunta.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break
            
        if not pregunta.strip():
            continue
            
        print("🔍 Buscando en los manuales y redactando respuesta...")
        
        # Invocamos la cadena moderna
        respuesta = agente.invoke(pregunta)
        
        print(f"\n🤖 Agente: {respuesta}")

if __name__ == "__main__":
    iniciar_chat()