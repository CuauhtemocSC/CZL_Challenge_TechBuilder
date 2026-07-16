import os

from langchain_community.document_loaders import Docx2txtLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def procesar_documentos():
    print("📂 Iniciando la lectura de guías operativas...")
    
    carpeta_docs = "./documentos"
    documentos_cargados = []
    
    if not os.path.exists(carpeta_docs):
        os.makedirs(carpeta_docs)
        print("📁 Se creó la carpeta './documentos'. Por favor guarda tus archivos .docx ahí.")
        return
    
    # 1. Buscar todos los archivos .docx
    for archivo in os.listdir(carpeta_docs):
        if archivo.endswith(".docx"):
            ruta_completa = os.path.join(carpeta_docs, archivo)
            print(f"📖 Cargando: {archivo}")
            # Usamos el cargador correcto
            loader = Docx2txtLoader(ruta_completa)
            documentos_cargados.extend(loader.load())
            
    if not documentos_cargados:
        print("⚠️ No se encontraron archivos .docx en la carpeta './documentos'. ¡Agrega al menos uno!")
        return

    # 2. Particionamiento inteligente (Chunking)
    print("✂️ Dividiendo los documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos_cargados)
    print(f"🧩 Creados {len(fragmentos)} fragmentos de texto.")

    # 3. Inicializar el modelo de Embeddings (100% Gratuito y Local)
    print("🤖 Inicializando el modelo de embeddings en tu CPU...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Crear y almacenar en la Base de Datos Vectorial (Chroma)
    print("💾 Guardando fragmentos en la base de datos vectorial...")
    ruta_db = "./vector_db"
    
    vector_store = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=ruta_db
    )
    
    print("✅ ¡Base de datos vectorial creada y guardada con éxito en './vector_db'!")

if __name__ == "__main__":
    procesar_documentos()