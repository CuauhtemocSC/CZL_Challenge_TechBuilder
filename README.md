# 🤖 Agente IA RAG: Consultor de Guías Operativas

Este proyecto implementa un sistema **RAG (Generación Aumentada por Recuperación)** que permite a los usuarios interactuar mediante lenguaje natural con las guías operativas y manuales en formato Word (`.docx`) de la organización. 

El agente extrae el contexto directamente de los documentos validados para redactar respuestas precisas, mitigando las alucinaciones del modelo y garantizando una operación **100% local, privada y sin costos de APIs externas**.

---

## 📸 Demostración de Caso de Uso

A continuación se muestra la interfaz interactiva construida con Streamlit, donde el usuario final puede resolver dudas operativas en tiempo real:

![Interfaz de Usuario del Agente de IA](./interfaz_chat.png)

*El agente recupera los fragmentos de los manuales más cercanos a la duda del usuario, los analiza y redacta un paso a paso estructurado.*

---

## 🛠️ Arquitectura y Tecnologías Utilizadas

El flujo de información se divide en dos fases críticas gobernadas por **LangChain**:

1. **Ingesta de Datos:** Conversión de documentos de texto a representaciones matemáticas abstractas (vectores) almacenados localmente.
2. **Generación de Respuestas:** Búsqueda semántica en la base de datos de vectores que sirve como contexto enriquecido para el Modelo de Lenguaje (LLM).

| Componente | Herramienta / Modelo | Características |
| :--- | :--- | :--- |
| **Orquestador** | LangChain (LCEL) | Controla el flujo de datos usando la sintaxis de tuberías moderna. |
| **Vector DB** | Chroma DB | Base de datos vectorial embebida y local, eficiente y sin costo. |
| **Embeddings** | `all-MiniLM-L6-v2` | Modelo de HuggingFace optimizado para CPU que indexa semántica. |
| **LLM Local** | `Qwen2.5-0.5B-Instruct` | Modelo conversacional ultraligero con excelente soporte en español. |
| **Interfaz Web** | Streamlit | UI interactiva nativa en Python ideal para herramientas de IA. |

---

## 🚀 Guía de Instalación y Uso Rápido

Sigue estos pasos en tu terminal para clonar y ejecutar el entorno de desarrollo local (probado y optimizado en entornos Windows):

### 1. Configurar el Entorno Virtual
```bash
# Crear el entorno aislado
python -m venv venv

# Activar en Windows (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1