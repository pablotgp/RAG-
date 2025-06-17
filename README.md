# Pipeline RAG Avanzado para Procesamiento de PDF y Evaluación de LLM

Este repositorio contiene un Jupyter Notebook (`RAG_clean (1) (1).ipynb`) que demuestra un pipeline completo de extremo a extremo para procesar documentos PDF destinados a aplicaciones de Generación Aumentada por Recuperación (RAG). Incluye limpieza avanzada de texto, detección de elementos estructurales, fragmentación semántica (chunking), creación de un almacén de vectores usando embeddings de Azure OpenAI, una estrategia de recuperación híbrida (FAISS + BM25) con reordenamiento (reranking) y un marco de evaluación para comparar diferentes Modelos de Lenguaje Grandes (LLM) en varias tareas RAG.

## Descripción General

El notebook realiza los siguientes pasos clave:

1.  **Procesamiento de PDF:** Extrae texto de un PDF (ejemplo: `Biologia.pdf`), realiza una limpieza robusta y detecta elementos estructurales como portadas, índices, secciones de bibliografía, fórmulas matemáticas e imágenes.
2.  **Fragmentación Semántica (Chunking):** Divide el texto limpio en fragmentos significativos basados en oraciones, utilizando NLTK con mecanismos de respaldo.
3.  **Enriquecimiento con Metadatos:** Anota cada fragmento con metadatos (p. ej., presencia de matemáticas heurísticas).
4.  **Creación del Almacén de Vectores:** Genera embeddings para los fragmentos de texto utilizando el modelo `text-embedding-3-small` de Azure OpenAI y construye un índice FAISS para una búsqueda eficiente por similitud vectorial.
5.  **Recuperación Híbrida:** Implementa un recuperador avanzado que combina la búsqueda densa de vectores (FAISS) y la búsqueda dispersa por palabras clave (BM25) con ponderación dinámica según el tipo de consulta.
6.  **Reordenamiento (Reranking):** Utiliza un modelo CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`) para reordenar los documentos candidatos recuperados por la búsqueda híbrida y mejorar la relevancia final.
7.  **Evaluación de LLM:** Establece un marco para evaluar múltiples LLM (Azure GPT-4o, Google Gemini 1.5 Pro, HuggingFace Hub Mixtral y Llama 3) en tareas basadas en RAG (Resumen, Generación de Preguntas, Preguntas y Respuestas) utilizando el contexto recuperado.
8.  **Recolección de Resultados:** Almacena los resultados de la evaluación, incluyendo tiempos de ejecución y respuestas de los LLM, en un archivo CSV.

## Características Clave

*   **Limpieza Robusta de PDF:** Funciones personalizadas (`clean_pdf_text_robust`) para manejar artefactos comunes de extracción de PDF (guiones, paginación, URLs, encabezados/pies de página específicos).
*   **Análisis Estructural de PDF:** Heurísticas para detectar y potencialmente excluir portadas, índices y secciones de bibliografía. Detecta secciones matemáticas (Teoremas, Definiciones, etc.) y regiones de imágenes.
*   **Detección de Fórmulas:** Identifica fórmulas tipo LaTeX y expresiones matemáticas heurísticas.
*   **Chunking Consciente de Oraciones:** Usa `nltk.sent_tokenize` para dividir texto respetando los límites de las oraciones, con alternativas para texto complejo.
*   **Embeddings de Azure OpenAI:** Aprovecha `text-embedding-3-small` para crear embeddings de texto de alta calidad.
*   **Indexación FAISS:** Crea un índice vectorial eficiente para búsquedas rápidas por similitud.
*   **Búsqueda Híbrida (FAISS + BM25):** Combina similitud semántica (FAISS) con relevancia de palabras clave (BM25) para una recuperación más robusta.
*   **Ponderación Dinámica:** Ajusta la contribución de las puntuaciones de FAISS y BM25 según el análisis de la consulta (específica vs. conceptual).
*   **Reordenamiento con CrossEncoder:** Mejora el ranking final de relevancia de los documentos recuperados usando un potente modelo reranker.
*   **Evaluación Multi-LLM:** Marco para comparar el rendimiento entre diferentes proveedores de LLM (Azure, Google, Hugging Face Hub).
*   **Tareas RAG Estandarizadas:** Incluye prompts y datos para evaluar Resumen, Generación de Preguntas y Preguntas y Respuestas (Q&A) basadas en el contexto recuperado.

## Configuración e Instalación

1.  **Entorno:**
    *   Se recomienda Python 3.10 (según los metadatos del notebook).
    *   Crea un entorno virtual (opcional pero recomendado).

2.  **Dependencias:** Instala las bibliotecas requeridas usando pip. Ejecuta **primero el bloque de Cell 1** en el notebook, que instala la mayoría de las dependencias. Luego, **ejecuta Cell 11** que instala versiones específicas de numpy/pandas/scikit-learn necesarias para compatibilidad:
    ```bash
    # (Contenido de Cell 1 - Instalación principal)
    %pip install pypdf2 python-dotenv langchain google-cloud-aiplatform rank_bm25 nltk faiss-cpu numpy sentence_transformers torch language_tool_python datasets bitsandbytes transformers peft pymupdf pytesseract pillow PyMuPDF transformers torch tiktoken python-multipart openai

    # (Contenido de Cell 11 - Versiones específicas)
    %pip install --upgrade --force-reinstall numpy==1.23.5 pandas==1.5.3 scikit-learn==1.1.3

    # (Contenido de Cell 2/3 - Desinstalar/Instalar PyMuPDF)
    %pip uninstall -y fitz
    %pip install PyMuPDF
    ```
    *Nota: Es mejor ejecutar estas celdas directamente en el notebook.*

3.  **Datos NLTK:** Descarga los datos necesarios de NLTK (el notebook lo hace en las celdas 10 y 12):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    # nltk.download('stopwords') # Descomentar si se usa el tokenizador nltk_tokenizer con stopwords
    ```

4.  **Variables de Entorno:** Crea un archivo `.env` en la raíz del proyecto (o especifica la ruta correcta en el notebook, como en la Celda 18) y añade tus claves de API y endpoints:
    ```dotenv
    # Azure OpenAI (Tanto para LLM principal como para Embeddings)
    AZURE_OPENAI_ENDPOINT="https://TU_ENDPOINT.openai.azure.com/"
    AZURE_OPENAI_API_KEY="TU_CLAVE_API_AZURE"
    AZURE_OPENAI_DEPLOYMENT_NAME="TU_DEPLOYMENT_GPT4_TURBO" # Nombre del deployment para el LLM principal (ej. gpt-4)
    AZURE_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-small" # Nombre del deployment para embeddings

    # Google AI (Gemini)
    GOOGLE_API_KEY="TU_CLAVE_API_GOOGLE"

    # Hugging Face Hub (para Mixtral/Llama3)
    HUGGINGFACEHUB_API_TOKEN="TU_TOKEN_HUGGINGFACE_HUB"
    ```
    *   *Nota Importante:* Asegúrate de que los nombres de tus deployments en Azure (`AZURE_OPENAI_DEPLOYMENT_NAME` y `AZURE_EMBEDDING_DEPLOYMENT_NAME`) coinciden con los especificados y usados en el código (Celdas 19 y 21).

## Flujo de Uso

1.  **Preparar PDF:** Coloca el archivo PDF que deseas procesar (p. ej., `Biologia.pdf`) en la ubicación especificada en la Celda 12 (`pdf_path`) o actualiza la ruta en esa celda.
2.  **Ejecutar Celdas Secuencialmente:** Ejecuta las celdas del notebook en orden.
    *   **Celdas 1-11:** Instalan bibliotecas y definen todas las funciones de procesamiento, chunking y detección. La Celda 11 instala versiones específicas compatibles.
    *   **Celda 12:** Realiza el procesamiento principal del PDF especificado (`Biologia.pdf`), limpia el texto, lo fragmenta y genera metadatos.
    *   **Celda 16:** Inicializa el cliente simple de Azure OpenAI (utilizado en la celda siguiente).
    *   **Celda 17:** Genera los embeddings para los fragmentos y crea el índice FAISS. Guarda `educacion.index`, `educacion_texts.pkl` y `educacion_metas.pkl`.
    *   **Celdas 18-22:** Definen la función de recuperación híbrida (`my_hybrid_rerank_retriever`) que carga el índice FAISS, textos, metadatos, inicializa BM25 y carga el modelo CrossEncoder. Se prueba la lógica del recuperador en la Celda 22.
    *   **Celda 19:** Carga los clientes para los diferentes proveedores de LLM usando las claves del archivo `.env`. Verifica la salida para confirmar la carga exitosa.
    *   **Celdas 20-23:** Preparan los prompts y los datos de evaluación (para Biología en este caso, pero adaptables para otras materias descomentando/modificando).
    *   **Celda 24:** Define la función RAG universal (`run_rag_based_task`).
    *   **Celda 25:** Ejecuta el bucle de evaluación RAG. Itera sobre las tareas de evaluación (Resumen, Generación de Preguntas, Q&A), llama a `retriever_function` para obtener el contexto, y luego usa los LLM seleccionados para generar respuestas basadas en el contexto y los prompts específicos de la tarea. Se recopilan los resultados (contexto, respuesta, tiempos, errores). *Nota: Esta celda incluye pausas después de las llamadas a Azure para gestionar los límites de tasa.*
    *   **Celda 26:** Calcula métricas básicas sobre los resultados recopilados (opcional).
3.  **Revisar Salidas:**
    *   Verifica la salida de la consola para los logs durante el procesamiento, indexación, recuperación y generación.
    *   Inspecciona los archivos generados:
        *   `educacion.index`: El almacén de vectores FAISS.
        *   `educacion_texts.pkl`: Lista serializada (pickle) de los fragmentos de texto.
        *   `educacion_metas.pkl`: Lista serializada (pickle) de los diccionarios de metadatos correspondientes a los fragmentos.
        *   `llm_rag_evaluation_results.csv`: Archivo CSV que contiene los resultados de las evaluaciones de los LLM en las tareas RAG.
