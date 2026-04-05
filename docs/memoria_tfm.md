# Memoria del Trabajo de Fin de Máster

**Título:** Predicción de la Siguiente Acción Óptima en un Pipeline CRM B2B Mediante Aprendizaje Automático

**Autor:** [Nombre del Autor]

**Máster:** [Nombre del Máster]

**Universidad:** [Nombre de la Universidad]

**Tutor/a:** [Nombre del Tutor]

**Fecha:** [Mes, Año]

---

## Índice

1. [Introducción](#1-introducción)
2. [Estado del Arte](#2-estado-del-arte)
3. [Metodología](#3-metodología)
4. [Generación de Datos Sintéticos](#4-generación-de-datos-sintéticos)
5. [Ingeniería de Características](#5-ingeniería-de-características)
6. [Modelado Predictivo](#6-modelado-predictivo)
7. [Evaluación y Resultados](#7-evaluación-y-resultados)
8. [Conclusiones](#8-conclusiones)
9. [Referencias](#9-referencias)

---

## 1. Introducción

### 1.1 Contexto y Motivación

En el ámbito de las ventas B2B, los equipos comerciales dedican una parte significativa de su tiempo a decidir cuál es la siguiente acción más adecuada para cada lead en el pipeline de ventas. Esta toma de decisiones, a menudo basada en la intuición del vendedor, puede beneficiarse enormemente de técnicas de aprendizaje automático que analicen patrones históricos de éxito.

El presente trabajo aborda este problema en el contexto de **MoveUp**, una empresa ficticia de movilidad corporativa (similar a Uber for Business). El objetivo principal es construir un sistema capaz de predecir automáticamente la siguiente acción óptima que un agente de ventas debe tomar tras cada interacción con un cliente potencial.

### 1.2 Objetivos

- **Objetivo principal:** Desarrollar un modelo de clasificación multi-clase que prediga la siguiente acción de ventas a partir de transcripciones de llamadas y datos del lead.
- **Objetivos secundarios:**
  - Generar un dataset sintético realista y anotado utilizando LLMs.
  - Explorar el uso de embeddings semánticos multilingües para representar transcripciones en español.
  - Evaluar la viabilidad de un sistema de recomendación de acciones para CRM B2B.

### 1.3 Estructura del Documento

[Descripción breve de cada capítulo.]

---

## 2. Estado del Arte

### 2.1 CRM Inteligente y Automatización de Ventas

[Revisar literatura sobre Sales Intelligence, lead scoring, next-best-action systems.]

### 2.2 Procesamiento de Lenguaje Natural para Ventas

[Análisis de sentimiento en llamadas, speech analytics, modelos de conversación.]

### 2.3 Modelos de Lenguaje Grandes (LLMs) para Generación de Datos

[GPT-4, datos sintéticos, data augmentation en NLP.]

### 2.4 Embeddings Semánticos Multilingües

[SBERT, paraphrase-multilingual-MiniLM-L12-v2, representaciones de texto.]

### 2.5 Gradient Boosting para Clasificación Tabular

[XGBoost, LightGBM, comparativas con redes neuronales en datos tabulares.]

---

## 3. Metodología

### 3.1 Visión General del Pipeline

[Diagrama del pipeline: generación → feature engineering → entrenamiento → predicción.]

### 3.2 Herramientas y Tecnologías

| Componente | Herramienta |
|---|---|
| Generación de datos | OpenAI GPT-4o |
| Embeddings | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Modelo ML | XGBoost |
| Encoders | scikit-learn (StandardScaler, OneHotEncoder, LabelEncoder) |
| Orquestación | Python 3.11, PyYAML |

### 3.3 Variable Objetivo

El sistema predice una de las siguientes 7 acciones:

1. "Recontactar en X días"
2. "Enviar documentación"
3. "Agendar demo/reunión con especialista"
4. "Escalar a manager del lead"
5. "Cerrar lead - no interesado"
6. "Cerrar lead - nurturing"
7. "Esperar confirmación cliente"

---

## 4. Generación de Datos Sintéticos

### 4.1 Diseño del Dataset

[Estructura de filas por interacción, campos estáticos vs. dinámicos.]

### 4.2 Proceso de Generación con GPT-4o

[Prompt engineering, instrucciones de formato JSON, señales aprendibles en transcripciones.]

### 4.3 Control de Calidad

[Validación de categorías, reintentos, guardado parcial.]

### 4.4 Distribución del Dataset

[Estadísticas del dataset generado: leads por estado, distribución de next_step, sectores, etc.]

---

## 5. Ingeniería de Características

### 5.1 Características Numéricas

[Descripción de las 5 variables numéricas y su preprocesamiento con StandardScaler.]

### 5.2 Características Categóricas

[OneHotEncoding de 7 variables, manejo de valores desconocidos.]

### 5.3 Embeddings Semánticos

[Arquitectura del modelo, dimensionalidad, estrategia dual (transcript + contexto).]

### 5.4 Matriz Final de Características

[Concatenación: numérico (5) + categórico (variable) + embeddings (768) = N dimensiones.]

---

## 6. Modelado Predictivo

### 6.1 Selección del Algoritmo

[Justificación de XGBoost frente a alternativas (Random Forest, redes neuronales).]

### 6.2 Configuración del Modelo

[Hiperparámetros utilizados, estrategia multi:softprob.]

### 6.3 Estrategia de Validación

[Stratified train/test split 80/20, validación cruzada estratificada 5 folds.]

---

## 7. Evaluación y Resultados

### 7.1 Métricas

[Accuracy, F1 weighted, matriz de confusión por clase.]

### 7.2 Resultados de Validación Cruzada

[F1 medio ± desviación estándar en 5 folds.]

### 7.3 Resultados en Test Set

[Classification report completo, análisis por clase.]

### 7.4 Análisis de Importancia de Características

[Top 30 features, peso relativo de embeddings vs. features tabulares.]

### 7.5 Discusión

[Análisis de errores, clases confundidas, limitaciones del dataset sintético.]

---

## 8. Conclusiones

### 8.1 Conclusiones Principales

[Resumen de resultados y aportaciones del trabajo.]

### 8.2 Limitaciones

[Dataset sintético vs. real, sesgo del LLM, distribución de clases.]

### 8.3 Trabajo Futuro

- Integración con CRM real (Salesforce, HubSpot).
- Fine-tuning de modelos de lenguaje para transcripciones de ventas.
- Modelos secuenciales (LSTM, Transformer) que modelen la progresión de llamadas.
- A/B testing de las recomendaciones con agentes reales.

---

## 9. Referencias

[APA 7th edition format]

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- OpenAI. (2024). GPT-4 Technical Report.
- [Añadir referencias sobre CRM analytics, sales AI, synthetic data generation]
