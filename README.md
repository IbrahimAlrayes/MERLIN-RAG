# Adaptive RAG for Emerging & Low-Frequency Entities

This repository explores building a Retrieval-Augmented Generation (RAG) system that adapts to **fresh, trending, and low-frequency entities**.  
The project focuses on integrating **web search tools, Google Trends, Wikipedia/Wikidata, and news feeds** to discover and evaluate popular or emerging people across countries.

---
## 🚀 Setup Instructions

To run this project, you’ll first need to set up the **supporting services** (SearXNG and Reader).

[👉 Go to Setup Guide](src/SETUP.md)
---

## 📌 Goals
- Improve **adaptability of RAG** systems by dynamically incorporating new or low-frequency entities.  
- Use **Google Trends** (for recently popular people) and **Wikipedia/Wikidata** (for long-tail/low-frequency entities).  
- Evaluate system performance on both **freshly popular** and **rare entities**.  
- Explore **multilingual and multimodal signals** for better disambiguation.

---

## 📚 Reading Log
| Paper / Resource | Notes / Insights |
|------------------|------------------|
| [Guide to Multimodal RAG for Images and Text](https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117) | ___ |
| [LangChain Semi-structured & Multimodal RAG Notebook](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.com) | ___ |
| [Integrating Text, Images and Audio in RAG](https://www.reddit.com/r/LangChain/comments/1enbqew/multimodal_rag_explainer_3_paths_to_integrating/) | ___ |


## 🔧 Tools
| Tool | What It Does |
|------|--------------|
| [SearXNG](https://github.com/searxng/searxng) | A Lets you query multiple search engines at once (Google, Bing, Wikipedia, etc.) and returns aggregated results. |
| [Perplexica](https://github.com/ItzCrazyKns/Perplexica) | An **AI-powered search engine** inspired by Perplexity.ai. It performs **web search + summarization**, then provides citations. Can be integrated as a higher-level **question-answering tool**. |
| [CoexistAI](https://github.com/SPThole/CoexistAI) | A **multi-agent framework** that supports dynamic **tool use and collaboration** between agents. Good for orchestrating **MLLM core + retrieval + tools**. |
| [GPTR-MCP](https://github.com/assafelovic/gptr-mcp) | Provides a **Model Context Protocol (MCP) implementation**. Helps connect LLMs to **external tools and APIs** in a standardized way. Helps with deep research implementation. |
| [Reader](https://github.com/intergalacticalvariable/reader) | A **universal AI-powered reader** that can parse and summarize **PDFs, websites, and documents**. |
| [Local Deep Researcher](https://github.com/langchain-ai/local-deep-researcher) | A **LangChain-powered research framework** that runs entirely **locally**. It performs **multi-step reasoning, document retrieval, and synthesis** to explore topics deeply without relying on external APIs. |



---

## 🧪 Progress Tracker
- [ ✔ ] Set up repo structure 
- [ ✔ ] Build an intial architecture 
- [ ] implement a goggle trends search functionality **Google Trends API** (`pytrends`)  
- [ ] implement a Wikidata lookup functionality **Wikidata SPARQL queries**   
- [ ] Design Multimodal retriever   
- [ ] Implement **Internet search** tool  
- [ ] Implement **Deep research** tool  
- [ ] Implement **Image search** tool  
- [ ] Implement **Hallucination checker** 
- [ ] First experiments  

---


# Initial Architecture

![alt text](assets/architecture.png)

# 🔎 Retrieval

## Embedding Models

### 1. Direct Multimodal Embeddings
Use a single **multimodal embedding model** to embed both text and images into the same vector space.

- [**SigLIP**](https://github.com/google-research/siglip)  
  Successor to CLIP from Google Research. Stronger scaling, improved multilingual performance, and better retrieval accuracy.

- [**EVA-CLIP**](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)  
  A large-scale CLIP variant from BAAI. Achieves state-of-the-art performance on many retrieval benchmarks.

- [**Multilingual CLIP**](https://github.com/FreddeFrallan/Multilingual-CLIP)  
  Extends CLIP to support multiple languages by aligning multilingual text embeddings with the CLIP image space.

---

### 2. Image → Text Summarization → Text Embeddings
Use a **multimodal LLM** to summarize or caption images, then pass both the **summaries and text data** through a strong text embedding model.  
This approach normalizes everything into text.
- Multimodal captioning/summary models:  
  - [**BLIP / BLIP-2**](https://github.com/salesforce/LAVIS) – Image captioning and vision-language models.  
  - [**LLaVA**](https://github.com/haotian-liu/LLaVA) – A multimodal GPT-style model that can answer questions and describe images.  
  - Or we can use Panega

- Text embedding models:  
  - [**E5**](https://huggingface.co/intfloat/multilingual-e5-base) – Strong multilingual embedding model, optimized for retrieval.  
  - [**Instructor-XL**](https://huggingface.co/hkunlp/instructor-xl) – Task-aware embeddings that allow instructions with queries.  
  - [**BGE Large**](https://huggingface.co/BAAI/bge-large-en) – A state-of-the-art English embedding model, highly effective for retrieval tasks.

## 3. Rerankers with Multimodal Embeddings
Use a two-stage pipeline:  
1. A **bi-encoder** (e.g., CLIP/SigLIP) retrieves top-k candidates quickly.  
2. A **reranker** (cross-encoder style) jointly encodes the query and candidate (text, image, or both) to rescore and reorder them.  
This improves accuracy, especially for **ambiguous or low-frequency entities**.

- [**mDPR + Cross-encoder**](https://github.com/facebookresearch/DPR) (concept adapted for multimodal)  
  Bi-encoder retrieval followed by cross-encoder reranking. Can be extended with CLIP embeddings for images.

- [**ColPali**](https://huggingface.co/vidore/colpali)  
  A **multimodal reranker** for documents that can handle **image-heavy PDFs** by combining text + layout + images.  
  Shows how rerankers improve grounding in multimodal settings.

- [**bge-reranker (text)**](https://huggingface.co/BAAI/bge-reranker-large)  
  Text-only reranker. While not multimodal, the same architecture idea can be adapted by feeding **image captions + text evidence** for final scoring.


- Retrieve candidates with **SigLIP** or **EVA-CLIP**.  
- For each candidate, pass `(query_text, candidate_text, candidate_image)` into a reranker.  
- Reranker outputs a relevance score → reorder top-k → feed top-n into the generator.

