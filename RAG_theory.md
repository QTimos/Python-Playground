# Introduction

Models can be trained on trillions of tokens and millions of data points, but their knowledge will always be stuck at the point where their training ends. This can lead to issues when the model is asked about something that has changed since its training. For example, consider a country’s inflation rate:

- The model can give you the latest data it was trained on (the last inflation numbers).
- The model can hallucinate an answer based on the patterns it was trained on (it will give you a completely fabricated number).

This means that if the data changes, you need to retrain the model, which is costly because training large models requires an extreme amount of processing power. This illustrates both the power and the limitations of parametric knowledge.

# Parametric Knowledge vs. Non-Parametric Knowledge

The parametric and non-parametric approaches to knowledge in computational systems are two fundamentally different approaches. You can think of them as two different library systems.

> **Parametric knowledge** is like a capable scholar who has read every book in the library, but when you ask them a question, they can only answer from memory and do not have access to the books.

> **Non-parametric knowledge** is like a searchable catalog where every book is accessible, but you can only search for information because the system has no ability to reason.

## Parametric Knowledge

Parametric knowledge is a model’s ability to hold information in its parameters—i.e., weights—by strengthening the weights that correspond to a pattern through training. This method is brilliantly powerful, but it can also be fragile. The way models are trained is non-deterministic, meaning that we cannot know which weights correspond to which pattern because all weights are shared. While training a model, as you train it on more and more data, these weights can change, making it hallucinate on some matters if the training data on those matters was small or if it was trained more on data that used those same weights (of course, all topics or data it is trained on use all the weights). That means the storage is implicit and distributed. This can show up in domain-specific data, as the representation in the training data skews heavily toward publicly available data.

Models also have a fixed number of parameters, meaning that any later training will change all the weights that were previously trained, which can cause degradation of other information.

The way models store data is another inherent limitation. Information is stored in the model’s parameters/weights, meaning that every time a model answers a query, it is interpolating the probabilities of tokens as the next token; it is not really memorizing and recalling the information.

That also means that knowledge compressed into fixed weights at a single point in time cannot be arbitrarily updated, cannot be inspected at a granular level, and cannot be guaranteed accurate.

**Key concepts discussed above:**

- `Compression artifacts`
- `Implicit storage`
- `Fixed capacity`
- `Interpolation over memorization`
- `Gradient forgetting`

## Non-Parametric Knowledge

Non-parametric knowledge refers to storing information and retrieving it at query time. The data is stored as documents—i.e., the knowledge is stored in some sort of external memory. At inference time, or in this case at query time, the system retrieves relevant information from this memory.

Think of it as an organized, instantly searchable filing cabinet that never forgets anything and can be updated by dropping in a new document. The cost of growing this external storage is orders of magnitude less than retraining a model, and it can grow indefinitely at the cost of retrieval time. Also, when you query information, you can know where that information came from.

This demonstrates key properties of non-parametric knowledge: documents are stored explicitly and can be inspected, additions are instant without training, and the source of every fact is known through its document identifier. Notice that the system returns the exact text that was stored, with no possibility of distortion or hallucination at the retrieval level. The retrieved documents provide a factual foundation that can then be processed by downstream systems.

Non-parametric knowledge has its own limitations, such as simple keyword retrieval being unable to handle vocabulary mismatch between the query and the documents.

**Key concepts discussed above:**

- `Explicit storage`
- `Unlimited capacity`
- `Instant update`
- `Provenance`
- `No compression loss`
- `Graceful handling of rare information`

# Complementary Relationship (RAG)

Parametric and non-parametric approaches have complementary strengths and weaknesses, as each excels in different aspects of the knowledge representation problem.

The combination of these two approaches is the core concept behind Retrieval-Augmented Generation (RAG). A system can use non-parametric retrieval to get relevant documents and use these documents to elevate the model’s generation and reasoning. This can help the model use the past information it holds in its parameters and the external information retrieved by the retrieval system to reason better, using the context rather than generating from all continuations.

This greatly reduces hallucination by grounding the model in the documents. It is achieved by changing the dynamics of generation: giving an explicit, external reference for factual claims, making them evidence-based. This also helps the model deal with uncertainty, because when no relevant document is retrieved, the model can express that rather than hallucinating an answer.

Another benefit is domain adaptation without retraining. You can focus the model’s answers on a certain field or on a specific proprietary project by retrieving from a curated corpus, and the model can reason when answering your queries using these grounded truths.

A major advantage is that RAG systems can cite their sources, making answers transparent because you can verify them yourself.

These kinds of systems do have limitations. One is that the whole system relies on the quality of the retrieved documents, meaning that a RAG system is only as good as its retrieval. This “garbage in, garbage out” dynamic means that retrieval quality often matters more than generation quality. Another limitation discussed above is retrieval latency: the larger the amount of documents in external storage, the more time it takes to rank and retrieve documents. The number of documents that can be used is also limited by the model’s context window. In addition, the knowledge base must be maintained and updated frequently and regularly.

Another failure mode occurs when a model confidently contradicts retrieved documents by favoring its own outdated or strongly held parametric training data, requiring strict prompt engineering and contradiction testing to fix.

## Some Use Cases for RAG

- **Enterprise Knowledge Management:** RAG enables employees to ask natural language questions and receive precise, cited answers directly from internal company documentation, drastically reducing time spent searching for information.
- **Customer Support:** RAG-powered support systems provide instant, accurate responses grounded in official content, reducing technical errors and helping human agents scale customer assistance efficiently.
- **Research and Analysis:** RAG synthesizes data across massive document collections to accelerate research workflows, providing traceable citations to original sources for reliable meta-analysis and verification.
- **Regulatory Compliance:** RAG gives compliance teams instant access to evolving regulatory language, allowing them to dynamically update their knowledge base without retraining to avoid costly legal infractions.
- **Code Assistance:** RAG coding assistants pull live documentation and internal repository examples, ensuring developers receive code that aligns with current library versions and company style guides.

**Think of a RAG system as a research assistant who has access to a library. When you ask a question, the assistant first searches the library's catalog and retrieves the most relevant books and articles. Then, with those materials spread on the desk, the assistant reads through them and composes an answer.  The quality of the assistant's response depends on two things: how good the librarian is at finding relevant materials (retrieval), and how skilled the assistant is at reading comprehension and synthesis (generation).**


# The component of a RAG system

A rag system consists of two primary components: a **retriever** that finds the relevant documents from a knowledge base, and a **generator** (typically a trained llm) that synthezies an answer using the retrieved documents as context. The retriever acts as a search engine, filtering all documents to get the most relevant documents to the query. The Generator reads these relevant documents as context for the query and reasons a response that adresses the users request. This is if you have an index of documents pre-created, if not there you will need a **chunker** component, that will chunk the given information being it text or code etc. Then **indexer** component that will save these documents with the position of each document to ensure siting sources will work correctly, here you can optimize what to save to disk with this indexer to heighten the preformance or retrieval. Though this can be a bit of a balancing act.

> **chunking** first, documents are split into smaller chunks. A 50-page technical report cannot be fed to a language model in its entirety, so we divide it into passages of a few hundred tokens each. The chunking strategy significantly affects retrieval quality: too small and you lose context, too large and you dilute relevant information with noise. Consider a chunk that contains only half a sentence: the retriever might match it to a query based on keyword overlap, but the generator would receive incomplete, potentially misleading information. Conversely, a chunk spanning multiple pages might contain one relevant paragraph buried among irrelevant content, which makes it harder for the generator to identify what matters..A common heuristic is 256 to 512 tokens per chunk with 10 to 20 percent overlap between adjacent chunks.
> **indexing** indexing is where you store the documents in an optimized index for fast search. For sparce retrieval, `inverted indexes` map each document in the vocabulary to the documents containing it. At query time we look up query terms to get documents that contain them and then score these documetens using a scorign/ranking fromulas like TF-IDF or BM25. For dense retrieval, `vector indexes` organize embeddings to enabel efficient nearest-neighbor search.
> **retrieving** when a query arrives, the retriever encodes it the same way as the documents—BM25 weights for sparse retrieval, or the same neural encoder for dense retrieval—so both live in the same representation space. It then searches the index for the top-k most similar documents and returns them with similarity scores, which guide ranking, filtering, and confidence. The value of k (typically 3–10) balances coverage against noise: too few documents may miss relevant information; too many dilute the context. The best k depends on the query—simple factual questions need fewer passages, complex analytical queries may need more.
> **generating** the generator takes the retrieved documents and the original query and produces a natural-language response, usually via an LLM. It must understand intent, find relevant passages, resolve conflicts between sources, and synthesize a coherent answer—more like a lawyer writing a brief from case files than a simple extractor, including acknowledging gaps instead of fabricating. Output quality depends on three interacting factors: retrieved context quality, prompt quality, and the model’s intrinsic capability; improving any one generally improves the final answer.
    - Context Integration
        The simplest integration method is prompt concatenation: insert retrieved documents before the query, with no architectural changes to the model. A typical prompt is Context: [docs] Question: [query] Answer based on the context above:. This leverages in-context learning—the model already knows how to read passages and answer questions, so you provide raw material rather than teach a new skill. The context window enables many documents in one forward pass, but larger contexts increase latency and attention cost, and models often attend less reliably to the middle of long contexts.
    - Attention Over Retrieved Context
        Retrieved documents are just additional tokens; self-attention lets generated tokens attend to all context and synthesize across documents, implicitly weighing conflicts and authority. Attention patterns can reveal which passages influenced the response and support attribution/citation tools, though they are not a formal faithfulness guarantee. Grounding also depends on training: instruction-tuned or QA-fine-tuned models use retrieved context more reliably than base next-token models.
    - Prompt Engineering for RAG
        Prompt wording strongly affects whether the model grounds, hallucinates, or expresses uncertainty. Key considerations: explicitly instruct the model to answer only from context and say “I don’t know” when information is missing; manage document ordering because of position bias; ask the model to cite document numbers; require uncertainty expression; and specify output format. Effective templates reduce hallucination, improve traceability, and make outputs more consistent.


#### Retrieval timing

- **Single-Shot Retrieval**: Single-shot retrieval retrieves documents once before generation begins. The query goes to the retriever, documents come back, and generation uses that fixed context. It works well when the query clearly specifies what is needed, documents likely contain complete answers, low latency is required, and no multi-hop reasoning is needed. Most production RAG systems use it because it is fast, simple, and easy to debug, you can check whether errors came from retrieval or generation. Its limitation is complex queries, a single retrieval may miss context or fail to adapt, such as a query requiring both policy changes and study outcomes.
- **Iterative Retrieval**: Iterative retrieval uses multiple retrieval steps. It starts with an initial retrieval, may generate partial output to reveal gaps, then formulates more specific sub-queries and retrieves again before synthesizing a final answer. This helps with multi-part queries (e.g., comparing two recessions) and multi-hop reasoning (e.g., first find the most popular Python web framework, then find who founded it). The trade-off is latency and complexity, each retrieval adds round-trip time and requires logic to decompose queries and decide when enough information has been gathered. This is closely related to agent-based systems where the model orchestrates tool calls, with the retriever as one tool.
- **Token-Level Retrieval**: Token-level retrieval sits at the opposite extreme from single-shot. Some architectures retrieve new documents throughout generation. DeepMind’s RETRO interleaves retrieval by splitting input into fixed-size chunks (typically 64 tokens), retrieving nearest-neighbor passages for each chunk, encoding them separately, and integrating them via cross-attention. This keeps context relevant as generation shifts topics or sub-questions. The cost is substantial computational overhead: each retrieval adds latency, and thousands of retrievals per generation are impractical for interactive use. It remains mainly a research technique, though its core insight—that retrieval should stay dynamically relevant during generation—has influenced designs like kNN-LM and newer memory-augmented approaches.


# Variety of RAG architectures

## Retrieve-then-Read (Sequential RAG)

The standard architecture we've been describing is sometimes called "retrieve-then-read"; retrieve first, then read (generate). Documents pass through a clean interface; the retriever's output becomes the generator's input. This separation creates a clear contract between components; the retriever promises to return relevant documents, and the generator promises to synthesize them into an answer. This separation enables independent optimization; you can upgrade the retriever without touching generation, or swap the LLM without changing retrieval. It also simplifies debugging—inspect retrieved documents to tell whether the failure is retrieval (wrong documents) or generation (right documents, wrong answer). The sequential architecture is the right starting point for most projects; build it first, measure where quality falls short, and add complexity only if it cannot meet your requirements.

## Fusion-in-Decoder (FiD)

FiD encodes each retrieved document-query pair independently, then fuses their representations in the decoder via cross-attention. This keeps encoding cost constant per document and scales better than concatenation, whose attention cost grows quadratically with sequence length. The original FiD paper fused up to 100 documents and achieved state-of-the-art open-domain QA. The trade-off is that FiD requires an encoder-decoder architecture (like T5), so it is less applicable to pure decoder models like LLaMA or GPT.

 ## Retrieval-Augmented Language Models (REALM and RAG)

REALM and the original RAG paper proposed end-to-end trainable retrieval by backpropagating through the retrieval step. Retrieval is treated as a latent variable; the answer probability is a weighted mixture over retrieved documents, P(a|q) = Σ P(a|q,di) · P(di|q), where weights come from retriever relevance scores. Because both terms are differentiable, the retriever learns to retrieve documents that help the generator. Training is expensive because updating document embeddings requires periodic re-indexing (asynchronous index updates in the original RAG). The resulting retrievers often outperform generic ones on knowledge-intensive benchmarks, at the cost of infrastructure complexity.

## RETRO; Chunked Cross-Attention

RETRO builds retrieval into the transformer forward pass; it splits input into chunks, retrieves nearest-neighbor passages for each chunk, encodes them separately, and integrates them via chunked cross-attention interleaved with self-attention. This keeps retrieved context dynamically relevant as generation proceeds. It scales retrieval to trillions of tokens while keeping the main model small—a 7B RETRO model can match a 25B model without retrieval. The implication is that scaling the retrieval database, not just parameters, may be the right strategy for knowledge-intensive tasks.

## Self-RAG; Retrieval as a Learned Decision

Self-RAG trains models to decide when to retrieve, what to retrieve, and how to use retrieved content, rather than retrieving for every query. It generates control tokens—[Retrieve], [IsREL], [IsSUP], [IsUSE]—to signal retrieval need, passage relevance, support for claims, and whether to use a passage. Trained with supervised and reinforcement learning, it filters irrelevant documents and relies on parametric knowledge when confident. This reduces latency and cost for simple queries and yields more calibrated uncertainty, closer to how a human expert draws on memory versus lookup.

## Hybrid Retrieval

Many production systems combine sparse and dense retrieval; sparse excels at exact term matches, dense at semantic similarity. The most common fusion is Reciprocal Rank Fusion (RRF), which scores each document as Σ 1/(k + rank_r(d)) over retrieval methods r, with k typically 60. RRF trusts rank over raw score because BM25 and cosine similarities are on different scales; a document ranking high in both lists gets a strong convergent signal. RRF is simple, requires no training, and consistently outperforms either retriever alone. More sophisticated learned fusion exists, but RRF is an excellent default across domains.


# Key RAG parameters

> **k**: Number of documents to retrieve (typically 3–10). Balances coverage against noise; higher k increases retrieval and generation latency. Tune from 1–20 by measuring answer quality.
> **score_threshold**: Minimum similarity score for relevance. Depends on retrieval method and corpus; calibrate on representative queries, as thresholds may not transfer.
> **max_tokens**: Maximum tokens to generate. Longer limits improve completeness but raise latency and cost; set based on typical answer length.
> **chunk_size**: Chunk size in tokens. Smaller chunks (128–256) suit precise factual retrieval; larger chunks (512–1024) preserve context for extended passages.
> **chunk_overlap**: Tokens shared by adjacent chunks. 10–20% overlap reduces splitting relevant content across boundaries at the cost of some index redundancy.

(-- LOAD_IMAGE = "https://assets.mbrenndoerfer.com/_optimized/notebooks/2_rag_architecture_files/rag-architecture-data-flow-dark-1200w.webp" --)


# Lexical representation of text

## TF-IDF





