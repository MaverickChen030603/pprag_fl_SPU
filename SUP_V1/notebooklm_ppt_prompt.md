# NotebookLM Prompt Pack: SUP_V1 Workflow PPT

Use the following prompts to generate a 5-slide English presentation. Keep the overall style minimal and technology-oriented, with a white background and subtle purple gradient accents. Use concise, rigorous language. Avoid dense paragraphs. Prefer clean diagrams, thin lines, simple icons, and clear workflow arrows.

## Slide 1 Prompt: Motivation and System Overview

Create Slide 1 titled **"SUP_V1: Hypernetwork-Guided Selective Upload for FedE4RAG"**.

Design a minimal technology-style opening slide with a white background and a soft purple gradient flowing from the top-right corner to the bottom-left. Add a clean two-stage system diagram:

**Upstream Federated Learning**
Local clients train retriever embedding updates under privacy constraints.

**Downstream RAG**
The trained retriever supports indexing, retrieval, generation, and evaluation.

Show the core motivation in three short points:

- Full-model upload is communication-expensive.
- Client updates are not equally important across model blocks.
- A hypernetwork can predict important blocks for selective upload.

Use a horizontal workflow visual:

`Local Data -> FedRAG Training -> Hypernetwork Selection -> Sparse Upload -> Global Retriever -> RAG Evaluation`

Keep the layout spacious and professional. Use purple only as an accent color for arrows, highlights, and section labels.

## Slide 2 Prompt: Upstream FL Baseline and Communication Bottleneck

Create Slide 2 titled **"Upstream Baseline: FedRAG with Full Parameter Upload"**.

Use a white background with a light purple gradient band at the bottom. Draw a client-server federated learning workflow:

1. The server sends the current retriever model to selected clients.
2. Each client performs local FedRAG training.
3. Each client uploads the full model update.
4. The server aggregates all received updates.

Include the FedRAG local objective as a compact formula:

`L_client = L_contrastive + lambda * L_distill`

Explain the bottleneck with three concise labels:

- Large embedding backbone.
- Repeated full-parameter transmission.
- Increasing cost with more clients and rounds.

Add a visual contrast: show thick purple upload arrows from clients to server labeled **"Full Delta Upload"**. Add a small warning tag: **"High Communication Cost"**.

Keep all text in English, concise, and suitable for an academic technical presentation.

## Slide 3 Prompt: Hypernetwork-Based Importance Estimation

Create Slide 3 titled **"Hypernetwork as a Parameter Importance Estimator"**.

Use a minimal white canvas with purple gradient nodes. Draw a compact architecture diagram:

`Client ID Embedding + Block ID Embedding + Previous Delta Statistics -> Hypernetwork -> Block Importance Scores`

Show model blocks for a BERT/BGE-style retriever:

- `embeddings`
- `encoder.layer.0 ... encoder.layer.11`
- `pooler`

For each block, display a score bar or heatmap cell. Highlight top-k blocks in purple. Use a small example:

`Top-k selected blocks: encoder.layer.9, encoder.layer.10, encoder.layer.11`

Explain the learning signal:

- Clients compute lightweight block statistics after local training.
- The server trains the hypernetwork to predict normalized delta importance.
- Future upload masks are selected from predicted importance scores.

Add the training target in one concise line:

`target(block) ~= normalized ||Delta_block||_2`

Maintain a clean technical style, with thin connectors and no visual clutter.

## Slide 4 Prompt: Selective Upload Workflow in SUP_V1

Create Slide 4 titled **"SUP_V1 Workflow: Sparse Delta Upload"**.

Use a white background and a subtle purple gradient around the central workflow. Draw the full upstream workflow as a numbered pipeline:

1. Server predicts an upload mask using the hypernetwork.
2. Server sends the model and selected block mask to each client.
3. Client trains the full local retriever with the original FedRAG loss.
4. Client computes full block-level delta statistics.
5. Client uploads only selected block deltas plus lightweight statistics.
6. Server performs sparse aggregation and updates the hypernetwork.

Show the client upload package as a clean card:

```text
{
  selected_delta_blocks,
  upload_mask,
  block_statistics,
  num_samples
}
```

Add a side comparison:

**Before:** full model delta upload.

**After:** top-k block delta upload.

Use a compact communication statement:

`Payload ratio ~= uploaded block parameters / total trainable parameters`

Highlight that training remains full local training, while communication becomes selective. Use purple arrows only for uploaded blocks and grey dashed arrows for non-uploaded blocks.

## Slide 5 Prompt: Downstream RAG Evaluation and Expected Impact

Create Slide 5 titled **"Downstream RAG: Evaluating Retriever Quality Under Lower Communication"**.

Use a clean end-to-end diagram connecting upstream output to downstream RAG:

`Selective FL Retriever -> Corpus Indexing -> Query Embedding -> Retrieval -> LLM Response -> Evaluation`

Show two metric groups:

**Retrieval Metrics**
Hit@k, Recall@k, MRR, MAP, NDCG.

**Generation Metrics**
ROUGE, CHRF, METEOR, WER, CER, Perplexity.

State the expected impact in three concise points:

- Lower communication cost through block-level sparse upload.
- Comparable or improved retrieval quality by prioritizing important updates.
- Better scalability for multi-client private RAG training.

Add a final visual message:

`Goal: preserve RAG effectiveness while reducing upstream FL communication overhead.`

Use a white background with a purple gradient footer. Keep the final slide polished, minimal, and suitable for a research progress report.

