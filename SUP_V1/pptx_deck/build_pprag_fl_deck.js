const pptxgen = require('pptxgenjs');
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require('./pptxgenjs_helpers/layout');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Codex';
pptx.subject = 'Privacy-preserving RAG based on federated learning';
pptx.title = 'PPRAG-FL: Hypernetwork-Guided Selective Parameter Upload for Federated RAG';
pptx.company = 'FedE4RAG';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Helvetica Neue',
  bodyFontFace: 'Helvetica Neue',
  lang: 'en-US',
};
pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'CUSTOM_WIDE';
pptx.margin = 0;
pptx.layout = 'LAYOUT_WIDE';

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;
const C = {
  ink: '141126',
  muted: '5E5870',
  lightText: '7A748C',
  purple: '6D36D8',
  purple2: '9A7BFF',
  purple3: 'C7B7FF',
  lavender: 'F4F0FF',
  lavender2: 'EDE6FF',
  blue: '2D74FF',
  cyan: '4DD1FF',
  green: '19A974',
  red: 'E24A62',
  white: 'FFFFFF',
  line: 'D8D0EF',
  grid: 'E9E1FF',
  chip: 'F8F5FF',
};

function addBg(slide, section = '', idx = 1) {
  slide.background = { color: C.white };
  // In-bounds edge accents create a white + purple gradient feel without colliding with content.
  slide.addShape(pptx.ShapeType.rect, { x: 13.08, y: 0, w: 0.25, h: 7.5, line: { color: C.lavender, transparency: 100 }, fill: { color: C.lavender, transparency: 18 } });
  slide.addShape(pptx.ShapeType.rect, { x: 13.02, y: 0, w: 0.04, h: 7.5, line: { color: C.purple3, transparency: 100 }, fill: { color: C.purple3, transparency: 38 } });
  slide.addShape(pptx.ShapeType.line, { x: 0.55, y: 6.95, w: 12.15, h: 0, line: { color: C.grid, transparency: 10, width: 1 } });
  slide.addText(section, { x: 0.62, y: 6.99, w: 6.4, h: 0.25, fontFace: 'Helvetica Neue', fontSize: 7.5, color: C.lightText, margin: 0, breakLine: false, fit: 'shrink' });
  slide.addText(String(idx).padStart(2, '0'), { x: 12.26, y: 6.94, w: 0.55, h: 0.24, fontFace: 'Helvetica Neue', fontSize: 8, bold: true, align: 'right', color: C.purple, margin: 0, fit: 'shrink' });
}

function title(slide, text, eyebrow = '') {
  if (eyebrow) {
    slide.addText(eyebrow.toUpperCase(), { x: 0.72, y: 0.45, w: 3.0, h: 0.22, fontSize: 8.5, bold: true, color: C.purple, charSpace: 1.1, margin: 0, fit: 'shrink' });
  }
  slide.addText(text, { x: 0.68, y: 0.72, w: 8.45, h: 0.58, fontFace: 'Helvetica Neue', fontSize: 25, bold: true, color: C.ink, margin: 0, breakLine: false, fit: 'shrink' });
  slide.addShape(pptx.ShapeType.line, { x: 0.72, y: 1.38, w: 1.05, h: 0, line: { color: C.purple, width: 3, beginArrowType: 'none', endArrowType: 'none' } });
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const fontSize = opts.fontSize || 15;
  const color = opts.color || C.muted;
  const bulletIndent = opts.bulletIndent || 14;
  const hanging = opts.hanging || 4;
  slide.addText(items.join('\n'), {
    x, y, w, h,
    fontFace: 'Helvetica Neue',
    fontSize,
    color,
    breakLine: false,
    valign: 'top',
    margin: 0.08,
    fit: 'shrink',
    paraSpaceAfterPt: opts.spaceAfter || 8,
    breakLine: false,
    bullet: { type: 'ul', indent: bulletIndent, hanging },
  });
}

function callout(slide, text, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, line: { color: opts.line || C.line, width: 1 }, fill: { color: opts.fill || C.chip, transparency: opts.transparency || 0 } });
  slide.addText(text, { x: x + 0.18, y: y + 0.16, w: w - 0.36, h: h - 0.28, fontFace: 'Helvetica Neue', fontSize: opts.fontSize || 13, bold: !!opts.bold, color: opts.color || C.ink, valign: 'mid', align: opts.align || 'center', margin: 0.02, fit: 'shrink' });
}

function card(slide, heading, body, x, y, w, h, accent = C.purple) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, line: { color: C.line, width: 1 }, fill: { color: C.white, transparency: 0 } });
  slide.addShape(pptx.ShapeType.roundRect, { x: x + 0.16, y: y + 0.16, w: 0.18, h: 0.18, rectRadius: 0.04, line: { color: accent, transparency: 100 }, fill: { color: accent } });
  slide.addText(heading, { x: x + 0.48, y: y + 0.12, w: w - 0.62, h: 0.28, fontSize: 12.2, bold: true, color: C.ink, margin: 0, fit: 'shrink' });
  slide.addText(body, { x: x + 0.18, y: y + 0.56, w: w - 0.36, h: h - 0.68, fontSize: 10.5, color: C.muted, valign: 'top', margin: 0.03, breakLine: false, fit: 'shrink' });
}

function flow(slide, labels, x, y, totalW, h, colors = []) {
  const gap = 0.24;
  const w = (totalW - gap * (labels.length - 1)) / labels.length;
  labels.forEach((label, i) => {
    const xx = x + i * (w + gap);
    callout(slide, label, xx, y, w, h, { fill: i === labels.length - 1 ? C.purple : C.white, line: i === labels.length - 1 ? C.purple : C.line, color: i === labels.length - 1 ? C.white : C.ink, fontSize: 11.5, bold: true });
    if (i < labels.length - 1) {
      slide.addShape(pptx.ShapeType.line, { x: xx + w + 0.04, y: y + h / 2, w: gap - 0.08, h: 0, line: { color: colors[i] || C.purple2, width: 1.8, beginArrowType: 'none', endArrowType: 'triangle' } });
    }
  });
}

function addEquationBox(slide, lines, x, y, w, h) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, line: { color: C.line, width: 1 }, fill: { color: 'FBFAFF' } });
  slide.addText(lines.join('\n'), { x: x + 0.24, y: y + 0.24, w: w - 0.48, h: h - 0.45, fontFace: 'Menlo', fontSize: 12.5, color: C.ink, margin: 0, breakLine: false, fit: 'shrink', valign: 'mid' });
}

function metricBar(slide, label, val, x, y, w, color) {
  slide.addText(label, { x, y, w: 2.0, h: 0.24, fontSize: 9.5, bold: true, color: C.ink, margin: 0, fit: 'shrink' });
  if (val >= 0.999) {
    slide.addShape(pptx.ShapeType.rect, { x: x + 2.05, y: y + 0.05, w, h: 0.12, line: { color, transparency: 100 }, fill: { color } });
  } else {
    slide.addShape(pptx.ShapeType.rect, { x: x + 2.05, y: y + 0.05, w, h: 0.12, line: { color: C.lavender, transparency: 100 }, fill: { color: C.lavender } });
    slide.addShape(pptx.ShapeType.rect, { x: x + 2.05, y: y + 0.07, w: w * val, h: 0.08, line: { color, transparency: 100 }, fill: { color } });
  }
}

function addSimpleNetwork(slide, x, y) {
  const nodes = [
    [x, y + 0.70, 'Client A'], [x, y + 1.72, 'Client B'], [x, y + 2.74, 'Client C'],
    [x + 3.25, y + 1.72, 'Server'], [x + 6.35, y + 0.95, 'Global Retriever'], [x + 6.35, y + 2.45, 'RAG Evaluation']
  ];
  nodes.forEach(([xx, yy, t], i) => callout(slide, t, xx, yy, i === 3 ? 1.35 : 1.55, 0.52, { fill: i === 3 ? C.purple : C.white, line: i === 3 ? C.purple : C.line, color: i === 3 ? C.white : C.ink, fontSize: 9.5, bold: true }));
  [[x + 1.55, y + 0.96, 1.68, 0.76], [x + 1.55, y + 1.98, 1.68, 0], [x + 1.55, y + 3.00, 1.68, -0.76], [x + 4.60, y + 1.98, 1.70, -0.76], [x + 4.60, y + 1.98, 1.70, 0.76]].forEach(([xx, yy, ww, hh]) => {
    slide.addShape(pptx.ShapeType.line, { x: xx, y: yy, w: ww, h: hh, line: { color: C.purple2, width: 1.4, endArrowType: 'triangle' } });
  });
}

const slides = [];
function newSlide(section, idx, slideTitle, eyebrow) {
  const s = pptx.addSlide();
  addBg(s, section, idx);
  if (slideTitle) title(s, slideTitle, eyebrow || section);
  slides.push(s);
  return s;
}

// 1
{
  const s = pptx.addSlide(); slides.push(s);
  s.background = { color: C.white };
  s.addShape(pptx.ShapeType.rect, { x: 13.02, y: 0, w: 0.31, h: 7.5, line: { color: C.lavender2, transparency: 100 }, fill: { color: C.lavender2, transparency: 14 } });
  s.addShape(pptx.ShapeType.rect, { x: 12.88, y: 0, w: 0.09, h: 7.5, line: { color: C.purple3, transparency: 100 }, fill: { color: C.purple3, transparency: 30 } });
  s.addText('PPRAG-FL', { x: 0.82, y: 0.72, w: 2.4, h: 0.3, fontSize: 13, bold: true, color: C.purple, charSpace: 1.6, margin: 0, fit: 'shrink' });
  s.addText('Privacy-Preserving RAG Based on Federated Learning', { x: 0.78, y: 1.38, w: 8.6, h: 0.58, fontSize: 28, bold: true, color: C.ink, margin: 0, breakLine: false, fit: 'shrink' });
  s.addText('A hypernetwork-guided selective parameter upload mechanism for communication-efficient FedE4RAG', { x: 0.82, y: 2.18, w: 8.9, h: 0.7, fontSize: 16, color: C.muted, margin: 0, breakLine: false, fit: 'shrink' });
  flow(s, ['Private RAG', 'Federated Retriever', 'Selective Upload', 'Lower Payload'], 0.86, 3.66, 8.35, 0.58);
  callout(s, '15-minute research proposal', 0.86, 5.48, 2.6, 0.46, { fill: C.chip, line: C.line, color: C.purple, fontSize: 10, bold: true });
  s.addText('01', { x: 12.22, y: 6.92, w: 0.58, h: 0.24, fontSize: 8, bold: true, align: 'right', color: C.purple, margin: 0, fit: 'shrink' });
}

// 2
{
  const s = newSlide('Outline', 2, 'Talk Roadmap', 'Outline');
  const items = ['Background', 'Related Work', 'Current Challenges', 'Research Objectives', 'The Method', 'Experiment Design', 'References', 'Appendix'];
  items.forEach((t, i) => {
    const row = Math.floor(i / 4), col = i % 4;
    const x = 0.86 + col * 3.02, y = 2.05 + row * 1.42;
    callout(s, `${String(i + 1).padStart(2, '0')}  ${t}`, x, y, 2.52, 0.78, { fill: i < 5 ? C.white : C.chip, line: C.line, fontSize: 12, align: 'left', bold: true });
  });
  s.addText('The deck follows the source proposal, but compresses the wording for an English 15-minute presentation.', { x: 0.9, y: 5.35, w: 8.8, h: 0.38, fontSize: 12, color: C.muted, margin: 0, fit: 'shrink' });
}

// 3
{
  const s = newSlide('Background', 3, 'RAG Improves Grounding, But Private Deployment Is Hard');
  card(s, 'Why RAG matters', 'Retrieval-augmented generation injects external evidence into LLM generation, improving factuality, traceability, and domain adaptation.', 0.8, 1.75, 3.75, 2.1, C.purple);
  card(s, 'Where it is needed', 'Finance, law, and healthcare often require vertical QA over high-value institutional knowledge bases.', 4.85, 1.75, 3.75, 2.1, C.blue);
  card(s, 'What blocks adoption', 'The most useful corpora are sensitive. Centralizing them can violate governance rules and internal compliance policies.', 8.9, 1.75, 3.75, 2.1, C.red);
  flow(s, ['User Query', 'Retriever', 'Private Corpus', 'LLM Answer'], 1.35, 4.65, 10.4, 0.62);
}

// 4
{
  const s = newSlide('Background', 4, 'Privacy Risk Exists Beyond Raw Text');
  addBullets(s, [
    'Raw documents cannot be freely pooled across institutions.',
    'Vector embeddings may still leak semantic content or membership signals.',
    'A shared vector database can become a governance bottleneck.',
    'The central question is how to improve retrieval without exposing local corpora.'
  ], 0.88, 1.78, 5.6, 3.2, { fontSize: 15 });
  card(s, 'Privacy surface', 'Raw records\nEmbeddings\nModel updates\nRetrieval traces', 7.0, 1.65, 2.3, 3.05, C.red);
  card(s, 'Desired property', 'Local data stays local; only controlled, protected learning signals are exchanged.', 9.65, 2.0, 2.55, 2.35, C.purple);
  s.addShape(pptx.ShapeType.line, { x: 9.32, y: 3.12, w: 0.3, h: 0, line: { color: C.purple2, width: 2, endArrowType: 'triangle' } });
}

// 5
{
  const s = newSlide('Background', 5, 'Federated Learning Is a Natural Training Paradigm');
  addSimpleNetwork(s, 1.0, 1.45);
  addBullets(s, [
    'Clients train retriever embeddings on local query-reference pairs.',
    'The server aggregates model updates instead of collecting private data.',
    'Cross-client knowledge can help under data scarcity and Non-IID distributions.'
  ], 1.0, 5.25, 9.7, 0.95, { fontSize: 13.5, spaceAfter: 4 });
}

// 6
{
  const s = newSlide('Related Work', 6, 'Existing Privacy-Oriented RAG Directions');
  card(s, 'DPRAG', 'Adds differential privacy noise at inference time. It reduces leakage risk, but privacy budget can limit generation length and accuracy.', 0.82, 1.72, 3.0, 2.1, C.purple);
  card(s, 'GraphRAG', 'Improves global knowledge organization and multi-document reasoning via knowledge graphs, but graph construction is costly.', 4.0, 1.72, 3.0, 2.1, C.blue);
  card(s, 'C-FedRAG', 'Uses trusted execution environments for hardware-level isolation, but faces hardware and memory constraints.', 7.18, 1.72, 3.0, 2.1, C.green);
  card(s, 'Gap', 'These approaches do not fully solve communication-efficient collaborative retriever training over private institutional corpora.', 10.36, 1.72, 2.45, 2.1, C.red);
  callout(s, 'Our focus: upstream federated retriever training, not only inference-side protection.', 1.5, 5.08, 9.95, 0.65, { fill: C.chip, line: C.line, fontSize: 13, bold: true });
}

// 7
{
  const s = newSlide('Related Work', 7, 'FedE4RAG Is the Baseline Framework');
  flow(s, ['FL', 'Knowledge Distillation', 'Homomorphic Encryption', 'Private RAG Retriever'], 1.0, 1.9, 10.9, 0.72);
  addBullets(s, [
    'FedE4RAG combines federated learning, knowledge distillation, and homomorphic encryption.',
    'Local retrievers learn from private query-reference data while aligning with a server-side teacher signal.',
    'It is a strong starting point for privacy-preserving vertical RAG.'
  ], 1.0, 3.35, 6.0, 2.0, { fontSize: 14.2 });
  card(s, 'Limitation', 'The default communication pattern still resembles conventional FL: clients frequently upload full embedding model parameters or full gradient updates.', 7.6, 3.16, 4.45, 1.8, C.red);
}

// 8
{
  const s = newSlide('Current Challenges', 8, 'Full Upload Becomes the Bottleneck');
  addBullets(s, [
    'Embedding models such as BERT/BGE-style retrievers contain many trainable parameters.',
    'Communication grows with model size, client count, and training rounds.',
    'Homomorphic encryption strengthens protection, but encrypted payloads and aggregation are more expensive.',
    'Many parameter blocks may contribute little in a given round, making full upload redundant.'
  ], 0.86, 1.75, 6.1, 3.85, { fontSize: 14.4 });
  metricBar(s, 'Full upload', 1.0, 7.35, 2.0, 3.6, C.red);
  metricBar(s, 'Selective upload', 0.25, 7.35, 2.78, 3.6, C.purple);
  metricBar(s, 'With HE', 1.0, 7.35, 3.56, 3.6, C.red);
  metricBar(s, 'Selected HE', 0.25, 7.35, 4.34, 3.6, C.purple);
  callout(s, 'Goal: cut payload without damaging downstream RAG quality.', 7.35, 5.12, 4.15, 0.62, { fill: C.purple, line: C.purple, color: C.white, fontSize: 13, bold: true });
}

// 9
{
  const s = newSlide('Current Challenges', 9, 'Problem Statement');
  addEquationBox(s, ['Full update:', 'Delta_i^t = theta_i^t - theta^t', '', 'Selective update:', 'Delta_{i,U}^t = { Delta_{i,b}^t | b in U_i^t }'], 0.95, 1.75, 5.1, 2.7);
  card(s, 'Key challenge', 'Identify and upload the most important parameter-block updates while preserving privacy and downstream retrieval performance.', 6.65, 1.9, 5.0, 1.42, C.purple);
  addBullets(s, [
    'Importance differs by client, round, and model block.',
    'Random sparsity can miss task-relevant semantic layers.',
    'Fixed compression rules may ignore client heterogeneity.'
  ], 6.9, 3.88, 4.55, 1.6, { fontSize: 13.5 });
}

// 10
{
  const s = newSlide('Research Objectives', 10, 'Research Objectives and Contributions');
  card(s, '1. Dynamic importance', 'Learn client- and block-aware scores from client embeddings, block embeddings, and update statistics.', 0.86, 1.75, 3.7, 2.2, C.purple);
  card(s, '2. Selective upload', 'Replace full updates with parameter-block top-K upload while keeping local FedRAG training objectives unchanged.', 4.82, 1.75, 3.7, 2.2, C.blue);
  card(s, '3. End-to-end evaluation', 'Connect upstream communication compression to downstream RAG retrieval and generation metrics.', 8.78, 1.75, 3.7, 2.2, C.green);
  callout(s, 'Reframing pFedLA: from a personalization generator to a communication controller.', 1.45, 5.12, 10.35, 0.64, { fill: C.chip, line: C.line, fontSize: 13, bold: true });
}

// 11
{
  const s = newSlide('The Method', 11, 'PPRAG-FL: Overall Workflow');
  const labels = ['Server sends model + mask', 'Client trains full local objective', 'Client uploads selected deltas + stats', 'Sparse aggregation', 'RAG evaluation'];
  flow(s, labels, 0.72, 1.82, 11.92, 0.72);
  addBullets(s, [
    'Warm-up rounds collect initial block statistics with full updates.',
    'After warm-up, the hypernetwork predicts a top-K upload set per client.',
    'The server updates only blocks that were uploaded by at least one selected client.'
  ], 1.05, 3.55, 6.4, 2.25, { fontSize: 14 });
  card(s, 'Design principle', 'Change the communication stage, not the local retriever learning objective.', 8.15, 3.65, 3.85, 1.48, C.purple);
}

// 12
{
  const s = newSlide('The Method', 12, 'Local Training Objective Is Preserved');
  addEquationBox(s, ['L_i = L_contrastive + lambda L_distill', '', 'Contrastive loss:', 'align query and reference embeddings', '', 'Distillation loss:', 'align local logits with server teacher logits'], 0.96, 1.7, 5.55, 3.5);
  addBullets(s, [
    'Each client owns private data D_i with query-reference pairs (q, r).',
    'Local training still optimizes retrieval matching and teacher-guided global knowledge alignment.',
    'Selective upload is applied after local training, so the retrieval learning signal remains intact.'
  ], 7.0, 1.88, 4.9, 2.6, { fontSize: 14 });
  callout(s, 'Privacy boundary: raw local corpora never leave the client.', 7.28, 4.95, 4.25, 0.62, { fill: C.chip, line: C.line, color: C.purple, fontSize: 13, bold: true });
}

// 13
{
  const s = newSlide('The Method', 13, 'Parameter-Block Selection Avoids Fragile Element-Wise Sparsity');
  const blocks = ['embeddings', 'layer 0', 'layer 1', '...', 'layer 11', 'pooler'];
  blocks.forEach((b, i) => callout(s, b, 0.88 + i * 1.88, 2.05, 1.48, 0.58, { fill: i === 5 ? C.purple : C.white, line: i === 5 ? C.purple : C.line, color: i === 5 ? C.white : C.ink, fontSize: 10.5, bold: true }));
  addBullets(s, [
    'Blocks match the Transformer retriever structure and are easier to index and analyze.',
    'Block-level upload reduces metadata overhead compared with unstructured parameter sparsity.',
    'The method can prioritize high-level semantic layers or client-specific blocks when useful.'
  ], 1.0, 3.65, 6.6, 2.15, { fontSize: 14 });
  card(s, 'Block universe', 'B = {embeddings, encoder.layer.0, ..., encoder.layer.11, pooler}', 8.15, 3.78, 3.75, 1.28, C.purple);
}

// 14
{
  const s = newSlide('The Method', 14, 'Hypernetwork as an Importance Estimator');
  addEquationBox(s, ['Input:', 'z_{i,b} = [e_i; e_b; s_{i,b}^{t-1}]', '', 'Score:', 'alpha_{i,b}^t = H_phi(z_{i,b})', '', 'Target:', 'y_{i,b}^t = ||Delta_{i,b}^t||_2 / max_b ||Delta_{i,b}^t||_2'], 0.86, 1.58, 6.0, 4.05);
  card(s, 'Client embedding e_i', 'Represents client identity or participation profile.', 7.35, 1.7, 4.0, 0.9, C.blue);
  card(s, 'Block embedding e_b', 'Represents model-block position and type.', 7.35, 2.85, 4.0, 0.9, C.purple);
  card(s, 'Update statistics s', 'Uses lightweight norms, mean magnitude, and max absolute update from history.', 7.35, 4.0, 4.0, 0.95, C.green);
  callout(s, 'Training target: normalized block update norm; loss: mean-squared prediction error.', 7.35, 5.32, 4.0, 0.55, { fill: C.chip, line: C.line, fontSize: 10.8, bold: true });
}

// 15
{
  const s = newSlide('The Method', 15, 'Selective Upload and Sparse Aggregation');
  addEquationBox(s, ['Upload set:', 'U_i^t = TopK_b(alpha_{i,b}^t)', '', 'Sparse aggregation:', 'theta_b^{t+1} = theta_b^t + sum_{i in S_b^t}', '  |D_i| / sum_{j in S_b^t}|D_j| * Delta_{i,b}^t'], 0.85, 1.62, 6.2, 3.8);
  addBullets(s, [
    'Only mask-selected block deltas are transmitted.',
    'Lightweight statistics for all blocks can still be reported to update the hypernetwork.',
    'If no client uploads a block in a round, the server keeps that block unchanged.'
  ], 7.55, 1.88, 4.55, 2.2, { fontSize: 13.7 });
  callout(s, 'Communication changes from full model update to important block update.', 7.55, 4.86, 4.1, 0.68, { fill: C.purple, line: C.purple, color: C.white, fontSize: 12.3, bold: true });
}

// 16
{
  const s = newSlide('The Method', 16, 'Privacy and Downstream RAG Integration');
  flow(s, ['Federated Retriever', 'Corpus Indexing', 'Query Embedding', 'Retrieval', 'LLM Generation', 'Evaluation'], 0.72, 1.82, 11.9, 0.67);
  addBullets(s, [
    'The method keeps FedE4RAG privacy assumptions: raw data remains local.',
    'Selective upload narrows the parameter exposure surface.',
    'If stronger protection is required, CKKS-style homomorphic encryption can be applied only to selected blocks.',
    'Downstream RAG metrics test whether communication savings preserve real retrieval and answer quality.'
  ], 0.96, 3.38, 7.0, 2.35, { fontSize: 13.7 });
  card(s, 'Expected effect', 'Less encrypted payload and lower aggregation cost when HE is enabled.', 8.45, 3.55, 3.5, 1.25, C.purple);
}

// 17
{
  const s = newSlide('Experiment Design', 17, 'Research Questions and Baselines');
  card(s, 'RQ1', 'Can PPRAG-FL significantly reduce upstream communication payload?', 0.82, 1.65, 3.6, 1.1, C.purple);
  card(s, 'RQ2', 'Does downstream retrieval remain stable under compression?', 4.86, 1.65, 3.6, 1.1, C.blue);
  card(s, 'RQ3', 'How does K trade off payload reduction and retrieval quality?', 8.9, 1.65, 3.6, 1.1, C.green);
  addBullets(s, [
    'FedE4RAG Full Upload',
    'Random Selective Upload',
    'Static Top-Layer Upload',
    'Delta-Norm Selection',
    'PPRAG-FL Hypernetwork Selection'
  ], 1.05, 3.55, 4.5, 2.2, { fontSize: 14.2 });
  callout(s, 'All baselines share the same data split, client count, training rounds, and local hyperparameters.', 6.2, 3.85, 5.2, 0.92, { fill: C.chip, line: C.line, fontSize: 12.2, bold: true });
}

// 18
{
  const s = newSlide('Experiment Design', 18, 'Datasets, Settings, Metrics, and Ablations');
  card(s, 'Data', 'Upstream: FedE/train_data to FedE/select_data.json.\nDownstream: RAGTest/data, including data_50.json and data_100.json.', 0.82, 1.62, 3.7, 2.2, C.purple);
  card(s, 'Default settings', '5 clients, 25 rounds, local epoch 1, batch size 8, learning rate 1e-5.\nHypernetwork: dim 64, hidden 128, lr 5e-3, warm-up 2, top-K 3.', 4.78, 1.62, 3.7, 2.2, C.blue);
  card(s, 'Metrics', 'Communication: payload ratio and uploaded parameters.\nRetrieval: Hit@k, Recall@k, MRR, MAP, NDCG.\nGeneration: ROUGE, CHRF, METEOR, WER, CER, PPL.', 8.74, 1.62, 3.7, 2.2, C.green);
  addBullets(s, [
    'Ablate upload budget K = 1, 3, 5, 7.',
    'Ablate warm-up rounds = 0, 1, 2, 5.',
    'Compare hypernetwork selection with random, static high-layer, and delta-norm heuristics.',
    'Optionally compare CKKS-enabled and non-encrypted communication cost.'
  ], 1.05, 4.62, 10.5, 1.25, { fontSize: 12.8, spaceAfter: 3 });
}

// 19
{
  const s = newSlide('References', 19, 'References and Methods Discussed');
  addBullets(s, [
    'FedE4RAG: federated learning, knowledge distillation, and homomorphic encryption for private RAG retriever training.',
    'pFedLA: personalized federated learning with hypernetwork-generated layer aggregation weights.',
    'DPRAG: differential privacy at the inference/output side of RAG.',
    'GraphRAG: knowledge-graph-enhanced retrieval and cross-document reasoning.',
    'C-FedRAG, FLERAG, FeB4RAG: related federated or trusted-execution RAG settings.',
    'CKKS homomorphic encryption: optional encrypted update transmission for stronger privacy.'
  ], 0.95, 1.7, 8.1, 4.1, { fontSize: 13.2, spaceAfter: 6 });
  callout(s, 'Reference details should be replaced with formal bibliographic entries before final submission if required by the venue.', 9.5, 2.16, 2.65, 1.85, { fill: C.chip, line: C.line, fontSize: 11.3, bold: true });
}

// 20
{
  const s = newSlide('Appendix', 20, 'Appendix: Implementation Hooks in the Current Repository');
  card(s, 'Upstream FL', 'FedE/main.py creates the federated task and initializes flgo.algorithm.fedrag.', 0.82, 1.62, 3.7, 1.45, C.purple);
  card(s, 'Local benchmark', 'FedE/flgo/benchmark/fedrag_classification/core.py defines task partitioning, training data, and client loss calculation.', 4.78, 1.62, 3.7, 1.45, C.blue);
  card(s, 'Server/client protocol', 'fedrag.py handles model broadcast, local training, client replies, and server aggregation.', 8.74, 1.62, 3.7, 1.45, C.green);
  card(s, 'Downstream RAGTest', 'RAGTest/index.py builds vector indexes; RAGTest/retriever.py implements retrieval; RAGTest/eval computes retrieval and generation metrics.', 2.0, 3.8, 4.35, 1.55, C.purple);
  card(s, 'Where PPRAG-FL fits', 'Add block masks, payload logging, hypernetwork scoring, sparse aggregation, and selected-block HE hooks around the communication phase.', 7.0, 3.8, 4.35, 1.55, C.red);
}

// Diagnostics required by the slides skill. Background shape containment is intentional.
pptx._slides.forEach((slide) => {
  warnIfSlideHasOverlaps(slide, pptx, { muteContainment: true, ignoreDecorativeShapes: true, ignoreLines: true });
  warnIfSlideElementsOutOfBounds(slide, pptx);
});

pptx.writeFile({ fileName: 'PPRAG_FL_research_proposal.pptx' });
