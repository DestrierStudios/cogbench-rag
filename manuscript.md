# Do Machines Remember Like We Do? CogBench-RAG: A Cognitive Benchmark for Retrieval-Augmented Generation Systems

Nikhil Saxena<sup>1</sup>

<sup>1</sup> Northeastern University, Boston, MA, USA

Corresponding author: saxena.ni@northeastern.edu

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are increasingly framed as "memory" for large language models, yet no framework exists for evaluating whether their retrieval behavior aligns with human memory phenomena. We introduce CogBench-RAG, a benchmark that tests RAG systems against five memory principles: encoding specificity, the fan effect, proactive and retroactive interference, serial position effects, and retrieval-induced forgetting. We evaluate sparse lexical retrieval (BM25) and dense semantic retrieval using a Cognitive Alignment Score. Both architectures exhibit encoding specificity, monotonically decreasing retrieval accuracy with increasing associative fan, and interference with stronger retroactive than proactive effects. Neither shows serial position effects or retrieval-induced forgetting. The architectures diverge on encoding specificity: BM25 exhibits near-absolute context dependence (mismatch MRR = 0.141), while dense retrieval partially bridges contextual gaps (mismatch MRR = 0.410). These findings suggest that competition-based memory phenomena emerge from similarity-based retrieval, while temporal and practice-dependent effects do not. The null serial position result suggests that retrieval systems do not contribute to "lost in the middle" effects observed in language models. CogBench-RAG provides a reusable framework for evaluating the cognitive properties of retrieval architectures.

Keywords: retrieval-augmented generation, cognitive science, human memory, benchmark, encoding specificity, information retrieval

---

## 1. Introduction

The rapid adoption of Retrieval-Augmented Generation (RAG) has established external retrieval as a primary mechanism for providing large language models with access to knowledge beyond their training data (Lewis et al., 2020). RAG systems are routinely described using memory metaphors: vector databases serve as "long-term memory," retrieval functions as "recall," and the interplay between parametric and non-parametric knowledge mirrors distinctions drawn between semantic and episodic memory systems (Tulving, 1972; Gutiérrez et al., 2024). These analogies have practical consequences. Recent systems explicitly draw on neuroscience, with HippoRAG modeling the hippocampal indexing theory of Teyler and Discenna (1986) through knowledge graphs and Personalized PageRank (Gutiérrez et al., 2024), EM-LLM implementing surprise-based episodic segmentation from event cognition research (Fountas et al., 2025; Zacks et al., 2007), and ARM incorporating Ebbinghaus-style memory decay and consolidation (Bursa, 2026; Ebbinghaus, 1885).

Despite this convergence between retrieval system design and cognitive science, an open question persists: do RAG systems actually behave like human memory? The existing literature uses cognitive science as design inspiration for building systems, but no work has systematically used cognitive science as an evaluation framework for understanding them. If retrieval systems exhibit human-like memory biases, practitioners need to anticipate and mitigate them. If they do not, the memory metaphors guiding system design may be misleading.

Human memory research offers over a century of rigorously characterized phenomena, from Ebbinghaus's (1885) foundational work on forgetting curves to contemporary models of retrieval-induced forgetting (Anderson et al., 1994). These phenomena are robust, well-quantified, and replicated across diverse materials and populations (Baddeley et al., 2015; Kahana, 2012). They provide precise behavioral predictions that can be operationalized as benchmark tasks. Yet no retrieval benchmark, including BEIR (Thakur et al., 2021), MTEB (Muennighoff et al., 2022), RAGAS (Es et al., 2024), or ARES (Saad-Falcon et al., 2024), tests whether retrieval systems exhibit these patterns.

We introduce CogBench-RAG, a benchmark suite comprising five modules, each grounded in a foundational human memory phenomenon:

1. Encoding specificity (Tulving & Thomson, 1973): retrieval success depends on the match between encoding and retrieval contexts.
2. The fan effect (Anderson, 1974): retrieval accuracy decreases as the number of facts associated with a concept increases.
3. Proactive and retroactive interference (Underwood, 1957; Müller & Pilzecker, 1900): prior learning impairs new retrieval (proactive) and new learning impairs retrieval of old information (retroactive).
4. Serial position effects (Murdock, 1962; Ebbinghaus, 1885): items at the beginning and end of a sequence are recalled better than middle items.
5. Retrieval-induced forgetting (Anderson et al., 1994): practicing retrieval of some items suppresses access to related unpracticed items.

For each phenomenon, we design controlled retrieval tasks, define expected human-like behavioral patterns, and introduce the Cognitive Alignment Score (CAS), a normalized metric quantifying the degree to which a system's behavior matches the human pattern.

We evaluate BM25 (sparse, lexical; Robertson & Zaragoza, 2009) and dense retrieval using sentence-transformers (semantic, embedding-based; Reimers & Gurevych, 2019) and find a pattern of selective alignment. Both systems exhibit human-like encoding specificity, fan effects, and interference, but neither shows serial position effects or retrieval-induced forgetting. The two architectures also diverge in how they exhibit shared phenomena: BM25 shows near-absolute encoding specificity where contextual mismatch is equivalent to retrieval failure, while dense retrieval partially bridges contextual gaps through semantic similarity.

Our contributions are:

- CogBench-RAG, an open-source benchmark suite that operationalizes five human memory phenomena as retrieval evaluation tasks.
- The Cognitive Alignment Score (CAS), a normalized metric for quantifying human-likeness of retrieval behavior.
- A systematic mapping between RAG system behavior and established human memory phenomena, suggesting selective alignment that varies by both phenomenon and architecture.
- Evidence that standard retrieval architectures do not contribute to "lost in the middle" effects (Liu et al., 2024) and that encoding specificity varies systematically between lexical and semantic retrieval.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG was introduced by Lewis et al. (2020), who combined a pre-trained seq2seq model with a dense passage retriever to ground language generation in retrieved evidence. Since then, the approach has diversified. Dense passage retrieval using learned dual-encoder embeddings (Karpukhin et al., 2020) coexists with sparse lexical methods such as BM25 (Robertson & Zaragoza, 2009), hybrid sparse-dense approaches (Ma et al., 2021), and graph-augmented systems (Edge et al., 2024). Iterative retrieval methods such as FLARE (Jiang et al., 2023) trigger retrieval based on generation-time uncertainty, while Self-RAG (Asai et al., 2024) learns when to retrieve and how to critique retrieved passages.

Recent work has moved toward cognitively-inspired architectures. HippoRAG (Gutiérrez et al., 2024) mimics the hippocampal memory indexing theory (Teyler & Discenna, 1986) using knowledge graphs and Personalized PageRank. EM-LLM (Fountas et al., 2025) applies surprise-based event segmentation from cognitive models of episodic memory (Zacks et al., 2007; Radvansky & Zacks, 2014). Bursa (2026) introduces Adaptive RAG Memory (ARM), implementing selective remembrance and decay inspired by memory consolidation (McGaugh, 2000). These systems reflect growing interest in cognitive alignment but evaluate success through task performance metrics rather than behavioral correspondence with human memory.

### 2.2 Retrieval Benchmarks

BEIR (Thakur et al., 2021) established the standard for zero-shot retrieval evaluation across 18 heterogeneous datasets. MTEB (Muennighoff et al., 2022) extended this to embedding evaluation across classification, clustering, reranking, and semantic similarity. RAG-specific evaluations have also expanded: RAGAS (Es et al., 2024) assesses faithfulness and context precision; ARES (Saad-Falcon et al., 2024) provides automated RAG evaluation; and RGB (Chen et al., 2024) benchmarks RAG robustness. All existing benchmarks evaluate task performance rather than behavioral characterization. CogBench-RAG addresses a different question: whether a system's retrieval behavior exhibits specific patterns predicted by cognitive theory, independent of its performance on any particular downstream task.

### 2.3 Cognitive Evaluation of AI Systems

A growing body of work tests whether AI systems replicate human cognitive patterns. Hagendorff et al. (2023) found that GPT-3 exhibits human-like intuitive behaviors on the Cognitive Reflection Test. Koo et al. (2024) benchmarked cognitive biases in LLM evaluation outputs, finding 40% of comparisons exhibited biases. Cheung et al. (2025) observed amplified omission bias in LLM moral decision-making. Binz and Schulz (2023) evaluated GPT-3 against human decision heuristics. Suri et al. (2024) investigated anchoring effects in LLMs. Kim et al. (2025) found that reasoning capabilities did not protect against clinical cognitive biases.

This work has focused on language model generation behavior. CogBench-RAG applies the same approach to the retrieval stage, testing whether systems that supply information to LLMs exhibit their own cognitive patterns.

### 2.4 Memory and Retrieval in Cognitive Science

The five phenomena we test represent well-established findings in cognitive science. Encoding specificity (Tulving & Thomson, 1973) has been observed across verbal, spatial, and environmental contexts (Godden & Baddeley, 1975; Eich, 1980; Marian & Neisser, 2000). The fan effect (Anderson, 1974) is a core prediction of ACT-R (Anderson et al., 2004; Anderson & Reder, 1999) and has been documented across propositional, spatial, and visual materials (Radvansky et al., 1993). Interference theory has been central to understanding forgetting since Müller and Pilzecker (1900), with the retroactive-exceeds-proactive asymmetry consistently replicated (Wixted, 2004; Kliegl & Bäuml, 2021; Postman & Underwood, 1973; McGeoch, 1932). Serial position effects (Murdock, 1962) arise from differential rehearsal (Rundus, 1971) and recency of activation in working memory (Glanzer & Cunitz, 1966; Atkinson & Shiffrin, 1968). Retrieval-induced forgetting (Anderson et al., 1994; Anderson, 2003) has been confirmed via meta-analysis across over 200 experiments (Murayama et al., 2014).

---

## 3. Methods

### 3.1 Overview

CogBench-RAG comprises five benchmark modules, each operationalizing a human memory phenomenon as a controlled retrieval task. Each module generates a corpus and query set organized into experimental conditions. Retrieval performance is measured using Mean Reciprocal Rank (MRR) and Recall@1, and a Cognitive Alignment Score (CAS) quantifies pattern correspondence with the expected human-like pattern (Section 3.7). All experiments use controlled synthetic corpora with fixed random seeds (numpy seed = 42, PYTHONHASHSEED = 0) following established benchmarking methodology (Germain et al., 2020; Luecken & Theis, 2019).

### 3.2 Systems Under Test

We evaluate two architecturally distinct retrieval systems representing the two dominant paradigms in modern information retrieval:

BM25 (Robertson & Zaragoza, 2009): A sparse lexical retrieval method based on the probabilistic relevance framework, scoring documents using term frequency, inverse document frequency, and document length normalization. We use the Okapi BM25 implementation with parameters k1 = 1.5 and b = 0.75 (Manning et al., 2008). BM25 operates on exact token matches and is invariant to semantic similarity between non-identical terms.

Dense Retrieval (MiniLM): A dense semantic retrieval method using the all-MiniLM-L6-v2 sentence-transformer model (Wang et al., 2020; Reimers & Gurevych, 2019) to encode documents and queries into 384-dimensional embeddings. Retrieval uses cosine similarity via FAISS (Johnson et al., 2019) exact inner product search on L2-normalized vectors. Dense retrieval captures semantic similarity through learned contextual embeddings and can match queries to documents sharing few lexical tokens.

Both systems use top-k = 10 retrieval. The architectural contrast allows distinguishing phenomena arising from general retrieval competition from those dependent on the specific nature of the similarity computation.

### 3.3 Module 1: Encoding Specificity

Tulving and Thomson (1973) showed that memory retrieval is most effective when cues present at retrieval match those present during encoding. This principle has been supported across environmental contexts (Godden & Baddeley, 1975), emotional states (Eich, 1980), and linguistic contexts (Marian & Neisser, 2000).

We construct eight topic pairs, each expressing the same core fact in two different domain framings. For example, the effects of rising ocean temperatures on marine ecosystems are expressed in both a marine biology framing (discussing zooxanthellae, thermal stress, and coral bleaching) and an economics framing (discussing seafood industry losses, tourism revenue, and insurance markets). This yields 16 documents. For each document, three query conditions are tested: context match (same domain vocabulary), context mismatch (other domain's vocabulary for the same underlying information), and unrelated (different topic entirely). This yields 48 queries. The critical comparison is context match vs. context mismatch.

### 3.4 Module 2: Fan Effect

Anderson (1974) showed that retrieval accuracy decreases as the number of facts associated with a concept increases. This fan of associations creates competition during retrieval, a core mechanism in the ACT-R cognitive architecture (Anderson et al., 2004; Anderson & Reder, 1999).

We generate 80 entities, each associated with a controlled number of documents: 20 entities at each fan size of 1, 2, 5, and 10. Each document describes a unique activity in a distinct domain, using lexically diverse templates to ensure discriminability. This yields 360 documents. Queries reference an entity and its associated domain keyword. At fan size 1, one document matches; at fan size 10, ten documents sharing the entity name compete for retrieval.

### 3.5 Module 3: Interference

Interference theory has been central to understanding forgetting since Müller and Pilzecker (1900). Underwood (1957) established proactive interference as an equally important mechanism. The retroactive-exceeds-proactive asymmetry is a consistent finding (Wixted, 2004; Kliegl & Bäuml, 2021), and interference is modulated by similarity (McGeoch, 1932; Osgood, 1949).

Six entities each have two documents representing sequential phases. Phase 1 documents describe initial activities; Phase 2 documents describe a strategic pivot. Both phases share entity name and general vocabulary but differ in specifics. Four control entities appear in only one phase. Queries use generic shared attributes that both phase documents could plausibly answer, analogous to the A-B, A-C paired-associate paradigm (Barnes & Underwood, 1959). This yields 32 queries.

### 3.6 Modules 4 and 5: Serial Position and Retrieval-Induced Forgetting

The serial position effect (Murdock, 1962; Glanzer & Cunitz, 1966; Kahana, 2012) produces a U-shaped recall curve with primacy attributed to rehearsal (Rundus, 1971) and recency to working memory activation (Atkinson & Shiffrin, 1968). We construct two 10-document sequences indexed in order, with positions classified as primacy (0-1), middle (2-7), and recency (8-9). This module is additionally motivated by the "lost in the middle" finding (Liu et al., 2024).

Anderson et al. (1994) showed that practicing retrieval of some category members suppresses access to unpracticed members, attributed to inhibitory control (Anderson, 2003; Murayama et al., 2014). We construct four document categories with practiced and unpracticed items, plus a baseline category with no practiced items. After three retrieval practice rounds, we test unpracticed item retrieval.

### 3.7 Cognitive Alignment Score (CAS)

For each module, we compute a CAS normalized to [0, 1], where 1.0 indicates correspondence with the human behavioral pattern and 0.0 indicates no alignment. Computation is module-specific: encoding specificity CAS weights the match-mismatch MRR gap (60%) and correct condition ordering (40%); fan effect CAS uses the Spearman correlation between fan size and MRR; interference CAS combines PI and RI effect magnitudes relative to controls; serial position CAS detects primacy and recency advantages; and RIF CAS measures the baseline-test difference.

---

## 4. Results

### 4.1 Encoding Specificity

Both systems exhibited encoding specificity, but with qualitatively different profiles (Figure 3). BM25 showed near-absolute context dependence: context-matched queries achieved perfect retrieval (MRR = 1.000), while context-mismatched queries performed at the level of unrelated queries (mismatch MRR = 0.141, unrelated MRR = 0.140). For BM25, querying about ocean warming effects using marine biology vocabulary versus economics vocabulary proved no more effective than querying about an entirely unrelated topic.

Dense retrieval also showed encoding specificity but with partial bridging of the contextual gap. Context-matched queries achieved perfect retrieval (MRR = 1.000), while context-mismatched queries performed above the unrelated baseline (mismatch MRR = 0.410, unrelated MRR = 0.101). Dense retrieval captures semantic overlap between domain-mismatched descriptions of the same phenomenon that BM25 cannot access.

Both systems achieved CAS = 1.00. The encoding specificity effect was larger for BM25 (match-mismatch delta = 0.859) than for dense retrieval (delta = 0.590), indicating that lexical retrieval is more context-dependent than semantic retrieval. This is consistent with Tulving and Thomson's (1973) observation that retrieval success depends on cue-encoding overlap, with the degree of dependence varying by the nature of the representation. The pattern is also interpretable through the levels-of-processing framework (Craik & Lockhart, 1972): BM25's surface-level processing produces context-bound representations, while dense retrieval's semantic processing produces more transferable traces.

### 4.2 Fan Effect

Both systems showed decreasing retrieval accuracy with increasing associative fan (Figure 4). BM25 exhibited a monotonic decline from MRR = 1.000 at fan size 1, through MRR = 1.000 at fan size 2 and MRR = 0.964 at fan size 5, to MRR = 0.900 at fan size 10 (Spearman r_s = -0.949). Dense retrieval showed a comparable pattern: MRR declined from 0.345 at fan size 1, through 0.301 at fan size 2 and 0.168 at fan size 5, to 0.245 at fan size 10 (r_s = -0.800).

Both correlations indicate a human-like fan effect consistent with the core pattern from Anderson's (1974) original experiments and with ACT-R's prediction that retrieval competition increases with the number of associated items (Anderson et al., 2004). BM25 maintains higher absolute performance across fan sizes, reflecting its lexical precision advantage on documents containing unique domain keywords, rather than a difference in the underlying pattern.

### 4.3 Interference

Both systems exhibited proactive and retroactive interference, with an asymmetry matching the human pattern (Figure 5). Control entities, appearing in only one phase, achieved perfect retrieval (MRR = 1.000 for both systems).

For BM25, retroactive interference was substantially stronger than proactive interference (RI: MRR = 0.521; PI: MRR = 0.833), representing drops of 0.479 and 0.167 relative to control, respectively. For dense retrieval, the same asymmetry was observed (RI: MRR = 0.612; PI: MRR = 0.917), though both effects were smaller than in BM25.

This retroactive-exceeds-proactive asymmetry is one of the most consistent findings in interference theory (Underwood, 1957; Postman & Underwood, 1973; Wixted, 2004). Its emergence in retrieval systems not designed to exhibit it suggests the asymmetry arises from structural properties of competitive retrieval. Phase 2 documents describe more recent events with vocabulary reflecting current activities, which tends to overlap more with generic present-tense queries than Phase 1's historical descriptions, creating a recency-favoring competition dynamic.

BM25 showed stronger interference overall (CAS = 0.72) than dense retrieval (CAS = 0.58), consistent with greater susceptibility to competition from lexically overlapping content.

### 4.4 Serial Position

Neither system exhibited serial position effects. Both BM25 and dense retrieval achieved MRR = 1.000 at every sequence position (CAS = 0.00).

Both lexical and semantic retrieval architectures are order-invariant: the position at which a document was indexed has no effect on retrieval. BM25's inverted index and FAISS's flat inner product search are both invariant to insertion order by construction.

The result is relevant to the "lost in the middle" phenomenon (Liu et al., 2024), in which LLMs underweight information in the middle of long contexts. Our results indicate that this positional bias originates in the generation stage rather than the retrieval stage. Standard RAG retrieval does not introduce serial position biases before the LLM processes retrieved content. Mitigation efforts should therefore focus on the generation stage, such as passage reordering or position-aware attention (Peysakhovich & Lerer, 2023), rather than retrieval modification.

This null result may not generalize to all retrieval architectures. Systems using approximate nearest neighbor search, insertion-order-dependent data structures, or learned indices with sequential training may exhibit position-dependent behavior.

### 4.5 Retrieval-Induced Forgetting

Neither system exhibited retrieval-induced forgetting. Unpracticed items in practiced categories were retrieved with identical accuracy to baseline items (MRR = 1.000, CAS = 0.00).

The null result follows from the stateless nature of both systems. BM25 and dense retrieval do not modify their indices based on prior queries, so retrieving one document cannot affect the accessibility of related documents. Standard RAG systems are therefore immune to the inhibitory suppression mechanism proposed by Anderson (2003). However, this immunity may not persist in emerging adaptive architectures with session-dependent caching or reinforcement learning-based re-rankers (Chen et al., 2025). CogBench-RAG's RIF module provides a framework for detecting such effects as they emerge.

### 4.6 Summary

Across all five modules, both systems exhibited three human-like patterns (encoding specificity, fan effect, interference) and two null results (serial position, RIF). The overall pattern indicates that competition-based memory phenomena emerge naturally from similarity-based retrieval, while temporal and practice-dependent effects require mechanisms absent in standard architectures.

---

## 5. Discussion

### 5.1 Selective Cognitive Alignment

The results suggest that RAG retrieval systems exhibit a specific subset of human memory phenomena, namely those arising from competition during retrieval, while being unaffected by phenomena arising from temporal dynamics and practice-dependent plasticity.

Encoding specificity, the fan effect, and interference all emerge from competition among candidate documents. This competition is inherent to similarity-based ranking and produces patterns paralleling human memory retrieval, consistent with the principle that competition is a fundamental constraint on any retrieval system, biological or artificial (Anderson et al., 2004; Watkins & Watkins, 1975).

Serial position effects and retrieval-induced forgetting require temporal or state-dependent mechanisms absent in stateless retrieval. Serial position effects in human memory arise from differential rehearsal (Rundus, 1971) and working memory recency (Glanzer & Cunitz, 1966; Atkinson & Shiffrin, 1968). RIF requires retrieval-dependent suppression of competing representations (Anderson, 2003). Neither mechanism has a counterpart in standard inverted indices or vector stores.

### 5.2 Architectural Differences

The divergence between BM25 and dense retrieval on encoding specificity connects to the levels-of-processing framework (Craik & Lockhart, 1972). BM25 encodes at a surface level, producing context-bound representations where retrieval depends on lexical match. Dense retrieval encodes at a semantic level, producing representations that generalize across surface variations. Both exhibit encoding specificity, but the grain of specificity differs.

The distinction carries practical weight. Applications requiring robustness to query rephrasing, such as conversational RAG or multilingual retrieval, may benefit from semantic retrieval's reduced encoding specificity. Applications requiring precise contextual discrimination, such as legal document retrieval where terminology carries specific meaning, may benefit from the stronger encoding specificity of lexical approaches.

### 5.3 Implications for Cognitively-Inspired Design

For systems like HippoRAG (Gutiérrez et al., 2024) and ARM (Bursa, 2026) that explicitly pursue cognitive alignment, the results have specific implications. Competition-based phenomena appear inherent to similarity-based retrieval and may not require special engineering. If alignment on serial position or RIF is desired, fundamentally different mechanisms, such as position-aware indexing or retrieval-dependent index modification, would likely be needed.

### 5.4 Limitations

Several limitations should be noted. First, the controlled synthetic corpus provides experimental precision but limited ecological validity; naturalistic corpus validation is needed. Second, we evaluate two architectures; hybrid, graph-based (Gutiérrez et al., 2024; Edge et al., 2024), and adaptive systems may show different profiles. Third, corpus scale is modest, and scaling effects at realistic knowledge base sizes are unknown. Fourth, CAS normalization involves design choices warranting sensitivity analysis. Fifth, the mapping between cognitive tasks and retrieval tasks is approximate: documents are not episodic memories, and similarity-based ranking is not associative recall. Finally, human behavioral baselines from crowdsourced experiments would strengthen the comparison.

### 5.5 Future Work

Extensions include testing graph-based systems like HippoRAG for differential alignment; evaluating adaptive systems for emergent RIF; scaling corpus size; adding human baselines via Prolific or Amazon Mechanical Turk; extending modules to include the spacing effect (Cepeda et al., 2006), the testing effect (Roediger & Karpicke, 2006), levels-of-processing effects (Craik & Lockhart, 1972), and generation effects (Slamecka & Graf, 1978); and naturalistic corpus validation.

---

## 6. Conclusion

We introduced CogBench-RAG, a benchmark suite that evaluates retrieval-augmented generation systems against established human memory phenomena. Our evaluation indicates that BM25 and dense retrieval exhibit competition-based memory phenomena, including encoding specificity, the fan effect, and proactive/retroactive interference, while showing no evidence of serial position effects or retrieval-induced forgetting. The two architectures diverge on encoding specificity, with lexical retrieval showing near-absolute context dependence and semantic retrieval partially bridging contextual gaps. CogBench-RAG provides an extensible framework for evaluating the cognitive properties of future retrieval architectures. Code and data are available at https://github.com/DestrierStudios/cogbench-rag.

---

## Declarations

**Funding** Not applicable.

**Conflicts of interest/Competing interests** The author declares no conflicts of interest.

**Ethics approval** Not applicable. No human or animal subjects were involved in this research.

**Consent to participate** Not applicable.

**Consent for publication** Not applicable.

**Availability of data and materials** All data generated during this study are available at https://github.com/DestrierStudios/cogbench-rag.

**Code availability** All benchmark code, figure generation scripts, and Docker configuration are available at https://github.com/DestrierStudios/cogbench-rag under an MIT license.

**Authors' contributions** Not applicable (sole author).

---

## References

Anderson, J.R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6, 451-474.

Anderson, J.R., Bothell, D., Byrne, M.D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review*, 111(4), 1036-1060.

Anderson, J.R., & Reder, L.M. (1999). The fan effect: New results and new theories. *Journal of Experimental Psychology: General*, 128(2), 186-197.

Anderson, M.C. (2003). Rethinking interference theory: Executive control and the mechanisms of forgetting. *Journal of Memory and Language*, 49(4), 415-445.

Anderson, M.C., Bjork, R.A., & Bjork, E.L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(5), 1063-1087.

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In *Proceedings of ICLR 2024*.

Atkinson, R.C., & Shiffrin, R.M. (1968). Human memory: A proposed system and its control processes. In K.W. Spence & J.T. Spence (Eds.), *The Psychology of Learning and Motivation* (Vol. 2, pp. 89-195). Academic Press.

Baddeley, A., Eysenck, M.W., & Anderson, M.C. (2015). *Memory* (2nd ed.). Psychology Press.

Barnes, J.M., & Underwood, B.J. (1959). Fate of first-list associations in transfer theory. *Journal of Experimental Psychology*, 58(2), 97-105.

Binz, M., & Schulz, E. (2023). Using cognitive psychology to understand GPT-3. *Proceedings of the National Academy of Sciences*, 120(6), e2218523120.

Bursa, O. (2026). A dynamic retrieval-augmented generation system with selective memory and remembrance. arXiv:2601.02428.

Cepeda, N.J., Pashler, H., Vul, E., Wixted, J.T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354-380.

Chen, J., Lin, H., Han, X., & Sun, L. (2024). Benchmarking large language models in retrieval-augmented generation. In *Proceedings of AAAI 2024*.

Chen, Y., Yan, L., Sun, W., Ma, X., Zhang, Y., Wang, S., Yin, D., Yang, Y., & Mao, J. (2025). Improving retrieval-augmented generation through multi-agent reinforcement learning. arXiv:2501.15228.

Cheung, V., Maier, M., & Lieder, F. (2025). Large language models show amplified cognitive biases in moral decision-making. *Proceedings of the National Academy of Sciences*, 122(25), e2412015122.

Craik, F.I.M., & Lockhart, R.S. (1972). Levels of processing: A framework for memory research. *Journal of Verbal Learning and Verbal Behavior*, 11(6), 671-684.

Ebbinghaus, H. (1885). *Über das Gedächtnis*. Leipzig: Duncker & Humblot.

Edge, D., Trinh, H., Cheng, N., et al. (2024). From local to global: A graph RAG approach to query-focused summarization. arXiv:2404.16130.

Eich, J.E. (1980). The cue-dependent nature of state-dependent retrieval. *Memory & Cognition*, 8(2), 157-173.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. In *Proceedings of EACL 2024*.

Fountas, Z., et al. (2025). Human-inspired episodic memory for infinite context LLMs. In *Proceedings of ICLR 2025*.

Germain, P.L., Sonrel, A., & Robinson, M.D. (2020). pipeComp: A general framework for the evaluation of computational pipelines. *Genome Biology*, 21, 227.

Glanzer, M., & Cunitz, A.R. (1966). Two storage mechanisms in free recall. *Journal of Verbal Learning and Verbal Behavior*, 5(4), 351-360.

Godden, D.R., & Baddeley, A.D. (1975). Context-dependent memory in two natural environments. *British Journal of Psychology*, 66, 325-331.

Gutiérrez, B.J., et al. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. In *NeurIPS 2024*.

Hagendorff, T., Fabi, S., & Kosinski, M. (2023). Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT. *Nature Computational Science*, 3(10), 833-838.

Jiang, Z., et al. (2023). Active retrieval augmented generation. In *Proceedings of EMNLP 2023*.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

Kahana, M.J. (2012). *Foundations of Human Memory*. Oxford University Press.

Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of EMNLP 2020*.

Kim, S.H., et al. (2025). LLM reasoning does not protect against clinical cognitive biases. medRxiv, 2025.06.22.25330078.

Kliegl, O., & Bäuml, K.-H.T. (2021). Buildup and release from proactive interference. *Neuroscience & Biobehavioral Reviews*, 120, 264-278.

Koo, R., et al. (2024). Benchmarking cognitive biases in large language models as evaluators. In *Findings of ACL 2024*, 517-545.

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *NeurIPS 2020*.

Liu, N.F., et al. (2024). Lost in the middle: How language models use long contexts. *Transactions of the ACL*, 12, 157-173.

Luecken, M.D., & Theis, F.J. (2019). Current best practices in single-cell RNA-seq analysis. *Molecular Systems Biology*, 15(6), e8746.

Ma, X., et al. (2021). A replication study of dense passage retriever. arXiv:2104.05740.

Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

Marian, V., & Neisser, U. (2000). Language-dependent recall of autobiographical memories. *Journal of Experimental Psychology: General*, 129(3), 361-368.

McGeoch, J.A. (1932). Forgetting and the law of disuse. *Psychological Review*, 39(4), 352-370.

McGaugh, J.L. (2000). Memory: A century of consolidation. *Science*, 287(5451), 248-251.

Muennighoff, N., et al. (2022). MTEB: Massive text embedding benchmark. arXiv:2210.07316.

Müller, G.E., & Pilzecker, A. (1900). Experimentelle Beiträge zur Lehre vom Gedächtnis. *Zeitschrift für Psychologie*, Supplement 1.

Murayama, K., Miyatsu, T., Buchli, D., & Storm, B.C. (2014). Forgetting as a consequence of retrieval: A meta-analytic review. *Psychological Bulletin*, 140(5), 1383-1409.

Murdock, B.B. (1962). The serial position effect of free recall. *Journal of Experimental Psychology*, 64, 482-488.

Osgood, C.E. (1949). The similarity paradox in human learning. *Psychological Review*, 56(3), 132-143.

Peysakhovich, A., & Lerer, A. (2023). Attention sorting combats recency bias in long context language models. arXiv:2310.01427.

Postman, L., & Underwood, B.J. (1973). Critical issues in interference theory. *Memory & Cognition*, 1, 19-40.

Radvansky, G.A., Spieler, D.H., & Zacks, R.T. (1993). Mental model organization. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 19(1), 95-114.

Radvansky, G.A., & Zacks, J.M. (2014). *Event Cognition*. Oxford University Press.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP 2019*.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

Roediger, H.L., & Karpicke, J.D. (2006). Test-enhanced learning. *Psychological Science*, 17(3), 249-255.

Rundus, D. (1971). Analysis of rehearsal processes in free recall. *Journal of Experimental Psychology*, 89(1), 63-77.

Saad-Falcon, J., et al. (2024). ARES: Automated evaluation framework for RAG systems. In *Proceedings of NAACL 2024*.

Slamecka, N.J., & Graf, P. (1978). The generation effect. *Journal of Experimental Psychology: Human Learning and Memory*, 4(6), 592-604.

Suri, G., et al. (2024). Do LLMs show decision heuristics similar to humans? *Journal of Experimental Psychology: General*, 153(4), 1066-1075.

Teyler, T.J., & Discenna, P. (1986). The hippocampal memory indexing theory. *Behavioral Neuroscience*, 100(2), 147-154.

Thakur, N., et al. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of IR models. In *NeurIPS 2021 Datasets and Benchmarks Track*.

Tulving, E. (1972). Episodic and semantic memory. In E. Tulving & W. Donaldson (Eds.), *Organization of Memory* (pp. 381-403). Academic Press.

Tulving, E., & Thomson, D.M. (1973). Encoding specificity and retrieval processes in episodic memory. *Psychological Review*, 80, 352-373.

Underwood, B.J. (1957). Interference and forgetting. *Psychological Review*, 64, 49-60.

Wang, W., et al. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In *NeurIPS 2020*.

Watkins, M.J., & Watkins, O.C. (1975). Buildup of proactive inhibition as a cue-overload effect. *Journal of Experimental Psychology: Human Learning and Memory*, 1(4), 442-452.

Wixted, J.T. (2004). The psychology and neuroscience of forgetting. *Annual Review of Psychology*, 55, 235-269.

Zacks, J.M., Speer, N.K., Swallow, K.M., Braver, T.S., & Reynolds, J.R. (2007). Event perception: A mind-brain perspective. *Psychological Bulletin*, 133(2), 273-293.
