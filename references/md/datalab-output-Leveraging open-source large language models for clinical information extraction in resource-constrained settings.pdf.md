

## Research and Applications

# Leveraging open-source large language models for clinical information extraction in resource-constrained settings

Luc Builtjes ![ORCID icon](e3f8612927870f2e0f9f5989e6dd3064_img.jpg) **MSc**<sup>1,\*</sup>, Joeran Bosma ![ORCID icon](a86c7d1c9cb81c81614634a31267440d_img.jpg) **MSc**<sup>1</sup>, Mathias Prokop ![ORCID icon](ce158fc5e55633398941d0898ae45661_img.jpg) **PhD, MD**<sup>1</sup>,  
Bram van Ginneken ![ORCID icon](6f77f2588732dff582d5f470675e762f_img.jpg) **PhD**<sup>1</sup>, Alessa Hering ![ORCID icon](802fbc25d869d680d37bfef9949fa598_img.jpg) **PhD**<sup>1</sup>

<sup>1</sup>Department of Radiology and Nuclear Medicine, Radboud University Medical Center, 6525GA Nijmegen, The Netherlands

\*Corresponding author: Luc Builtjes, MSc, Department of Radiology and Nuclear Medicine, Radboud University Medical Center, Geert Grooteplein Zuid 10, 6525GA Nijmegen, The Netherlands (luc.builtjes@radboudumc.nl)

## Abstract

**Objective:** We aimed to evaluate the zero-shot performance of open-source generative large language models (LLMs) on clinical information extraction from Dutch medical reports using the Diagnostic Report Analysis: General Optimization of NLP (DRAGON) benchmark.

**Methods:** We developed and released the `11m_extractor` framework, a scalable, open-source tool for automating information extraction from clinical texts using LLMs. We evaluated 9 multilingual open-source LLMs across 28 tasks in the DRAGON benchmark, covering classification, regression, and named entity recognition (NER). All tasks were performed in a zero-shot setting. Model performance was quantified using task-specific metrics and aggregated into a DRAGON utility score. Additionally, we investigated the effect of in-context translation to English.

**Results:** Llama-3.3-70B achieved the highest utility score (0.760), followed by Phi-4-14B (0.751), Qwen-2.5-14B (0.748), and DeepSeek-R1-14B (0.744). These models outperformed or matched a fine-tuned RoBERTa baseline on 17 of 28 tasks, particularly in regression and structured classification. NER performance was consistently low across all models. Translation to English consistently reduced performance.

**Discussion:** Generative LLMs demonstrated strong zero-shot capabilities on clinical natural language processing tasks involving structured inference. Models around 14B parameters performed well overall, with Llama-3.3-70B leading but at high computational cost. Generative models excelled in regression tasks, but were hindered by token-level output formats for NER. Translation to English reduced performance, emphasizing the need for native language support.

**Conclusion:** Open-source generative LLMs provide a viable zero-shot alternative for clinical information extraction from Dutch medical texts, particularly in low-resource and multilingual settings.

## Lay Summary

Our study tested the ability of open-source artificial intelligence (AI) language models to understand and extract information from Dutch medical reports without any prior training on the tasks. Using our custom `11m_extractor` tool, we evaluated 9 models on 28 clinical tasks in the DRAGON 2024 benchmark. Several models around 14 billion parameters performed competitively, coming close to matching a top-performing, fine-tuned system built specifically for these tasks. Surprisingly, even without training, the generative models often outperformed the specialized model on several tasks, especially those involving numbers and structured reasoning. However, all models struggled with tasks that required identifying specific medical terms, likely due to the format required by the challenge. We also found that translating Dutch reports into English before analysis made results worse. Overall, our results suggest that modern generative AI can support clinical data analysis in Dutch with minimal setup, offering a flexible alternative to traditional models that require large amounts of labeled training data.

**Key words:** natural language processing; large language models; information storage and retrieval; artificial intelligence.

## Introduction

Medical reports contain highly detailed patient information, including diagnoses, procedures, medications, and clinical observations, making them a valuable resource for data analysis for large-scale medical research.<sup>1</sup> This density of clinically relevant information is especially valuable for developing artificial intelligence (AI) applications in healthcare, which depend on large, well-labeled datasets.<sup>2</sup> When processed effectively, these reports can yield a wide variety of training labels, supporting the development of accurate and generalizable AI models.

The utility of medical reports is often limited by their unstructured textual format, which can vary significantly across institutions and individual practitioners.<sup>3</sup> Combined with the frequent use of domain-specific medical jargon, this lack of standardization presents a major challenge for information extraction, a critical step in converting raw clinical narratives into structured, machine-readable data.

Traditionally, the field of natural language processing (NLP) has relied on rule-based systems for information extraction, though these methods tend to struggle greatly with unstructured

Received: June 30, 2025; Revised: August 25, 2025; Accepted: August 30, 2025

© The Author(s) 2025. Published by Oxford University Press on behalf of the American Medical Informatics Association.

This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License (<https://creativecommons.org/licenses/by-nc/4.0/>), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact [reprints@oup.com](mailto:reprints@oup.com) for reprints and translation rights for reprints. All other permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site—for further information please contact [journals.permissions@oup.com](mailto:journals.permissions@oup.com).

text.<sup>4</sup> The emergence of transformer-based models, such as Bidirectional Encoder Representations from Transformers (BERT),<sup>5</sup> enabled the extraction and structuring of meaningful data from more complex text. Domain-specific adaptations like Med-BERT<sup>6</sup> have further refined these capabilities, achieving state-of-the-art performance in tasks like text classification and extraction. However, their effectiveness hinges on the availability of large quantities of labeled training data, which limits their scalability and adaptability for new tasks.

Recent advancements in generative Large Language Models (LLMs) have introduced a transformative shift in NLP. These models can be adapted to diverse tasks through the use of prompting techniques, reducing or even eliminating the reliance on task-specific training data. Their application in healthcare has already shown promise in areas such as clinical decision support,<sup>7-10</sup> medical text summarization,<sup>11</sup> and question answering.<sup>12,13</sup>

However, a substantial portion of the current literature<sup>7-9,12-20</sup> is focused primarily on proprietary models such as OpenAI's GPT-4,<sup>21</sup> which pose challenges related to transparency, reproducibility, and ethical concerns in clinical applications. These systems generally require transmitting data via an API to external servers where the models are hosted. This approach raises significant concerns under modern privacy regulations governing medical data, which mandate strict oversight over any information leaving hospital IT systems. Additionally, the training of many proprietary models on mostly undisclosed datasets raises ethical questions about data sourcing and contamination, privacy, and representativeness.<sup>22</sup>

To address these limitations, the development and application of open-source LLMs have gained significant attention. These models offer researchers the opportunity to evaluate and adapt LLMs for medical tasks while ensuring greater accountability and control over input data by maintaining operations within local infrastructure. Open-source models also generally provide greater transparency regarding their pretraining datasets, enabling a clearer understanding of their limitations and biases.<sup>23</sup>

One of such limitations is the ability to effectively handle mid- to low-resource languages. Medical reports are predominantly written in the primary language of the care facility where they are produced. Proprietary models like GPT-4 benefit from extensive pre-training on datasets obtained through large-scale web scraping, which often include a variety of languages. In contrast, open-source LLMs are typically pre-trained on more curated datasets, leading to a disproportionate representation of high-resource languages such as English, Chinese, and Spanish. This imbalance results in a significant performance gap between these widely spoken languages and less common ones.<sup>24</sup> The challenge is further compounded when dealing with text rich in specialized jargon, such as the terminology found in medical contexts, where the disparity in linguistic resources becomes even more pronounced. Despite the practical significance of these issues, research on the performance of open-source models in such contexts remains limited.<sup>25-27</sup>

The introduction of the Diagnostic Report Analysis: General Optimization of NLP (DRAGON) challenge<sup>28</sup> provides a valuable benchmark for addressing this issue. DRAGON includes 28 824 annotated medical reports from 5 care centers, covering 28 medically relevant information extraction tasks, such as classification, regression, and named entity recognition (NER), in the relatively uncommon Dutch language.

In this work, we present a systematic evaluation of several widely used open-source LLMs on domain-specific, resource-constrained language texts, with a focus on medical information extraction tasks. Our objective is to build a knowledge base identifying which models are most suitable for specific tasks in this setting, highlighting their strengths and limitations across various applications.

To support this evaluation, we developed a user-friendly and scalable framework that automates the application of open-source LLMs to diverse information extraction tasks on medical datasets in a language-agnostic manner. The framework enforces structured JavaScript Object Notation (JSON) output generation, enabling standardized and machine-readable outputs that facilitate both seamless evaluation and integration into downstream clinical or analytical pipelines. This design lowers the barrier to entry for deploying such models in complex, domain-specific contexts, while ensuring consistency and usability of the extracted information. Furthermore, the framework includes a web-based user interface that eliminates the need for any coding expertise, allowing less technical researchers to easily configure tasks, define output formats, and run models through an intuitive point-and-click environment.

The main contributions of our work are as follows:

- 1) We introduce and publicly release `llm_extractorator`, a scalable, language-agnostic, open-source framework for automating data extraction tasks with LLMs, designed for ease of use and broad applicability. It is available at [https://github.com/DIAGNijmegen/llm\\_extractorator](https://github.com/DIAGNijmegen/llm_extractorator).
- 2) We perform a comprehensive evaluation of 9 widely used open-source LLMs on 28 medically relevant information extraction tasks using Dutch clinical reports in a zero-shot setting, as visually summarized in [Figure 1](#). This evaluation offers a realistic estimate for model performance and practical insights into the utility of generative LLMs in resource-constrained, domain-specific environments. All associated code repositories are available in [Note S5](#).

By focusing on smaller, open-source generative models, our work contributes to bridging the gap between state-of-the-art AI capabilities and practical, real-world applications in health-care. This study not only fills a critical void in the literature but also lays the foundation for future research in leveraging open-source LLMs for multilingual and resource-constrained medical environments.

## Methods

### `llm_extractorator`

`llm_extractorator` ([https://github.com/DIAGNijmegen/llm\\_extractorator](https://github.com/DIAGNijmegen/llm_extractorator)) is an open-source pipeline that converts free text into structured JSON outputs based on user-defined schemas. [Figure 2](#) gives an overview; below we describe the 4 stages of every run.

### Task specification

A *Taskfile* (JSON) defines the data source, task description, the field to parse, and the desired output schema, which can be either Pydantic or pure JSON. Taskfiles can be created using the integrated Streamlit Studio GUI, offering a drag-and-drop interface with live schema previews, or by directly editing the JSON and parser script. Additionally, a schema-

![Figure 1: Overview of our submissions to the DRAGON 2024 challenge using the llm_extractorator framework. The diagram shows a workflow from Clinical NLP Tasks and Open-Source LLMs to the llm_extractorator framework, which involves Data preprocessing, Prompt generation, Model Inference, and Output parsing, leading to Submission to DRAGON 2024. The submission includes a Taskfile and Evaluation metrics.](7055f51feb10ea4ea48b27c36f085286_img.jpg)

**Clinical NLP Tasks**

- Classification**
  - 8 Single Label Binary
  - 6 Single Label Multi Class
  - 2 Multi Label Binary
  - 2 Multi Label Multi Class
- Regression**
  - 5 Single Label
  - 1 Multi Label
- Natural Entity Recognition**
  - 2 Single Label
  - 2 Multi Label

**Open-Source LLMs**

- Mistral
- Llama-3.1-8B
- Llama-3.2-3B
- Llama-3.3-70B
- Gemma-2-2B
- Gemma-2-9B
- Phi-4-14B
- Owen-2.5-14B
- DeepSeek-R1-14B

**llm\_extractorator**

- Data preprocessing**
  - Optional translation to English
  - Adjusting context length based on document token count
- Prompt generation** Using description from the Taskfile
- Model Inference** Zero-shot
- Output parsing** Based on task-specific JSON format

**Submission to DRAGON 2024**

**Taskfile**

- JSON file including
  - Manually created descriptions of the tasks
  - Manually created output formats for each task

**Evaluation**

- Calculate metric per task (AUC, Cohen's kappa, RSMAPE, F1)
- Arithmetic mean over all tasks for DRAGON 2024 utility score  $S_{\text{DRAGON}}$

Figure 1: Overview of our submissions to the DRAGON 2024 challenge using the llm\_extractorator framework. The diagram shows a workflow from Clinical NLP Tasks and Open-Source LLMs to the llm\_extractorator framework, which involves Data preprocessing, Prompt generation, Model Inference, and Output parsing, leading to Submission to DRAGON 2024. The submission includes a Taskfile and Evaluation metrics.

**Figure 1.** Overview of our submissions to the DRAGON 2024 challenge using the llm\_extractorator framework. We evaluate 9 distinct LLMs: Mistral-Nemo-12B,<sup>29</sup> Llama-3.1-8B, Llama-3.2-3B, and Llama-3.3-70B,<sup>30</sup> Gemma-2-2B and Gemma-2-9B,<sup>31</sup> Phi-4-14B,<sup>32</sup> Qwen-2.5-14B,<sup>33</sup> and DeepSeek-R1-14B.<sup>34</sup> The input token length of each of the 28 clinical NLP tasks is measured, and model context windows are adapted accordingly. In 3 experiments, input text is translated to English using the LLM itself. For each task, we define a description and expected output format in a JSON-based Taskfile. This metadata guides prompt generation, which is followed by model inference and automatic output parsing. Task performance is assessed using the appropriate metric (AUC, Cohen's kappa, RSMAPE, or F1), and the final DRAGON 2024 utility score  $S_{\text{DRAGON}}$  is computed as the arithmetic mean across all task metrics.

![Figure 2: Overview of the llm_extractorator pipeline. The diagram shows a workflow from Task specification to Prompt construction, Model inference, and Validation.](163688ca8da9787f5d48edd68d8cc75b_img.jpg)

**Task specification**

- JSON Taskfile
- Output parser
- Data json or csv

**Prompt construction**

- LangChain
- System prompt with CoT reasoning
- Task description
- Input text
- Output schema

**Model inference**

- Pull Ollama model
- Calculate context length based on input data
- Host model locally

**Validation**

- Parse output schema
- Retry failed cases
- Flag failures

Figure 2: Overview of the llm\_extractorator pipeline. The diagram shows a workflow from Task specification to Prompt construction, Model inference, and Validation.

**Figure 2.** Overview of the llm\_extractorator pipeline. The framework consists of 4 modular stages. (1) **Task specification**: users define the input source, task description, and output schema. (2) **Prompt construction**: the framework leverages LangChain to build a prompt that combines task instructions, the output schema, and zero-shot chain-of-thought reasoning cues. (3) **Model inference**: the prompt is sent to an LLM served through Ollama, which handles local deployment and manages resource-efficient inference through context-length scaling. (4) **Post-processing and validation**: the LLM output is parsed into the user-defined schema, with automatic reformating of invalid responses and fallback placeholder injection to ensure pipeline continuity. Each stage is configurable but designed to operate out-of-the-box with default settings.

builder tool is included to assist in creating a Pydantic parser without any necessary coding knowledge.

output schema. For non-English input, optional automatic translation can be applied prior to extraction.

### Prompt construction

At inference time, LangChain<sup>35</sup> dynamically constructs a prompt by combining the raw input text, output schema, and task description. The prompt includes a general system message with zero-shot chain-of-thought reasoning instructions,<sup>36</sup> enforced through an explicit reasoning field in the

### Model inference

llm\_extractorator supports any LLM hosted on the Ollama hub.<sup>37</sup> The framework automatically downloads the selected model and runs it locally with user-defined inference settings. It dynamically determines the appropriate context length based on input size. For improved efficiency, inputs

can optionally be split into short and long subsets, each processed with a tailored context window.

### Post-processing and validation

Model outputs are validated against the defined schema using Ollama's enforced formatting. If validation fails, the framework initiates up to 3 self-correction cycles, in which the model receives its previous response and a list of schema violations. Remaining failures are replaced with empty or random placeholders to prevent downstream pipeline interruptions. These cases can later be flagged for manual review, though this functionality was disabled in the current study, where evaluation was performed fully automatically.

### DRAGON challenge

To evaluate model performance, we applied our framework to the DRAGON challenge,<sup>28</sup> a benchmark initiative for clinical NLP tasks in Dutch. The DRAGON dataset comprises 28 824 annotated medical reports collected from 5 Dutch healthcare institutions, covering 28 clinically relevant tasks. These tasks span a diverse range of categories: 8 single-label binary classification tasks, 6 single-label multi-class classification tasks, 2 multilabel binary classification tasks, 2 multilabel multi-class classification tasks, 5 single-label regression tasks, 1 multilabel NER tasks, and 2 multilabel NER tasks. A complete overview of the tasks is provided in [Figure 3](#) and in [Note S1](#). A representative input text for each task is available at: [https://github.com/DIAG-Nijmegen/dragon\\_sample\\_reports](https://github.com/DIAG-Nijmegen/dragon_sample_reports).

The challenge is hosted on the Grand Challenge platform,<sup>38</sup> a fully cloud-based environment powered by Amazon Web Services. The full challenge workflow involves 2 stages: fine-tuning BERT-like models on a provided training and validation dataset, followed by inference on a separate test set. However, since our study aims to evaluate the zero-shot capabilities of generative LLMs, we bypassed the fine-tuning phase and conducted direct inference via prompting.

The outputs generated by the models were post-processed into the JSON format required for automatic evaluation. These predictions were then evaluated against the ground truth test labels, producing task-specific performance scores. For the binary classification tasks (T1-T8), performance was assessed using the Area Under the Receiver Operating Characteristic Curve (AUC). For the multi-class classification tasks (T9-T14), performance was measured using either unweighted or linearly weighted Cohen's Kappa, depending on the task. Multi-label classification tasks (T15-T18) were evaluated using either macro-averaged AUC or unweighted Kappa. Regression tasks (T19-T24) used the Robust Symmetric Mean Absolute Percentage Error Score (RSMAPEs) with task-specific tolerance margins. Named entity recognition tasks (T25-T28) were evaluated using macro or weighted F1 scores. All metrics were computed automatically by the Grand Challenge platform upon submission, and we did not have direct access to any per-example results.

The individual task scores were aggregated into the DRAGON 2024 utility score  $S_{\text{DRAGON}}$ , computed as the arithmetic mean of the performance metrics across all 28 tasks. While the standard DRAGON evaluation protocol recommends five test runs using different random seeds to account for sampling variability, we opted for a single test run by setting the model's sampling temperature to 0. This configuration enforces fully deterministic outputs in token

generation, allowing for more stable, direct comparisons of zero-shot model performance. This approach also offers practical advantages by substantially reducing computational costs, as it eliminates the need for 4 additional inference runs per model.

### Models

We evaluated 9 widely used open-source multilingual LLMs available through the Ollama model hub: Llama3.1-8B, Llama3.2-3B, and Llama3.3-70B,<sup>30</sup> Gemma2-2B and Gemma2-9B,<sup>31</sup> Phi4-14B,<sup>32</sup> Qwen2.5-14B,<sup>33</sup> DeepSeek-R1-14B,<sup>34</sup> and Mistral-NeMo.<sup>29</sup> For inference efficiency, all models were run in 4-bit quantized format. Specifically, we used the q4\_0 quantization scheme for Mistral-Nemo and Gemma2, and q4\_K\_M for the remaining models. These configurations reflect the default quantization settings provided by the Ollama model hub.

Our primary focus was on models with fewer than 15 billion parameters to ensure practical feasibility within typical hospital IT environments. All such models can be run on consumer-grade GPUs with 12GB of VRAM when quantized to 4-bit precision. Although the llm\_extractor framework supports CPU offloading to accommodate larger models on limited hardware, this results in significant reductions in processing speed, rendering it impractical for clinical deployment. Given that most healthcare facilities lack ready access to high-performance GPUs, we prioritized smaller models for this study. Nevertheless, we also included one larger model (Llama-3.3-70B) to serve as a high-performance benchmark for institutions with greater computational resources. This helps contextualize the performance of smaller models and provides a reference point for future hardware scaling scenarios.

### Prompting strategies

All experiments were conducted under a strict zero-shot setting, meaning that no task-specific fine-tuning was employed and no examples were provided in-context. This approach tests the models' inherent capabilities to perform unfamiliar medical tasks based solely on their pretrained knowledge. To encourage model reasoning, we consistently applied zero-shot chain-of-thought prompting across all tasks. A full list of all prompts used is provided in [Note S4](#).

Moreover, we investigated the effect of translating the original Dutch reports into English prior to inference, given that the LLMs were predominantly trained on English corpora. This intermediate translation step was hypothesized to improve performance by aligning the input language with the models' primary training distribution, thus potentially reducing comprehension errors arising from linguistic mismatch.

### Statistical analysis

Task-level scores with and without in-context translation were compared for each model using the non-parametric Kruskal-Wallis H-test. *P*-values were Holm-Bonferroni adjusted ( $\alpha=0.05$ ), and effect sizes reported as matched-pairs rank-biserial correlations *r*. Statistical analyses were performed using Python 3.11 with SciPy v1.15.3.

## Results

We evaluated 9 publicly available generative LLMs on the 28 tasks of the DRAGON challenge using the llm\_extractor framework under zero-shot conditions. A representative

![Icon of two hands holding a small object. Icon of lungs with a blue circle. Icon of kidneys with a blue circle. Icon of a document being discarded. Icon of a clock. Icon of lungs with a question mark. Icon of lungs with a question mark. Icon of a hand with a question mark. Icon of a pancreas. Icon of a prostate with a monitor. Icon of a prostate with a magnifying glass. Icon of a surgical tool. Icon of a person with location pins. Icon of a document with a question mark. Icon of a colon. Icon of a person with question marks. Icon of a pancreas. Icon of a hip joint. Icon of a prostate. Icon of a prostate. Icon of a prostate with a test tube. Icon of a pancreas. Icon of lungs. Icon of a person with a test tube. Icon of a person with a question mark. Icon of a document with a question mark. Icon of a prostate with a biopsy needle. Icon of a hand with a blue circle.](7a3561af571faf036baa93f5f4b1bdb9_img.jpg)

Single-label binary classification

|                           |                                      |                                           |                                            |
|---------------------------|--------------------------------------|-------------------------------------------|--------------------------------------------|
| <br>T1: Adhesion presence | <br>T2: Pulmonary nodule presence    | <br>T3: Kidney abnormality identification | <br>T4: Skin histopathology case selection |
| <br>T5: RECIST timeline   | <br>T6: Histopathology cancer origin | <br>T7: Pulmonary nodule size presence    | <br>T8: PDAC size presence                 |

Single-label multi-class classification

|                                       |                                                |                                                      |                                     |
|---------------------------------------|------------------------------------------------|------------------------------------------------------|-------------------------------------|
| <br>T9: PDAC diagnosis                | <br>T10: Prostate radiology suspicious lesions | <br>T11: Prostate histopathology significant cancers | <br>T12: Histopathology sample type |
| <br>T13: Histopathology sample origin | <br>T14: Entailment diagnostic sentences       |                                                      |                                     |

Multi-label binary classification

|                                         |                                      |
|-----------------------------------------|--------------------------------------|
| <br>T15: Colon histopathology diagnosis | <br>T16: RECIST lesion size presence |
|-----------------------------------------|--------------------------------------|

Multi-label multi-class classification

|                          |                                        |                                      |                                                |
|--------------------------|----------------------------------------|--------------------------------------|------------------------------------------------|
| <br>T17: PDAC attributes | <br>T18: Hip Kellgren-Lawrence scoring | <br>T19: Prostate volume measurement | <br>T20: Prostate specific antigen measurement |
|--------------------------|----------------------------------------|--------------------------------------|------------------------------------------------|

Single-label regression

|                                                        |                                |                                            |                                         |
|--------------------------------------------------------|--------------------------------|--------------------------------------------|-----------------------------------------|
| <br>T21: Prostate specific antigen density measurement | <br>T22: PDAC size measurement | <br>T23: Pulmonary nodule size measurement | <br>T24: RECIST lesion size measurement |
|--------------------------------------------------------|--------------------------------|--------------------------------------------|-----------------------------------------|

Single-label named entity recognition

|                        |                                          |
|------------------------|------------------------------------------|
| <br>T25: Anonymization | <br>T26: Medical terminology recognition |
|------------------------|------------------------------------------|

Multi-label named entity recognition

|                                   |                                        |
|-----------------------------------|----------------------------------------|
| <br>T27: Prostate biopsy sampling | <br>T28: Skin histopathology diagnosis |
|-----------------------------------|----------------------------------------|

Legend:

- Number of reports: Blue bar
- Median report length: Orange bar
- Max. report length: Green bar
- 10,000: Solid line
- 1000: Dashed line
- 100: Dotted line

Icon of two hands holding a small object. Icon of lungs with a blue circle. Icon of kidneys with a blue circle. Icon of a document being discarded. Icon of a clock. Icon of lungs with a question mark. Icon of lungs with a question mark. Icon of a hand with a question mark. Icon of a pancreas. Icon of a prostate with a monitor. Icon of a prostate with a magnifying glass. Icon of a surgical tool. Icon of a person with location pins. Icon of a document with a question mark. Icon of a colon. Icon of a person with question marks. Icon of a pancreas. Icon of a hip joint. Icon of a prostate. Icon of a prostate. Icon of a prostate with a test tube. Icon of a pancreas. Icon of lungs. Icon of a person with a test tube. Icon of a person with a question mark. Icon of a document with a question mark. Icon of a prostate with a biopsy needle. Icon of a hand with a blue circle.

**Figure 3.** An overview of the 28 different tasks of the DRAGON challenge grouped by task type. The bar graphs show the number of reports, the median report length, and the maximum report length in each dataset based on tokenization using an xlm-roberta-base tokenizer. Figure reproduced with permission from Bosma et al.<sup>28</sup>

input text for each tasks is available at: [https://github.com/DIAGNijmegen/dragon\\_sample\\_reports](https://github.com/DIAGNijmegen/dragon_sample_reports).

We followed the proposed metrics of the challenge organizers to quantify performance: area under the receiver-operating-characteristic curve (AUC) for binary classification, Cohen's  $\kappa$  for multi-class classification, RSMAPES for regression, and  $F_1$  score for NER. To facilitate model-level comparisons, we utilize the DRAGON 2024 utility score,  $S_{\text{DRAGON}}$ , defined as the arithmetic mean of each model's performance across all 28 tasks. The resulting score lies in the range [0,1], with 1 indicating perfect performance.

Additionally, the challenge organizers provide interpretability thresholds per metric which we use to categorize each result into one of 6 qualitative performance tiers: *Excellent*, *Good*, *Moderate*, *Poor*, *Minimal*, or *Fail*. Additional details on the metrics can be found in [Note S2](#). The full results are documented in [Note S3](#).

### Model-level performance

Model performance naturally clustered into 3 general tiers. The top-performing group consisted of Llama-3.3-70B ( $S_{\text{DRAGON}} = 0.760$ ), Phi-4-14B (0.751), Qwen-2.5-14B (0.748), and DeepSeek-R1-14B (0.744). Llama-3.3-70B scores best with an *Excellent* performance on 12 out of 28 tasks. Phi-4-14B achieved this performance on 10 out of 28 tasks, followed closely by Qwen-2.5-14B and DeepSeek-R1-14B, each with 9.

A second tier included Gemma2-9B and Mistral-Nemo-12B, both achieving  $S_{\text{DRAGON}} = 0.688$ , with *Good* or better performance on roughly half the tasks. Llama-3.1-8B scored notably lower ( $S_{\text{DRAGON}} = 0.588$ ), achieving *Excellent* or *Good* performance on just 7 tasks.

The lowest tier comprised Llama-3.2-3B ( $S_{\text{DRAGON}} = 0.271$ ), which achieved only *Minimal* to *Fail* performance across all tasks. Gemma2-2B consistently failed to produce valid JSON outputs and thus could not be evaluated meaningfully.

[Table 1](#) summarizes  $S_{\text{DRAGON}}$  scores alongside the number of tasks each model performed at each qualitative level. RoBERTa large with domain-specific pretraining, the best performing baseline model provided by the challenge organizers, is included for reference. [Figure 4](#) visualizes average performance per task type across all models we tested. While the tiered structure is generally consistent across task types, certain models exhibit domain-specific strengths. For instance, Mistral-Nemo-12B performed comparably to top-tier models on regression tasks but underperformed on multilabel classification. Conversely, Gemma2-9B demonstrated relatively

weaker performance on regression despite competitive results in other task types.

### Task-level performance

[Figure 5](#) shows task-specific performance distributions for the top 4 models. Across the 6 regression tasks, all models achieved scores  $\ge 0.87$ , with an average RSMAPES of 0.971 and 22 out of 24 model-task combinations rated *Excellent*. Binary classification tasks showed greater variability: while the group mean AUC of the 4 models over all tasks was 0.84, certain tasks (eg, T04 and T06) saw performance near chance level for at least one model.

Ordinal classification tasks revealed broad score distributions, with Cohen's  $\kappa$  values ranging from 0.51 to 0.98. The largest intra-task spread (T14,  $\sigma = 0.09$ ) illustrates the potential impact of model selection. Some tasks (eg, T10 and T12) consistently produced high scores with low inter-model variability. In contrast, tasks T11, T14, and T18 displayed high variance and low scores, indicating task-level difficulty or sensitivity to model architecture.

NER performance was uniformly poor: none of the evaluated models exceeded an  $F_1$  score of 0.47. The modal qualitative label for NER tasks was *Fail*.

### Comparison to BERT-style baseline model

[Table 2](#) provides a detailed task-by-task comparison between the current top-performing model in the DRAGON 2024 challenge, DRAGON RoBERTa Large Domain-specific (<https://grand-challenge.org/algorithms/dragon-roberta-large-domain-specific/>), and our best performing model Llama-3.3 (<https://grand-challenge.org/algorithms/llm-extractinator-llama33/>). The better-performing model for each task is highlighted in bold. RoBERTa's results are reported as the mean and SD from 5-fold cross-validation, where the Llama-3.3 scores are derived from a single deterministic inference run with zero temperature.

The RoBERTa model achieved a higher overall DRAGON 2024 utility score ( $S_{\text{DRAGON}} = 0.819 \pm 0.021$ ) than any of the generative LLMs. However, across the 28 tasks, DRAGON RoBERTa Large Domain-specific achieved higher scores than Llama-3.3 on only 11 tasks (T01, T02, T04, T05, T09, T11, T15, T25, T26, T27, T28), with performance differences exceeding the upper bound of RoBERTa's SD, whereas the Llama model scored higher than RoBERTa on 14 tasks (T06, T07, T08, T12, T13, T16, T17, T18, T19, T20, T21, T22, T23, T24). For the remaining 3 tasks (T03, T10, and T14), the score of the generative LLM fell within one SD of RoBERTa's mean score.

**Table 1.** DRAGON 2024 utility scores ( $S_{\text{DRAGON}}$ ) and qualitative ratings across the 28 DRAGON tasks.

| Model            | $S_{\text{DRAGON}}$ | Excellent | Good | Moderate | Poor | Minimal | Fail |
|------------------|---------------------|-----------|------|----------|------|---------|------|
| LLaMA 3.3 70B    | 0.760               | 12        | 3    | 7        | 3    | 0       | 3    |
| Phi-4 14B        | 0.751               | 10        | 6    | 5        | 4    | 0       | 3    |
| Qwen 2.5 14B     | 0.748               | 9         | 7    | 6        | 2    | 1       | 3    |
| DeepSeek-R1 14B  | 0.744               | 9         | 6    | 5        | 5    | 1       | 2    |
| Gemma 2 9B       | 0.688               | 6         | 7    | 6        | 4    | 1       | 4    |
| Mistral-Nemo 12B | 0.688               | 7         | 6    | 5        | 5    | 2       | 3    |
| LLaMA 3.1 8B     | 0.588               | 3         | 4    | 4        | 4    | 5       | 8    |
| LLaMA 3.2 3B     | 0.271               | 0         | 0    | 0        | 0    | 7       | 21   |
| RoBERTa Large    | 0.819               | 10        | 8    | 6        | 2    | 2       | 0    |

RoBERTa large is included as a reference, representing the current best-performing BERT-style baseline model provided by the challenge organizers (<https://grand-challenge.org/algorithms/dragon-roberta-large-domain-specific/>). Unlike our models, this baseline was trained directly on all 28 tasks and is not evaluated in a zero-shot setting.

![Heatmap showing mean model performance across 8 models and 8 task types. The color scale represents the mean score from 0.2 (dark purple) to 0.8 (light green).](c0843c6d138705289960d9f53a6e72a1_img.jpg)

Mean Model Performance By Task Type

|                  | SL Bin | SL Multi | ML Bin | ML Multi | SL Reg | ML Reg | SL NER | ML NER |
|------------------|--------|----------|--------|----------|--------|--------|--------|--------|
| Llama3.3-70B     | 0.82   | 0.74     | 0.94   | 0.75     | 0.98   | 0.96   | 0.21   | 0.31   |
| Phi4-14B         | 0.80   | 0.73     | 0.92   | 0.69     | 0.97   | 0.96   | 0.29   | 0.30   |
| Qwen2.5-14B      | 0.82   | 0.75     | 0.91   | 0.64     | 0.98   | 0.87   | 0.20   | 0.30   |
| Deepseek-R1-14B  | 0.82   | 0.71     | 0.91   | 0.70     | 0.98   | 0.94   | 0.21   | 0.28   |
| Gemma2-9B        | 0.77   | 0.62     | 0.88   | 0.63     | 0.88   | 0.91   | 0.23   | 0.27   |
| Mistral-Nemo-12B | 0.77   | 0.60     | 0.84   | 0.59     | 0.97   | 0.93   | 0.13   | 0.28   |
| Llama3.1-8B      | 0.62   | 0.55     | 0.73   | 0.41     | 0.90   | 0.72   | 0.13   | 0.24   |
| Llama3.2-3B      | 0.50   | 0.14     | 0.50   | 0.08     | 0.17   | 0.28   | 0.11   | 0.14   |
| Roberta-Large    | 0.86   | 0.71     | 0.95   | 0.66     | 0.92   | 0.78   | 0.83   | 0.78   |

Heatmap showing mean model performance across 8 models and 8 task types. The color scale represents the mean score from 0.2 (dark purple) to 0.8 (light green).

**Figure 4.** Heatmap illustrating the average performance of models across various task categories. Each cell shows the mean model score across tasks within a category. Scores range from 0 (worst) to 1 (best), except for multiclass classification tasks evaluated with Cohen's kappa, which ranges from -1 (complete disagreement) to 1 (perfect agreement), with 0 indicating chance. For binary classification, 0.5 reflects chance-level performance. Task types are abbreviated as follows: **SL** = Single-label, **ML** = Multi-label, **Bin** = Binary classification, **Multi** = Multi-class classification, **Reg** = Regression, **NER** = Named Entity Recognition. The colormap represents average performance scores, with exact values annotated in each cell. The evaluation metric varies by task type: AUC is used for binary classification, Cohen's Kappa for multi-class classification, RSMAPES for regression, and F1 score for NER. The performance of the best performing baseline RoBERTa model of the challenge organizers (<https://grand-challenge.org/algorithms/dragon-roberta-large-domain-specific/>) is provided for reference.

![Bar chart showing mean performance scores and standard deviation for 28 tasks across 10 task types. The y-axis is 'Score' from 0.0 to 1.0. The x-axis lists tasks T01 to T28. A legend identifies the task types by color. Dotted red lines indicate the performance of the RoBERTa large model.](c64e9e9f3b0b828a5f6ac70441877764_img.jpg)

Task-Level Mean Scores and Variability for Leading Models

| Task | SL Bin | SL Multi | ML Bin | ML Multi | SL Reg | ML Reg | SL NER | ML NER |
|------|--------|----------|--------|----------|--------|--------|--------|--------|
| T01  | 0.97   |          |        |          |        |        |        |        |
| T02  | 0.82   |          |        |          |        |        |        |        |
| T03  | 0.91   | 0.45     |        |          |        |        |        |        |
| T04  | 0.53   |          |        |          |        |        |        |        |
| T05  | 0.84   |          |        |          |        |        |        |        |
| T06  | 0.70   |          |        |          |        |        |        |        |
| T07  | 0.89   |          |        |          |        |        |        |        |
| T08  | 0.86   |          |        |          |        |        |        |        |
| T09  |        | 0.68     |        |          |        |        |        |        |
| T10  |        | 0.96     |        |          |        |        |        |        |
| T11  |        | 0.61     |        |          |        |        |        |        |
| T12  |        | 0.84     |        |          |        |        |        |        |
| T13  |        | 0.71     |        |          |        |        |        |        |
| T14  |        | 0.64     |        |          |        |        |        |        |
| T15  |        |          | 0.91   |          |        |        |        |        |
| T16  |        |          | 0.92   |          |        |        |        |        |
| T17  |        |          | 0.74   |          |        |        |        |        |
| T18  |        |          | 0.63   |          |        |        |        |        |
| T19  |        |          |        | 0.98     |        |        |        |        |
| T20  |        |          |        | 0.98     |        |        |        |        |
| T21  |        |          |        | 0.98     |        |        |        |        |
| T22  |        |          |        | 0.95     |        |        |        |        |
| T23  |        |          |        | 0.92     |        |        |        |        |
| T24  |        |          |        | 0.91     |        |        |        |        |
| T25  |        |          |        |          | 0.10   |        |        |        |
| T26  |        |          |        |          | 0.36   |        |        |        |
| T27  |        |          |        |          | 0.44   |        |        |        |
| T28  |        |          |        |          | 0.16   |        |        |        |

Bar chart showing mean performance scores and standard deviation for 28 tasks across 10 task types. The y-axis is 'Score' from 0.0 to 1.0. The x-axis lists tasks T01 to T28. A legend identifies the task types by color. Dotted red lines indicate the performance of the RoBERTa large model.

**Figure 5.** Mean performance scores and SD for each of the 28 tasks, computed over the final evaluation metric across the top 4 performing models (Llama-3.3-70B, Phi-4-14B, Qwen-2.5-14B, and DeepSeek-R1-14B). Task types are color-coded and are abbreviated as follows: **SL** = Single-label, **ML** = Multi-label, **Bin** = Binary classification, **Multi** = Multi-class classification, **Reg** = Regression, **NER** = Named Entity Recognition. The mean performance per task of the best performing baseline RoBERTa model of the challenge organizers (<https://grand-challenge.org/algorithms/dragon-roberta-large-domain-specific/>) is provided as dotted red lines for reference.

**Table 2.** Task-wise comparison between LLaMA3.3-70B (<https://grand-challenge.org/algorithms/llm-extractor-llama33/>) and DRAGON RoBERTa large domain-specific (<https://grand-challenge.org/algorithms/dragon-roberta-large-domain-specific/>).

| Task  | LLaMA3.3 | RoBERTa large        |
|-------|----------|----------------------|
| T01   | 0.971    | <b>0.983 ± 0.004</b> |
| T02   | 0.788    | <b>0.958 ± 0.008</b> |
| T03   | 0.923    | <b>0.842 ± 0.096</b> |
| T04   | 0.500    | <b>0.996 ± 0.001</b> |
| T05   | 0.840    | <b>0.944 ± 0.010</b> |
| T06   | 0.708    | <b>0.631 ± 0.042</b> |
| T07   | 0.955    | <b>0.870 ± 0.040</b> |
| T08   | 0.883    | <b>0.640 ± 0.050</b> |
| T09   | 0.619    | <b>0.767 ± 0.039</b> |
| T10   | 0.978    | <b>0.975 ± 0.003</b> |
| T11   | 0.669    | <b>0.861 ± 0.003</b> |
| T12   | 0.842    | <b>0.428 ± 0.057</b> |
| T13   | 0.736    | <b>0.669 ± 0.022</b> |
| T14   | 0.573    | <b>0.577 ± 0.009</b> |
| T15   | 0.917    | <b>0.991 ± 0.010</b> |
| T16   | 0.959    | <b>0.903 ± 0.015</b> |
| T17   | 0.767    | <b>0.639 ± 0.074</b> |
| T18   | 0.732    | <b>0.686 ± 0.015</b> |
| T19   | 0.995    | <b>0.981 ± 0.002</b> |
| T20   | 0.991    | <b>0.974 ± 0.002</b> |
| T21   | 0.993    | <b>0.955 ± 0.012</b> |
| T22   | 0.976    | <b>0.854 ± 0.003</b> |
| T23   | 0.955    | <b>0.818 ± 0.012</b> |
| T24   | 0.961    | <b>0.783 ± 0.003</b> |
| T25   | 0.028    | <b>0.816 ± 0.007</b> |
| T26   | 0.401    | <b>0.835 ± 0.003</b> |
| T27   | 0.467    | <b>0.898 ± 0.010</b> |
| T28   | 0.161    | <b>0.666 ± 0.035</b> |
| Score | 0.760    | <b>0.819 ± 0.021</b> |

RoBERTa scores include SD based on 5-fold cross-validation. A score is bolded when it is higher than the other model's and lies outside of RoBERTa's SD.

### Effect of in-context translation

To assess the impact of in-context translation, we compared model performance on Dutch inputs with and without prior translation to English by the LLMs themselves. To assess the impact of in-context translation, we compared model performance on Dutch inputs with and without prior translation to English by the LLMs themselves. This translation strategy led to statistically significant performance degradation for Phi-4-12B ( $S_{\text{DRAGON}} 0.751 \to 0.533$ ,  $\Delta = -0.22$ ,  $H = 11.56$ ,  $P_{\text{adj}} < .001$ ,  $r = -0.94$ ) and Llama-3.1-8B ( $0.588 \to 0.337$ ,  $\Delta = -0.25$ ,  $H = 11.56$ ,  $P_{\text{adj}} < .001$ ,  $r = -0.94$ ). For Mistral-Nemo-12B, the decrease from 0.688 to 0.573 ( $\Delta = -0.11$ ,  $H = 2.50$ ,  $P_{\text{adj}} = .114$ ,  $r = -0.30$ ) was not statistically significant after adjustment. These findings indicate that in-context translation can markedly impair downstream performance for some models.

Figure 6 details the differences in performance for all models where translation was tested, showing consistent task-level reductions in the relevant performance metric. These results suggest that translation-induced noise undermines clinical information extraction accuracy, underscoring the importance of native-language inference for domain-specific tasks.

## Discussion

In this study, we evaluated the zero-shot performance of 9 widely used open-source LLMs using the `llm_extractorator`

framework on the DRAGON challenge, a Dutch clinical NLP benchmark. Our results highlight both the promise and limitations of deploying such models for real-world information extraction tasks in healthcare.

We found that models with around 14B parameters, including Phi-4, Qwen-2.5, and DeepSeek-R1, performed well across most tasks, achieving average DRAGON utility scores near 0.75. The Llama-3.3-70B model outperformed all others with a utility score of 0.76, consistent with prior findings that larger models tend to generalize better.<sup>39,40</sup> However, this improvement came with significant computational cost and only translated into higher task-level performance in 11 of 28 cases. This suggests that performance gains from scaling are not uniform across task types, and that deploying larger models is most justifiable when computational resources are readily available and the marginal performance gains are considered worthwhile.

Regression tasks, such as extracting lesion sizes or PSA levels, were a relative strength for all tested LLMs. These results contrast with the weaker performance by fine-tuned BERT-style models on the multi-label regression tasks in particular. Generative models appear to handle numeric value reproduction especially well due to their copy-and-reason capabilities. This aligns with prior intuitions that generative models retain quantitative tokens during inference whereas this is more difficult for encoder-based models.<sup>41</sup>

Performance declined markedly on classification tasks and collapsed on NER. Even the strongest models achieved F1 scores below 0.5 on the latter. This underperformance was likely exacerbated by the token-level output format required by the DRAGON challenge. Generative models are not naturally suited for generating sparsely populated token-level lists, and our structured prompting followed by post-processing likely introduced conversion errors. It has been shown in other work that more suitable evaluation formats can yield good performance on similar tasks.<sup>42</sup> As such, our results represent only a conservative estimate of model capabilities on these tasks.

Additionally, certain tasks were inherently unsuited to zero-shot evaluation and were therefore unlikely to succeed. For instance, Task 04 (Skin histopathology case selection) asked models to determine whether a pathology report should be excluded based on vague criteria such as being incomplete or lacking a definitive diagnosis. In the intended use case, where models are trained on labeled examples, such patterns could be learned. However, in a zero-shot setting, where no task-specific feedback or examples are available, the prompt alone provides insufficient guidance. This limitation is reflected in the near-random performance of even the top-performing models on this task.

Annotation quality also influenced model performance. As reported by the DRAGON organizers, some tasks showed relatively low inter-annotator agreement. Notably, Task 06 (histopathology cancer origin) had a Krippendorff's Alpha of 0.333, Task 14 (textual entailment) scored 0.550, Task 17 (PDAC attributes) 0.677, and Task 18 (hip osteoarthritis scoring) 0.557. Inconsistent labeling likely contributed to higher variance across models and limited overall accuracy, even for top-performing systems.

While the DRAGON leaderboard is currently led by fine-tuned encoder models, our zero-shot evaluation of generative models paints a more nuanced picture. Although Llama-3.3 trailed the top-performing RoBERTa-based model overall

![Scatter plot showing the impact of machine translation on LLM performance across 28 clinical NLP tasks. The y-axis represents tasks 01 to 28. The x-axis represents the delta score (with translation - no translation), ranging from -0.7 to 0.1. Three models are compared: Phi4-14B (blue circles), Mistral-Nemo-12B (orange squares), and Llama3.1-8B (green diamonds). Most tasks show negative deltas, indicating that translation generally degrades performance.](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

Effect of Translation on Model Performance

Task

$\Delta$  score (with translation - no translation)

Phi4-14B  
Mistral-Nemo-12B  
Llama3.1-8B

| Task | Phi4-14B ( $\Delta$ score) | Mistral-Nemo-12B ( $\Delta$ score) | Llama3.1-8B ( $\Delta$ score) |
|------|----------------------------|------------------------------------|-------------------------------|
| 01   | -0.05                      | 0.00                               | -0.05                         |
| 02   | -0.15                      | -0.05                              | -0.10                         |
| 03   | -0.25                      | -0.05                              | -0.15                         |
| 04   | -0.05                      | 0.05                               | -0.05                         |
| 05   | -0.15                      | -0.05                              | -0.10                         |
| 06   | -0.10                      | 0.00                               | -0.05                         |
| 07   | -0.20                      | -0.10                              | -0.15                         |
| 08   | -0.15                      | -0.05                              | -0.10                         |
| 09   | -0.25                      | -0.10                              | -0.20                         |
| 10   | -0.10                      | -0.15                              | -0.10                         |
| 11   | -0.20                      | -0.15                              | -0.15                         |
| 12   | -0.30                      | -0.20                              | -0.25                         |
| 13   | -0.40                      | -0.25                              | -0.30                         |
| 14   | -0.50                      | -0.30                              | -0.35                         |
| 15   | -0.60                      | -0.35                              | -0.40                         |
| 16   | -0.45                      | -0.40                              | -0.45                         |
| 17   | -0.55                      | -0.45                              | -0.50                         |
| 18   | -0.65                      | -0.50                              | -0.55                         |
| 19   | -0.50                      | -0.55                              | -0.60                         |
| 20   | -0.60                      | -0.60                              | -0.65                         |
| 21   | -0.70                      | -0.65                              | -0.70                         |
| 22   | -0.55                      | -0.70                              | -0.75                         |
| 23   | -0.65                      | -0.75                              | -0.80                         |
| 24   | -0.75                      | -0.80                              | -0.85                         |
| 25   | -0.60                      | -0.85                              | -0.90                         |
| 26   | -0.70                      | -0.90                              | -0.95                         |
| 27   | -0.75                      | -0.95                              | -1.00                         |
| 28   | -0.80                      | -1.00                              | -1.05                         |

Scatter plot showing the impact of machine translation on LLM performance across 28 clinical NLP tasks. The y-axis represents tasks 01 to 28. The x-axis represents the delta score (with translation - no translation), ranging from -0.7 to 0.1. Three models are compared: Phi4-14B (blue circles), Mistral-Nemo-12B (orange squares), and Llama3.1-8B (green diamonds). Most tasks show negative deltas, indicating that translation generally degrades performance.

**Figure 6.** Impact of machine translation on LLM performance across 28 clinical NLP tasks in the DRAGON challenge. This plot illustrates the performance deltas (with translation—without translation) for Phi-4-14B, Mistral-NeMo-12B, and Llama-3.1-9B. Negative deltas indicate that translation on average degrades model performance across tasks.

(0.760 vs 0.819 utility score), this difference is primarily due to strong relative performance of the RoBERTa model on NER tasks and Task 04. Excluding these tasks shifts the average score in favor of Llama-3.3. Its *SDRAGON* rises to 0.858, while RoBERTa's drops to 0.814. This suggests complementary strengths: encoder models excel at token-level classification, while generative LLMs are better suited to structured inference and regression tasks.

Importantly, these comparisons must be contextualized within the operational and data constraints of real-world deployments. Fine-tuned RoBERTa models require supervised training on labeled data for each task, and are tightly coupled to their respective training distributions. By contrast, Llama-3.3 and other generative LLMs were evaluated strictly in a zero-shot setting, without any parameter updates or task-specific examples in-context. That they perform comparably under these conditions, sometimes even exceeding RoBERTa's performance, suggests that generative models are becoming increasingly viable alternatives for scalable, plug-and-play clinical NLP, especially in settings where labeled data is scarce or task requirements evolve frequently.

Another key insight from our analysis is the significant negative impact of translating Dutch medical texts into English prior to inference. Across the board, naïve translation consistently degraded performance. While the literature presents mixed findings depending on the context, with some studies reporting benefits from pre-translation<sup>43,44</sup> while others observe better outcomes without it,<sup>45</sup> our results suggest that translation on this dataset introduces artifacts and erodes clinical nuance. These findings underscore the need for robust native-language support in clinical NLP tools.

Smaller models, such as Llama-3.2-3B and Gemma-2-2B, consistently failed across tasks, producing nonsensical outputs. This establishes a practical lower bound for model scale for zero-shot clinical NLP in a non-English language. Our *llm\_extractorator* framework supports efficient inference of larger models on consumer-grade GPUs. This effectively eliminates the need to rely on underpowered models. While smaller models may offer marginal improvements in inference speed, the trade-off in output quality is steep: the risk of generating entirely unusable results outweighs any computational performance gains.

This study has several limitations. First, a uniform prompting approach was used across tasks and models, without extensive task-specific engineering. While this supports reproducibility, it likely underestimates achievable performance. Second, our evaluation was limited to zero-shot settings in Dutch. Generalizability to other languages remains to be tested and there is room for future research to explore the effects of few-shot prompting,<sup>46</sup> lightweight instruction-tuning,<sup>47</sup> or retrieval-augmented generation<sup>48</sup> on model performance. Third, due to resource constraints, we only evaluated one model over 15B parameters and did not include any of the largest open-source LLMs. Future work should explore their capabilities, especially in high-resource settings. Finally, since all evaluations were handled through the Grand Challenge platform without access to per-example outputs, we were unable to perform detailed error analysis. This restricts insights into where and why models may have failed.

In summary, this work demonstrates that open-source generative LLMs can serve as powerful tools for medical information extraction in Dutch. With minimal infrastructure or

labeled data, several models approach or surpass fine-tuned encoder baselines in clinical NLP tasks. By streamlining this process through our *llm\_extractorator* framework, we lower the barrier to applying these models in real-world clinical research.

## Author contributions

Luc Builtjes (Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Visualization, Writing—original draft), Joeran Bosma (Data curation, Resources, Writing—original draft), Mathias Prokop (Supervision), Bram van Ginneken (Supervision, Writing—original draft), and Alessa Hering (Conceptualization, Supervision, Writing—original draft)

## Supplementary material

*Supplementary material* is available at JAMIA Open online.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

## Conflicts of interest

The authors have no competing interests to declare.

## Data availability

This study is based on data from the DRAGON challenge hosted on the Grand Challenge platform (<https://dragon.grand-challenge.org/>). The DRAGON challenge is a Type 3 challenge, meaning that neither the training nor test datasets are accessible to participants. Instead, algorithms are developed and submitted by participants to be executed on hidden datasets maintained by the organizers. As such, the data are not available for download or direct inspection. Researchers can participate in the challenge and evaluate their methods by submitting their algorithms through the challenge platform.

## References

1. Dash S, Shakyawar SK, Sharma M, et al. Big data in healthcare: management, analysis and future prospects. *J Big Data*. 2019;6:25.
2. Hosny A, Parmar C, Quackenbush J, et al. Artificial intelligence in radiology. *Nat Rev Cancer*. 2018;18:500-510.
3. Meystre SM, Savova GK, Kipper-Schuler KC, et al. Extracting information from textual documents in the electronic health record: a review of recent research. *Yearb Med Inform*. 2008;17:128-144.
4. Adnan K, Akbar R. An analytical study of information extraction from unstructured and multidimensional big data. *J Big Data*. 2019;6:38.
5. Devlin J. Bert: pre-training of deep bidirectional transformers for language understanding. 2018. <https://arxiv.org/abs/1810.04805>, preprint: not peer reviewed.
6. Rasmy I, Xiang Y, Xie Z, et al. Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. *NPJ Digit Med*. 2021;4:86.

7. Berg HT, van Bakel B, van de Wouw L, et al. ChatGPT and generating a differential diagnosis early in an emergency department presentation. *Ann Emerg Med.* 2024;83:83-86.
8. Kanjee Z, Crowe B, Rodman A. Accuracy of a generative artificial intelligence model in a complex diagnostic challenge. *JAMA.* 2023;330:78-80.
9. Eriksen AV, Möller S, Ryg J. Use of GPT-4 to diagnose complex clinical cases. *NEJM AI.* 2024;1:Alp2300031.
10. Fast D, Adams LC, Busch F, et al. Autonomous medical evaluation for guideline adherence of large language models. *NPJ Digit Med.* 2024;7:358.
11. Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. *Nat Med.* 2024;30:1134-1142.
12. Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge. *Nature.* 2023;620:172-180.
13. Gilson, Aidan, Safranek, Conrad W, Huang, Thomas, et al. How does ChatGPT perform on the United States medical licensing examination (USMLE)? the implications of large language models for medical education and knowledge assessment. *JMIR Med Educ.* 2023;9:e45312.
14. Shi F, Suzgun M, Freitag M, et al. Language models are multilingual chain-of-thought reasoners. 2022. <https://arxiv.org/abs/2210.03057>, preprint: not peer reviewed.
15. Li J, Dada A, Puladi B, et al. ChatGPT in healthcare: a taxonomy and systematic review. *Comput Methods Programs Biomed.* 2024;245:108013.
16. Goh E, Gallo RJ, Strong E, et al. GPT-4 assistance for improvement of physician performance on patient care tasks: a randomized controlled trial. *Nat Med.* 2025;31:1233-1238.
17. Mitsuyama Y, Tatekawa H, Takita H, et al. Comparative analysis of GPT-4-based ChatGPT's diagnostic performance with radiologists using real-world radiology reports of brain tumors. *Eur Radiol.* 2025;35:1938-1947.
18. Li KD, Fernandez AM, Schwartz R, et al. Comparing GPT-4 and human researchers in health care data analysis: qualitative description study. *J Med Internet Res.* 2024;26:e56500.
19. Park J, Oh K, Han K, et al. Patient-centered radiology reports with generative artificial intelligence: adding value to radiology reporting. *Sci Rep.* 2024;14:13218.
20. Noda R, Tanabe K, Ichikawa D, et al. GPT-4's performance in supporting physician decision-making in nephrology multiple-choice questions. *Sci Rep.* 2025;15:15439.
21. Achiam J, Adler S, Agarwal S, et al. Gpt-4 technical report. 2023. <https://arxiv.org/abs/2303.08774>, preprint: not peer reviewed.
22. Balloccu S, Schmidtová P, Lango M, et al. Leak, cheat, repeat: data contamination and evaluation malpractices in closed-source LLMs. 2024. <https://arxiv.org/abs/2402.03927>, preprint: not peer reviewed.
23. Kukreja S, Kumar T, Purohit A, et al. A literature survey on open source large language models. *Proceedings of the 2024 7th International Conference on Computers in Management and Business.* Association for Computing Machinery (ACM); 2024:133-143.
24. Hasan MA, Taranum P, Dey K, et al. Do large language models speak all languages equally? A comparative study in low-resource settings. 2024. <https://arxiv.org/abs/2408.02237>, preprint: not peer reviewed.
25. Vanroy B. Language resources for Dutch large language modelling. 2023. <https://arxiv.org/abs/2312.12852>, preprint: not peer reviewed.
26. Gangavarapu A. Introducing l2m3, a multilingual medical large language model to advance health equity in low-resource regions. 2024. <https://arxiv.org/abs/2404.08705>, preprint: not peer reviewed.
27. Wassie AK, Molaei M, Moslem Y. Domain-specific translation with open-source large language models: resource-oriented analysis. 2024. <https://arxiv.org/abs/2412.05862>, preprint: not peer reviewed.
28. Bosma JS, Dercksen K, Builtjes L, et al. The DRAGON benchmark for clinical NLP. *NPJ Digit Med.* 2025;8:289.
29. Mistral AI Team. Mistral NeMo. Accessed May 2025. <https://mistral.ai/news/mistral-nemo/>
30. Grattafiori A, Dubey A, Jauhri A, et al. The llama 3 herd of models. 2024. <https://arxiv.org/abs/2407.21783>, preprint: not peer reviewed.
31. Gemma Team. Gemma 2: improving open language models at a practical size. 2024. <https://arxiv.org/abs/2408.00118>, preprint: not peer reviewed.
32. Abdin M, Aneja J, Behl H, et al. Phi-4 technical report. 2024. <https://arxiv.org/abs/2412.08905>, preprint: not peer reviewed.
33. Yang A, Yang B, Zhang B, et al. Qwen2.5 technical report. 2024. <https://arxiv.org/abs/2412.15115>, preprint: not peer reviewed.
34. Guo D, Yang D, Zhang H, et al. Deepseek-r1: incentivizing reasoning capability in LLMs via reinforcement learning. 2025. <https://arxiv.org/abs/2501.12948>, preprint: not peer reviewed.
35. LangChain: the platform for reliable agents. Accessed May 2025. <https://www.langchain.com/>
36. Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits reasoning in large language models. *Adv Neural Inf Process Syst.* 2022;35:24824-24837.
37. Ollama. Accessed May 2025. <https://ollama.com>
38. Meakin J, Gerke P, Kerksstra S, et al. Grand-Challenge.org (v2024.11). 2024. Accessed May 2025. [10.5281/zenodo.14040002](https://zenodo.org/record/14040002)
39. Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models. 2020. <https://arxiv.org/abs/2001.08361>, preprint: not peer reviewed.
40. Ahuja S, Aggarwal D, Gumma V, et al. Megaverse: benchmarking large language models across languages, modalities, models and tasks. 2023. <https://arxiv.org/abs/2311.07463>, preprint: not peer reviewed.
41. Wallace E, Wang Y, Li S, et al. Do NLP models know numbers? probing numeracy in embeddings; 2019. <https://arxiv.org/abs/1909.07940>, preprint: not peer reviewed.
42. Wiest IC, Lefmann M-E, Wolf F, et al. Deidentifying medical documents with local, privacy-preserving large language models: the LLM-Anonymizer. *NEJM AI.* 2025;2:Aidbp2400537.
43. Mondshine I, Paz-Argaman T, Tsarfaty R. Beyond English: the impact of prompt translation strategies across languages and tasks in multilingual LLMs. 2025. <https://arxiv.org/abs/2502.09331>, preprint: not peer reviewed.
44. Chen Y, Shah V, Ritter A. Translation and fusion improves zero-shot cross-lingual information extraction. 2023. <https://arxiv.org/abs/2305.13582>, preprint: not peer reviewed.
45. Intrator Y, Halfon M, Goldenberg R, et al. Breaking the language barrier: can direct inference outperform pre-translation in multilingual LLM applications?. 2024. <https://arxiv.org/abs/2403.04792>, preprint: not peer reviewed.
46. Brown T, Mann B, Ryder N, et al. Language models are few-shot learners. *Adv Neural Inf Process Syst.* 2020;33:1877-1901.
47. Wei J, Bosma M, Zhao VY, et al. Finetuned language models are zero-shot learners. 2021. <https://arxiv.org/abs/2109.01652>, preprint: not peer reviewed.
48. Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *Adv Neural Inf Process Syst.* 2020;33:9459-9474.

© The Author(s) 2025. Published by Oxford University Press on behalf of the American Medical Informatics Association.

This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License (<https://creativecommons.org/licenses/by-nc/4.0/>), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact [reprints@oup.com](mailto:reprints@oup.com) for reprints and translation rights for reprints. All other permissions can be obtained through our RightsLink service via the Permissions link on the article page on our site—for further information please contact [journals.permissions@oup.com](mailto:journals.permissions@oup.com).

JAMIA Open, 2025, 8, 1–11

<https://doi.org/10.1093/jamiaopen/oaf109>

Research and Applications