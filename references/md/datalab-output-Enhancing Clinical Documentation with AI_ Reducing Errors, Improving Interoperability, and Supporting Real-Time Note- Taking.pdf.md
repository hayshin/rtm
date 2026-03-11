

![InfoScience Trends logo featuring a stylized head with a brain and an 'i' inside a circle.](2dfa6ac3edfe874f68aa0cbccaa42322_img.jpg)

InfoScience Trends logo featuring a stylized head with a brain and an 'i' inside a circle.

![InfoPub logo featuring a stylized 'P' inside a square.](64662465bba247703fdec49c8f3309f9_img.jpg)

InfoPub logo featuring a stylized 'P' inside a square.

## Review Article

# Enhancing Clinical Documentation with AI: Reducing Errors, Improving Interoperability, and Supporting Real-Time Note-Taking

Saeed Saadat <sup>1\*</sup> ![ORCID icon](e3f8612927870f2e0f9f5989e6dd3064_img.jpg), Majid Khalilizad Darounkolaei <sup>2</sup> ![ORCID icon](a86c7d1c9cb81c81614634a31267440d_img.jpg), Mohsen Qorbani <sup>3</sup> ![ORCID icon](ce158fc5e55633398941d0898ae45661_img.jpg), Atefe Hemmat <sup>4</sup> ![ORCID icon](6f77f2588732dff582d5f470675e762f_img.jpg), Sadaf Hariri <sup>5</sup> ![ORCID icon](802fbc25d869d680d37bfef9949fa598_img.jpg)

**Submitted:** 02 Dec. 2024; **Accepted:** 12 Jan. 2025; **Published:** 14 Jan. 2025

![Check for updates button with a red circular arrow icon.](6ed175c791b5e156d9c98a8dbcc3318c_img.jpg)

Check for updates button with a red circular arrow icon.

## Abstract

The increasing administrative burden in healthcare, particularly in clinical documentation, has driven research into artificial intelligence (AI)-powered solutions to enhance transcription accuracy, improve interoperability in electronic health record (EHR) systems, and enable real-time clinical note generation. This study systematically reviewed 14 relevant studies to evaluate the role of AI technologies, including natural language processing (NLP) and automatic speech recognition (ASR), in addressing these challenges. Results showed that AI significantly reduces transcription errors when combining ASR with domain-specific NLP models, such as ClinicalBERT, and fine-tuned large language models (LLMs) like GPT-4, by improving context understanding and terminology accuracy. Real-time clinical note generation was commonly achieved using hybrid extractive-abstractive summarization techniques and structured templates, such as SOAP (Subjective, Objective, Assessment, Plan) notes, with enhanced usability and time savings demonstrated in clinical settings. Additionally, systems employing semantic knowledge graphs and ontologies (e.g., UMLS) facilitated greater standardization and interoperability between disparate EHR systems. However, critical challenges were noted with hallucination risks in text generation, data privacy concerns, and low clinician trust in automated tools. Evaluation metrics such as ROUGE, BERTScore, and domain-specific measures (e.g., DeepScore) revealed variability in the quality and factual consistency of AI-generated notes. This review highlights the potential of AI to alleviate documentation burdens, though further advances in real-time integration, accuracy, and user acceptability are required for widespread adoption in healthcare environments.

**Keywords:** Clinical Documentation, Artificial Intelligence, Electronic Health Record, Large Language Models, Natural Language Processing.

---

1\*. Corresponding author; Department of Epidemiology, University of North Carolina at Charlotte, 9201, University City Boulevard, Charlotte, NC, 28223, USA. Email: [ssaadat@charlotte.edu](mailto:ssaadat@charlotte.edu); 2. Department of Orthopedics, Clinical Research Center of Shahid Beheshti Hospital, Babol, Iran.; 3. Department of Radiology, Faculty of Medicine, Shahid Beheshti University of Medical Sciences, Tehran, Iran.; 4. Department of Biology, Faculty of Sciences, Central Tehran Branch, Islamic Azad University, Tehran, Iran.; 5. Research committee, Faculty of Medicine, Urmia university of medical sciences, Urmia, Iran.; / Open Access. © 2025 the author(s), published by InfoPub. This work is licensed under the Creative Commons Attribution 4.0 International License. (Journal homepage: <https://www.isitrend.com>) <https://doi.org/10.61186/ist.202502.01.01>

https://doi.org/10.61186/ist.202502.01.01 || InfoPub"}

ongoing efforts to reshape medical documentation practices by integrating cutting-edge AI technologies.

## Method

The research employs a Systematic Literature Review (SLR) approach, following the methodology presented in this document [24]. It outlines the SLR process through six distinct phases: Main Questions, Databases and Search Strategy, Criteria for Inclusion and Exclusion, and Data Extraction.

### Main Research Question

1. How can AI, including natural language processing (NLP), reduce transcription errors in medical documentation?
2. How can AI improve interoperability between healthcare systems?
3. How can AI support real-time clinical note-taking to enhance documentation workflows for healthcare providers?

### Databases and Search Strategy

The research for this study on the impact of artificial intelligence (AI) in clinical documentation utilized several key databases to ensure a comprehensive review of the literature. The databases accessed included PubMed, Google Scholar, IEEE Xplore, Scopus, Web of Science, and the ACM Digital Library. These platforms were selected for their extensive collections of peer-reviewed articles and conference proceedings relevant to healthcare technology and medical informatics.

To facilitate an effective search, a carefully curated set of keywords was employed. The keywords included phrases such as "Clinical documentation," "AI scribe" OR "Digital scribe," "Automated medical documentation," "Electronic medical records" OR "EHR," "Transcription error reduction," "Speech recognition" OR "ASR," "Natural language processing" OR "NLP," "Interoperability" AND "health systems," and "Real-time note-taking." This diverse range of terms was designed to capture various aspects of AI applications in clinical settings, ensuring that both broad and specific topics were addressed (See [supplementary file 1](#)).

The search strategy followed a systematic, stepwise approach. Initially, broad combinations of keywords were used alongside Boolean operators (AND, OR) to identify a wide array of relevant literature. The search was limited to articles published between 2004 and 2024 to encompass both foundational studies and recent advancements in the field. This time frame allowed for a thorough exploration of how AI technologies have evolved over the past two decades and their impact on clinical workflows.

### Inclusion Criteria

Use of AI/NLP: Papers must explicitly utilize AI or NLP methods in:

- Clinical documentation or transcription.
- Improving EHR interoperability.
- Real-time or near real-time clinical note-taking.

System Descriptions: Articles should describe systems that:

- Reduce transcription errors.
- Structure clinical notes (e.g., SOAP format).
- Implement EHR-compatible workflows.

![InfoPub logo featuring a stylized head with a question mark inside.](4c7edbe6e90384d6698dcc59ef3aa0cd_img.jpg)

InfoPub logo featuring a stylized head with a question mark inside.

*Evaluation Focus:* Selected studies must focus on evaluated or benchmarked systems, avoiding purely theoretical concepts.

*Healthcare-Specific Datasets:* Articles must mention the use of healthcare-specific datasets, such as MIMIC-III, CliniNote, or AMI dataset.

*Publication Standards:* Papers should be published in reputable journals, conferences, or ArXiv if the content is deemed high-impact.

### Exclusion Criteria

*Non-Healthcare NLP Use Cases:* Papers focusing on NLP applications outside of healthcare, such as patient sentiment analysis, will be excluded.

*General AI/ML Papers:* Articles that discuss AI/ML without specific ties to clinical documentation will not be included.

*Retrospective-Only Systems:* Studies that focus solely on retrospective transcription or summarization systems lacking real-time applicability will be excluded.

*Irrelevant Topics:* Articles related to diagnostic AI, population health, or unrelated EHR analyses (e.g., billing, fraud detection) will not be considered.

### Data Extraction

The data extraction will focus on capturing general information (e.g. study objectives, methods, and datasets used) and goal-specific details for each study. For transcription error reduction, information on methods such as Automatic Speech Recognition (ASR), fine-tuned domain-specific models, and metrics like Word Error Rate (WER) or error categorization will be extracted. Interoperability data will include techniques for terminology standardization (e.g., SNOMED-CT, ICD), ontology-based mapping tools, transformer models (e.g., BioBERT), as well as measured outcomes like improved cross-platform data compatibility. Real-time enhancements will be evaluated by capturing system features like live transcription, workflow integration with Electronic Health Records (EHRs), and metrics on time savings or usability. In addition, supplementary information on datasets, system usability, error types (e.g., omissions, hallucinations), and ethical considerations will be collected to provide a comprehensive understanding of how AI systems are being applied.

## Results

### Article Selection

Based on our comprehensive search strategy across multiple databases, we initially identified a total of 4,969 articles. After removing 2,363 duplicate entries, we were left with 2,606 unique articles to screen. Upon review, all 2,606 articles were excluded as they did not align with our study objectives. This exclusion included 1,845 articles from various academic disciplines and 330 articles published in languages other than English.

As a result of this filtering process, we identified 431 articles that appeared to be potentially eligible for further consideration. However, a subsequent review of the titles and abstracts led to the exclusion of an additional 263 articles that did not meet our study design criteria, particularly those addressing unrelated topics. We then proceeded to evaluate the full texts of the remaining 168 articles for inclusion in our study. After a thorough examination, we excluded another 154 articles, which brought us to a final selection of 14 articles that were included in the literature review and evidence synthesis (Figure 1).

![InfoPub logo featuring a stylized human head profile with a brain icon inside.](4a00fa6a0859c67e717c83baab9dd87f_img.jpg)

InfoPub logo featuring a stylized human head profile with a brain icon inside.

![PRISMA flow diagram for a systematic review. The process starts with Identification: 4969 records from Scopus, IEEE Xplore, PubMed, ACM Digital Library, Web of Science, and Google Scholar. Duplicates are removed (2363), leaving 2606. Screening: 2175 records are excluded for other disciplines (1845) and other languages (330), leaving 431. Eligibility: 263 are excluded in title selection and 154 in full-text selection, leaving 168. Included: 14 studies are selected.](690fce4fb5c9cbb8beb560cb2a3fcbeb_img.jpg)

```

graph TD
    subgraph Identification
        A[Database searches identified the following records (n= 4969):  
Scopus (n= 1632), IEEE Xplore (n= 1245), PubMed (n= 1145), ACM Digital Library (n= 360), Web of Science (n= 450), Google Scholar (n= 137).]
    end
    A --> B[Records duplicates removed (n= 2363)]
    B --> C[Records duplicates after removed (n= 2606)]
    C --> D[Records excluded (n= 2175)]
    C --> E[Records remaining after title/ abstract selection (n=431)]
    D --> F[Other disciplines: (n= 1845)  
Other languages (n=330)]
    E --> G[Records remaining after Full-text selection (n= 168)]
    E --> H[Excluded, in the title selection  
Other study design: 263]
    G --> I[Studies selected (n=14)]
    G --> J[Excluded, in the Full-text selection  
Other study design: 154]
    style A fill:#fff,stroke:#333
    style B fill:#fff,stroke:#333
    style C fill:#fff,stroke:#333
    style D fill:#fff,stroke:#333
    style E fill:#fff,stroke:#333
    style F fill:#fff,stroke:#333
    style G fill:#fff,stroke:#333
    style H fill:#fff,stroke:#333
    style I fill:#fff,stroke:#333
    style J fill:#fff,stroke:#333
    style Identification fill:#d9edf7,stroke:#337ab7
    style Screening fill:#d9edf7,stroke:#337ab7
    style Eligibility fill:#d9edf7,stroke:#337ab7
    style Included fill:#d9edf7,stroke:#337ab7
  
```

PRISMA flow diagram for a systematic review. The process starts with Identification: 4969 records from Scopus, IEEE Xplore, PubMed, ACM Digital Library, Web of Science, and Google Scholar. Duplicates are removed (2363), leaving 2606. Screening: 2175 records are excluded for other disciplines (1845) and other languages (330), leaving 431. Eligibility: 263 are excluded in title selection and 154 in full-text selection, leaving 168. Included: 14 studies are selected.

**Figure 1.** The PRISMA flow diagram for systematic review.

### Data Extraction Findings

The data extraction process was conducted across 14 studies and focused on identifying findings relevant to transcription error reduction, interoperability improvements, and real-time clinical note generation using AI. The detailed results of this data extraction are provided in [Supplementary File 2](#), where findings are organized in tabular format to facilitate a clear summary of the evidence from each study.

#### RQ1: How can AI reduce transcription errors in medical documentation?

The results about the first research question from studies [25,26,27, 28, 16, 29], which include a review of the Method/Technology and findings on reducing transcription errors, are thoroughly detailed in Table 1.

Table 1: How can AI reduce transcription errors in medical documentation?

| Reference | Method/Technology                                                                  | Findings on Reducing Transcription Errors                                                                                         |
|-----------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [25]      | Early framework combining ASR (automatic speech recognition) with extraction tools | Demonstrated feasibility of real-time accurate transcription of clinical dialogues but faced challenges with accents and noise.   |
| [26]      | MediNotes (ASR + LLMs + Retrieval-Augmented Generation)                            | Significantly improved transcription fidelity by combining domain-specific NLP models with ASR, reducing errors in medical terms. |

|      |                                                                         |                                                                                                                                     |
|------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| [27] | Multi-accent adaptive ASR system integrated with EMRs                   | Enhanced accuracy of transcription through accent adaptation and fault tolerance, outperforming traditional ASR systems.            |
| [28] | Comparative study of SR (speech recognition) and keyboard transcription | Found ASR had higher transcription error rates than manual documentation, particularly in complex tasks (errors increased by 331%). |
| [16] | Integration of cTAKES with ASR in a speech transcriber plugin           | Able to automatically annotate medical terms for improved transcription but required post-editing for corrections.                  |
| [29] | AutoScribe system for real-time dialogue parsing                        | Used context prediction to identify and reduce irrelevant information, improving task-focused accuracy.                             |

#### RQ2: How can AI improve interoperability between healthcare systems?

The findings related to the second research question from studies [5, 13, 16, 27, 6, 26,] encompassing a review of the AI techniques and insights on interoperability improvements, are extensively presented in Table 2.

**Table 2:** How can AI improve interoperability between healthcare systems?

| Reference | AI Technique                                           | Findings on Interoperability Improvements                                                                                      |
|-----------|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [5]       | Care2Report system using semantic knowledge graphs     | Improved terminology standardization across systems by aligning ASR transcription outputs with formal ontologies (e.g., UMLS). |
| [13]      | Speaker diarization + medical speech recognition       | Produced standardized sections of clinical notes, facilitating integration with EHR systems.                                   |
| [16]      | Use of cTAKES for clinical annotation                  | Enabled semantic standardization for integration into EHRs but lacked direct cross-platform compatibility features.            |
| [27]      | Multi-accent adaptive ASR + linguistic models          | Integrated transcription results into structured EHR formats with a focus on adaptation to regional/national standards.        |
| [6]       | Domain-specific LLMs with in-house AI systems          | Suggests using in-house AI systems (as opposed to external models) to enhance data security and ensure compatibility.          |
| [26]      | MediNotes (ASR + NLP + Retrieval-Augmented Generation) | Supports integration with healthcare providers' existing EHR workflows for better continuity of data sharing across systems.   |

#### RQ3: How can AI support real-time clinical note generation?

The findings concerning the third research question from studies [26, 30, 11, 31, 7, 25, 32], which include a review of the Model/System and insights on real-time clinical note generation, are thoroughly detailed in Table 3.

**Table 3:** How can AI support real-time clinical note generation?

| Reference | Model/System                                              | Findings on Real-Time Clinical Note Generation                                                                                        |
|-----------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| [26]      | MediNotes                                                 | Generated SOAP notes in real time or near-real time from both live and pre-recorded conversations with high usability scores.         |
| [30]      | K-SOAP note format + CliniKnote dataset                   | Proposed pipeline enhanced readability and quick information retrieval, suitable for live updates and retrospective edits.            |
| [11]      | Cluster2Sent (hybrid summarization approach)              | Extracted and clustered critical utterances during consultations to structure notes immediately; improved accuracy for real-time use. |
| [31]      | ChatGPT-4 applied to real-time conversations              | Generated SOAP notes in real time but flagged critical issues with omissions and hallucinations, raising safety concerns.             |
| [7]       | Dutch digital scribe system                               | Demonstrated notable time savings and quality improvements during real-time use with post-editing for accuracy.                       |
| [25]      | Early real-time editable encounter note system            | Provided live transcription and displayed evolving notes during patient visits for clinician review.                                  |
| [32]      | Template-based summarization controlled by semantic rules | High-quality clinical notes in real time using rule-based templates, though integration with EHR systems was limited.                 |

## Discussion

### Main Summary

AI systems employing advanced NLP models, speech recognition technologies, and extractive-abstractive hybrid frameworks (e.g., [11, 26, 32]) are highly effective at automating clinical note generation and reducing transcription errors but face challenges in real-time integration with EHR systems and ensuring interoperability.

Systems like MediNotes [26] and Cluster2Sent [11] effectively generate accurate SOAP notes using a hybrid extractive-abstractive summarization approach combined with speech recognition. Template-based methods [32] and keyword-enhanced formats like K-SOAP [30] enable structured, clinician-friendly outputs with reduced errors and omissions.

Tools such as AutoScribe [17, 29] and digital scribe prototypes [7, 13] focus on real-time audio transcription and note generation integrated into light EHR workflows.

Real-world usability studies [7, 22] emphasize the importance of editable drafts and high physician satisfaction during live use.

Modern ASR systems incorporate domain-specific acoustic models, multi-accent adaptation, and NLP-enhanced processing [27] to improve transcription accuracy for complex clinical terminology.

Integration with semantic interpretation tools like cTAKES [16] and knowledge graphs [5] enhances the fidelity of extracted information.

Fine-tuned LLMs like GPT variants and LLaMA [12, 14] show promise in generating draft notes but face challenges of hallucinations, omissions, and high computational cost.

Retrieval-augmented generation (RAG) and parameter-efficient tuning further optimize LLM performance for constrained clinical environments [12].

New metrics (e.g., knowledge-graph embeddings, DeepScore) specifically assess medical note-generation quality in terms of factual correctness, hallucination rate, and clinical relevance [19, 20].

Gold-standard manual annotation and live editing remain key for validating automated notes [19, 31].

Few studies explicitly tackle the standardization of terms or semantic mapping for EHR interoperability, despite its importance [5, 16].

Major issues involve hallucinations and critical omissions in generative models [14, 19, 31]. Clinician trust in AI tools depends largely on generating editable drafts and human oversight [7, 21].

Real-time systems [17] struggle with balancing seamless integration into physician workflows while ensuring output quality doesn't introduce cognitive load or safety concerns [7, 21].

AI systems for clinical documentation are advancing rapidly, with hybrid NLP-ASR pipelines and fine-tuned LLMs achieving notable successes in reducing transcription errors and supporting real-time clinical note-taking. However, challenges in ensuring reliability, interoperability, and clinician trust highlight the need for further refinement and careful deployment. Comprehensive approaches that combine usability testing, tailored datasets (e.g., [30]), and domain-specific evaluation metrics [19, 20] are critical for widespread adoption.

### Reducing Transcription Errors with AI and NLP

AI and NLP-driven systems have shown significant promise in reducing transcription errors by addressing the limitations of manual documentation and traditional transcription methods.

![InfoPub logo featuring a stylized head with a brain and a gear.](7c805c7712b39895742750555067b283_img.jpg)

InfoPub logo featuring a stylized head with a brain and a gear.

Many studies emphasize the integration of automatic speech recognition (ASR) technology with domain-specific language models to achieve higher accuracy and contextual fidelity in transcriptions. For instance, Stephen Wenceslao and M.R. Estuar (2019) demonstrated the utility of ASR combined with cTAKES for medical annotation, producing real-time clinical encounter drafts while maintaining consistency in recognizing medical terminology [16]. Similarly, Xin Xia et al. (2022) developed an ASR framework with multi-accent adaptation and fault-tolerance mechanisms for use in online medical record systems, noting major improvements in transcription accuracy, especially in scenarios with diverse speaker accents [27].

Advanced language models specifically trained on medical datasets, such as ClinicalBERT or LLaMA fine-tuned models, have further enhanced transcription performance. Hui Yi Leong et al. (2024) achieved notable reductions in errors through MediNotes, a generative AI framework combining domain-specific training with techniques like Quantized Low-Rank Adaptation (QLoRA) [26]. This innovation enables efficient error correction while managing resource constraints. Other approaches, such as the modular summarization framework proposed by Kundan Krishna et al. (2020), focus on extracting relevant utterances for transcription tasks, highlighting that hybrid extractive-abstractive models significantly outperform end-to-end methods in transcription accuracy by reducing extraneous or irrelevant information [11].

Despite these advances, challenges remain. Tobias Hodgson et al. (2017) revealed that while ASR systems can improve efficiency, they may still introduce user-related errors such as omissions or misclassifications, especially in complex or interrupted tasks [28]. Therefore, there is a consensus that pairing ASR outputs with error-detection mechanisms, as explored by Asma Ben Abacha and colleagues (2023), where metrics like factual correctness and hallucination rates are calculated, is critical to achieving reliable transcription quality [19].

### Enhancing Interoperability in Healthcare Systems

Interoperability between healthcare systems remains a critical challenge, with AI solutions often focusing on integration of ontologies and semantic standardization to ensure compatibility across Electronic Health Record (EHR) platforms. Jeffrey Klann and Peter Szolovits (2009) highlighted early efforts to integrate ASR systems with tools for clinical meaning extraction, paving the way for structured data that could be interpreted across systems [25]. More recent work by Faiza Khan Khattak et al. (2019) with AutoScribe explored context-driven parsing of clinical conversations to produce semantically normalized outputs that align with medical vocabularies, such as SNOMED-CT and ICD codes [29].

Semantic embeddings and ontology-driven NLP models have seen notable advancements in this domain. Binh-Nguyen Nguyen et al. (2023) demonstrated how semantic partitioning could improve the alignment of concept clusters with SOAP note subsections, allowing downstream models to generate interoperable documentation summaries [32]. Similarly, modern transformer-based models, such as those described by Khalid Nawab (2024), dynamically map ambiguous terms (e.g., "BP" as blood pressure or bipolar disorder) using contextual embeddings, reducing the risk of misinterpretation when exchanging records between institutions [6].

Excitingly, integrated implementations have also been introduced, such as the Care2Report system by L. Maas et al. (2020), which employs knowledge graphs to ensure semantic accuracy and compatibility with widely used EHR standards like HL7 FHIR [5]. However, challenges persist. Current studies lack consensus on scalable solutions to address ambiguous or overlapping terminologies, and practical adoption of interoperable frameworks remains limited. For example, while Annessa Kernberg et al. (2023) evaluated ChatGPT-4 for its ability to generate structured SOAP

![InfoPub logo featuring a stylized head with a brain and a gear.](4dbddc2a60ea4ad6037f1f360ae0fea3_img.jpg)

InfoPub logo featuring a stylized head with a brain and a gear.

notes, comprehensive integration with cross-platform terminology frameworks was not explored [31]. Thus, the integration of dynamic ontologies with AI tools continues to require further research and broader adoption.

### Supporting Real-Time Clinical Note-Taking with AI

Supporting real-time clinical note-taking has been a focus of numerous AI initiatives, emphasizing both workflow efficiency and clinician usability. Early implementations, such as those proposed by Greg P. Finley et al. (2018), designed systems incorporating ASR, speaker diarization, and real-time knowledge extraction to alleviate documentation burdens during live patient-provider interactions [33]. The natural evolution of these systems can be seen in modern frameworks like MediGen, introduced by Hui Yi Leong et al. (2024), which leverages Parameter-Efficient Fine-Tuning (PEFT) on LLaMA-based models to balance real-time transcription accuracy with minimal computational demand [12].

Another commonly adopted framework is the SOAP note structure, which many real-time systems aim to replicate through structured summarization. For instance, Bin Han et al. (2023) used T5-large fine-tuning to classify and align transcriptions with SOAP sections in real-time [34]. Duy-Cat Can et al. (2023) further refined this process, introducing semantic partitioning that allowed for the generation of subsection-specific summaries in real-time, reducing physician cognitive load while improving the coherence of outputs [32].

A key feature of many modern systems is the integration of editable draft workflows. Both the study by Brenna Li et al. (2021), which involved "Wizard of Oz" simulations with digitally generated notes [22], and the usability study by Marieke M. van Buchem et al. (2024) with Dutch scribe systems, demonstrated that clinicians value the ability to immediately review and modify AI-generated outputs [7]. Such workflows merge automation with human oversight, building trust and mitigating risks associated with AI hallucinations or inaccuracies.

Despite progress, practical limitations of real-time systems, such as transcription lag or usability concerns, remain a challenge. Tom Knoll et al. (2022) stressed that real-time note generation must fit seamlessly within established workflows to avoid hindering clinician-patient communication [21]. Additionally, ethical and regulatory considerations, highlighted by Anjanava Biswas and Wrick Talukdar (2024), such as maintaining patient privacy under HIPAA, ensuring AI-generated notes follow clinical guidelines, and handling model biases, are paramount to the success of such systems [14].

The reviewed literature underscores the transformative potential of AI in medical documentation. NLP-enabled transcription systems, such as those proposed by Stephen Wenceslao and M.R. Estuar et al. (2019) [16] and Hui Yi Leong et al. (2024) [12], significantly reduce errors, while semantic standardization frameworks like Care2Report by Maas et al. (2020) highlight the importance of interoperability [5]. Furthermore, real-time note-taking tools increasingly favor human-AI collaboration, as demonstrated by Brenna Li et al. (2021) [22] and Bin Han et al. (2023) [34], to enhance documentation workflows. Still, unresolved issues related to system integration, trust, and regulation necessitate continued attention. Future research must focus on scalable, secure, and ethical solutions to achieve widespread clinical adoption.

### Limitations

There are several limitations to consider. The exclusion of gray literature is a significant issue, as it may have resulted in publication bias by ignoring impactful but unpublished or non-peer-reviewed work, such as conference proceedings or technical reports.

The review's scope appears heavily weighted toward transcription and SOAP note generation, with less attention given to interoperability solutions, such as AI-driven terminology standardization or cross-platform compatibility in healthcare systems, creating potential evidence gaps.

The included studies show significant heterogeneity in design, ranging from proof-of-concept evaluations to usability studies and prototype development. This diversity makes it difficult to compare findings systematically or draw quantitative conclusions, especially as different evaluation metrics (e.g., ROUGE, BERTScore, PDQI-9) and datasets are used across studies. Furthermore, many studies rely on simulations, benchmark datasets, or mock consultations rather than long-term clinical deployments, limiting insights into real-world usability or adoption challenges. There is also a noticeable lack of attention to ethical issues, including data privacy, patient confidentiality, and algorithmic biases, which are crucial in sensitive healthcare contexts.

Another limitation is the lack of formal risk-of-bias assessments for the included studies. Questions remain about the representativeness of datasets, external validation of performance metrics, and the mitigation of confounding variables, such as how AI tools integrate into diverse real-world healthcare workflows.

The review also underrepresents multilingual or multi-accent scenarios, which are increasingly relevant in global healthcare systems. Despite some mentions of ASR model personalization, the studies offer limited evaluations of how these systems handle non-English language data, accents, or regional medical terminology. Together, these limitations highlight the need for more rigorous, transparent, and inclusive systematic reviews in the future.

## Conclusion

The purpose of reviewing these 14 studies was to examine the application of AI, particularly natural language processing (NLP), in addressing challenges related to transcription errors, interoperability between healthcare systems, and real-time clinical note-taking in medical documentation. This area of research is critically important as the administrative burdens of clinical documentation are a significant contributor to physician burnout and inefficiencies in patient care. AI-based solutions, including transcription systems, summarization tools, and real-time documentation assistants, have emerged as promising technologies to alleviate these burdens by improving accuracy, standardizing data for easier sharing across platforms, and integrating seamlessly into clinicians' workflows.

Our review revealed substantial progress in AI's capacity to automate clinical documentation and reduce transcription errors, particularly through hybrid approaches that combine automatic speech recognition (ASR) with NLP for generating structured formats like SOAP notes. Additionally, the utilization of large language models (LLMs) and task-specific fine-tuning demonstrates promise in creating accurate and contextually relevant medical notes. However, the review also uncovered notable gaps, particularly in the area of interoperability, as few studies directly tackled the challenges of cross-platform data sharing or terminology standardization. Similarly, critical issues like ethical deployment, data privacy, algorithmic bias, and real-world usability remain underexplored in many studies.

We undertook this review to consolidate knowledge in a fragmented and rapidly evolving research space and to identify both the accomplishments and the limitations of current AI-driven medical documentation systems. The findings underscore the potential of these technologies to transform healthcare by reducing clinicians' administrative burden and improving documentation quality, but they also highlight the need for further research to address practical, ethical, and

technical challenges. Ultimately, this review aims to inform future research efforts, guide the development of more robust and trustworthy AI solutions, and support their integration into clinical workflows in a way that benefits both clinicians and patients.

## Abbreviations

**AI**: Artificial Intelligence; **EHR**: Electronic Health Record; **NLP**: Natural Language Processing; **ASR**: Automatic Speech Recognition; **LLMs**: Large Language Models; **SOAP**: Subjective, Objective, Assessment, Plan; **SLR**: Systematic Literature Review; **WER**: Word Error Rate; **RAG**: Retrieval-Augmented Generation; **QLoRA**: Quantized Low-Rank Adaptation; **PEFT**: Parameter-Efficient Fine-Tuning.

## Ethical approval

Ethics approval for conducting this systematic review was not required. No participants were involved in this research.

## Availability of data and materials

The data used and analyzed during the current study is available in supplementary files [1](#) and [2](#).

## Funding

This research was not funded or supported by any organizations.

## Authors' Contribution

Conceptualization, S.S.; Methodology, M.KD.; Resources, M.Q.; Writing—original draft preparation, A.H.; Writing—review and editing, S.H., and S.S.; Supervision, S.S.; Project administration, M.KD.; Search databases, M.Q., A.H., and S.S.; Study Selection, M.KD., and S.S.; Quality Assessment, S.S., and S.H.; Data Extraction, S.H. and M.KD., and S.S. All authors have read and agreed to the published version of the manuscript.

## Acknowledgment

Not applicable.

## Consent for publication

The authors provided their consent for the publication of the study results.

### Competing interests

The authors declare no competing interests.

## References

- [1]. Demsash AW, Kassie SY, Dubale AT, Chereka AA, Ngusie HS, Hunde MK, et al. Health professionals' routine practice documentation and its associated factors in a resource-limited setting: a cross-sectional study. *BMJ health & care informatics*. 2023;30(1). <https://doi.org/10.1136/bmjhci-2022-100699>
- [2]. Karajizadeh M, Nikandish R, Zalpour Z, Roozrokh Arshadi Montazer M, Soleimanijafarbiglo M, Mazaher Y, et al. Identification of the Information Needs of a Nurse-led Rapid Response Team to Design and Develop an Electronic Medical Record System. *Health Management & Information Science*. 2022;9(4):236-42. <https://doi.org/10.30476/hmsi.2023.96214.1143>
- [3]. Quinn M, Forman J, Harrod M, Winter S, Fowler KE, Krein SL, et al. Electronic health records, communication, and data sharing: challenges and opportunities for improving the diagnostic process. *Diagnosis* (Berlin, Germany). 2019;6(3):241-8. <https://doi.org/10.1515/dx-2018-0036>
- [4]. Karajizadeh M, Nikandish R, Yousefianzadeh O, Hamedi Z, Saeidnia HR. Usability Evaluation of the Electronic Medical Record of the Rapid Response Team: a Case Study. *Applied Health Information Technology*. 2021. <https://doi.org/10.18502/ahit.v2i2.7991>

![InfoPub logo: a stylized head profile with a question mark inside, symbolizing inquiry and knowledge.](a44b1f8627b760ca21aa18b125cd472f_img.jpg)

InfoPub logo: a stylized head profile with a question mark inside, symbolizing inquiry and knowledge.

- [5]. Maas L, Geurtsen M, Nouwt F, Schouten SF, Van De Water R, Van Dulmen S, et al., editors. [The Care2Report System: Automated Medical Reporting as an Integrated Solution to Reduce Administrative Burden in Healthcare](#). HICSS; 2020: 3608-17.
- [6]. Nawab K. Artificial intelligence scribe: A new era in medical documentation. Artificial Intelligence in Health. 2024;1(4):12-5. <https://doi.org/10.36922/aih.3103>
- [7]. van Buchem MM, Kant IM, King L, Kazmaier J, Steyerberg EW, Bauer MP. Impact of a digital scribe system on clinical documentation time and quality: usability study. JMIR AI. 2024;3(1):e60020. <https://doi.org/10.2196/60020>
- [8]. Maleki Varnosfaderani S, Forouzanfar M. The Role of AI in Hospitals and Clinics: Transforming Healthcare in the 21st Century. Bioengineering (Basel, Switzerland). 2024;11(4). <https://doi.org/10.3390/bioengineering11040337>
- [9]. Davenport T, Kalakota R. The potential for artificial intelligence in healthcare. Future healthcare journal. 2019;6(2):94-8. <https://doi.org/10.7861/futurehosp.6-2-94>
- [10]. Kreimeyer K, Foster M, Pandey A, Arya N, Halford G, Jones SF, et al. Natural language processing systems for capturing and standardizing unstructured clinical information: A systematic review. Journal of Biomedical Informatics. 2017;73:14-29. <https://doi.org/10.1016/j.jbi.2017.07.012>
- [11]. Krishna K, Khosla S, Bigham JP, Lipton ZC. Generating SOAP notes from doctor-patient conversations using modular summarization techniques. arXiv preprint arXiv:200501795. 2020. <https://doi.org/10.48550/arXiv.2005.01795>
- [12]. Leong HY, Gao YF, Shuai J, Zhang Y, Pamuksuz U. Efficient fine-tuning of large language models for automated medical documentation. arXiv preprint arXiv:240909324. 2024. <https://doi.org/10.48550/arXiv.2409.09324>
- [13]. Finley G, Edwards E, Robinson A, Brenndoerfer M, Sadoughi N, Fone J, et al., editors. An automated medical scribe for documenting clinical encounters. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations; 2018. <https://doi.org/10.18653/v1/N18-5003>
- [14]. Biswas A, Talukdar W. Intelligent Clinical Documentation: Harnessing Generative AI for Patient-Centric Clinical Note Generation. arXiv preprint arXiv:240518346. 2024. <https://doi.org/10.48550/arXiv.2405.18346>
- [15]. Reisman M. EHRs: [The Challenge of Making Electronic Data Usable and Interoperable. P & T : a peer-reviewed journal for formulary management](#). 2017;42(9):572-5.
- [16]. Wenceslao SJMC, Estuar MRJE, editors. Using cTAKES to build a simple speech transcriber plugin for an EMR. Proceedings of the 3rd International Conference on Medical and Health Informatics; 2019. <https://doi.org/10.1145/3340037.3340044>
- [17]. Crampton NH, editor Ambient virtual scribes: Mutuo Health's AutoScribe as a case study of artificial intelligence-based technology. Healthcare management forum; 2020: SAGE Publications Sage CA: Los Angeles, CA. <https://doi.org/10.1177/0840470419872775>
- [18]. Quiroz JC, Laranjo L, Kocaballi AB, Berkovsky S, Rezazadegan D, Coiera E. Challenges of developing a digital scribe to reduce clinical documentation burden. NPJ digital medicine. 2019;2(1):114. <https://doi.org/10.1038/s41746-019-0190-1>
- [19]. Abacha AB, Yim W-w, Michalopoulos G, Lin T. An investigation of evaluation metrics for automated medical note generation. arXiv preprint arXiv:230517364. 2023. <https://doi.org/10.48550/arXiv.2305.17364>
- [20]. Oleson J. DeepScore: A Comprehensive Approach to Measuring Quality in AI-Generated Clinical Documentation. arXiv preprint arXiv:240916307. 2024. <https://doi.org/10.48550/arXiv.2409.16307>
- [21]. Knoll T, Moramarco F, Korfiatis AP, Young R, Ruffini C, Perera M, et al. User-driven research of medical note generation software. arXiv preprint arXiv:220502549. 2022. <https://doi.org/10.48550/arXiv.2205.02549>
- [22]. Li B, Crampton N, Yeates T, Xia Y, Tian X, Truong K, editors. Automating clinical documentation with digital scribes: Understanding the impact on physicians. Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems; 2021. <https://doi.org/10.1145/3411764.3445172>
- [23]. Saeidnia HR, Hashemi Fotami SG, Lund B, Ghiasi N. Ethical considerations in artificial intelligence interventions for mental health and well-being: Ensuring responsible implementation and impact. Social Sciences. 2024;13(7):381. <https://doi.org/10.3390/socsci13070381>
- [24]. Kitchenham B, Brereton OP, Budgen D, Turner M, Bailey J, Linkman S. Systematic literature reviews in software engineering—a systematic literature review. Information and software technology. 2009;51(1):7-15. <https://doi.org/10.1016/j.infsof.2008.09.009>
- [25]. Klann JG, Szolovits P. An intelligent listening framework for capturing encounter notes from a doctor-patient dialog. BMC medical informatics and decision making. 2009;9:1-10. <https://doi.org/10.1186/1472-6947-9-S1-S3>
- [26]. Leong HY, Gao Y, Ji S. A gen ai framework for medical note generation. In2024 6th International Conference on Artificial Intelligence and Computer Applications (ICAICA) 2024 Nov 28 (pp. 423-429). IEEE. <https://doi.org/10.1109/ICAICA63239.2024.10823004>
- [27]. Xia X, Ma Y, Luo Y, Lu J. An online intelligent electronic medical record system via speech recognition. International Journal of Distributed Sensor Networks. 2022;18(11):15501329221134479. <https://doi.org/10.1177/15501329221134479>
- [28]. Hodgson T, Magrabi F, Coiera E. Efficiency and safety of speech recognition for documentation in the electronic health record. Journal of the American Medical Informatics Association. 2017;24(6):1127-33. <https://doi.org/10.1093/jamia/ocx073>

![InfoPub logo: a stylized human head profile with a brain-like pattern inside.](10953d657a5f47fdc829a800419dd370_img.jpg)

InfoPub logo: a stylized human head profile with a brain-like pattern inside.

- [29]. Khattak FK, Jeblee S, Crampton N, Mamdani M, Rudzicz F. AutoScribe: extracting clinically pertinent information from patient-clinician dialogues. MEDINFO 2019: Health and Wellbeing e-Networks for All: IOS Press; 2019. p. 1512-3. <https://doi.org/10.3233/shti190510>
- [30]. Li Y, Wu S, Smith C, Lo T, Liu B. Improving Clinical Note Generation from Complex Doctor-Patient Conversation. arXiv preprint arXiv:240814568. 2024. <https://doi.org/10.48550/arXiv.2408.14568>
- [31]. Kernberg A, Gold JA, Mohan V. Using ChatGPT-4 to Create Structured Medical Notes From Audio Recordings of Physician-Patient Encounters: Comparative Study. Journal of Medical Internet Research. 2024;26:e54419. <https://doi.org/10.2196/54419>
- [32]. Can DC, Nguyen QA, Nguyen BN, Nguyen MQ, Nguyen KV, Do TH, Le HQ. [UETCorn at MEDIQA-Sum 2023: Template-based Summarization for Clinical Note Generation from Doctor-Patient Conversation. InCLEF \(Working Notes\) 2023 \(pp. 1423-1432\).](#)
- [33]. Finley GP, Edwards E, Robinson A, Sadoughi N, Fone J, Miller M, et al., editors. [An Automated Assistant for Medical Scribes. INTERSPEECH; 2018. 3212-13.](#)
- [34]. Han B, Zhu H, Zhou S, Ahmed S, Rahman MM, Xia F, Lybarger K. [HuskyScribe at MEDIQA-Sum 2023: Summarizing Clinical Dialogues with Transformers. InCLEF \(Working Notes\) 2023 \(pp. 1488-1509\).](#)

## Publisher's Note

**Disclaimer:** This article has been reviewed only by the journal's editors. The opinions expressed in this article are those of the author(s) and do not necessarily reflect the views or opinions of the journal's editorial board.

© 2025 The Author(s). Published by InfoPub.  
Publisher homepage: <https://infopub.info/>

![InfoPub logo featuring a stylized head with a brain and an eye.](b8205e5e617a8946ddc956c816156fec_img.jpg)

InfoPub logo featuring a stylized head with a brain and an eye.