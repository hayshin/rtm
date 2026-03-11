

# Semantic NLP Pipelines for Interoperable Patient Digital Twins from Unstructured EHRs

Rafael Brens   Yuqiao Meng   Luoxi Tang   Zhaohan Xi  
Binghamton University

## Abstract

Digital twins—virtual replicas of physical entities—are gaining traction in healthcare for personalized monitoring, predictive modeling, and clinical decision support. However, generating interoperable patient digital twins from unstructured electronic health records (EHRs) remains challenging due to variability in clinical documentation and lack of standardized mappings. This paper presents a semantic NLP-driven pipeline that transforms free-text EHR notes into FHIR-compliant digital twin representations. The pipeline leverages named entity recognition (NER) to extract clinical concepts, concept normalization to map entities to SNOMED-CT or ICD-10, and relation extraction to capture structured associations between conditions, medications, and observations. Evaluation on MIMIC-IV Clinical Database Demo with validation against MIMIC-IV-on-FHIR reference mappings demonstrates high F1-scores for entity and relation extraction, with improved schema completeness and interoperability compared to baseline methods.

## 1 Introduction

Digital twins—computational models that represent patients as dynamic, semantically structured entities—are emerging as key enablers of personalized and interoperable healthcare systems (Grieves and Vickers, 2017; Björnsson et al., 2020). Recent work has demonstrated their utility in cardiology (Corral-Acero et al., 2020), chronic disease management (Voigt et al., 2021), and precision medicine more broadly (Laubenbacher et al., 2024). However, building patient digital twins from clinical data remains challenging due to the heavy reliance on unstructured electronic health records (EHRs) such as physician notes and discharge summaries (Wang et al., 2018). These free-text narratives contain rich clinical information but lack the standardization necessary for interoperability

across institutions and clinical decision support systems.

To address this challenge, international standards such as HL7 Fast Healthcare Interoperability Resources (FHIR) define modular schemas for representing patient conditions, observations, medications, and encounters in a unified format (Mandel et al., 2016; Bender and Sartipi, 2013). A FHIR-compliant representation requires standardized terminology alignment using ontologies such as SNOMED-CT (Donnelly, 2006), ICD-10, LOINC (McDonald et al., 2003), and RxNorm (Nelson et al., 2011), as well as structured resource construction following FHIR JSON properties. These properties allow independently developed systems to consume patient data consistently.

In this work, we explore how semantic Natural Language Processing (NLP) can automatically map unstructured EHR text to FHIR-compliant digital twin structures. Our objective is twofold: first, to characterize the specific FHIR properties required to build a minimal interoperable patient digital twin, and second, to design an NLP pipeline that extracts and normalizes clinical information so it can be assembled into these FHIR resources. By aligning the pipeline explicitly with FHIR’s schema requirements, we aim to improve both interoperability and semantic completeness in the resulting digital twin representations.

To bridge this gap, we propose a semantic NLP pipeline that automatically transforms unstructured clinical narratives into FHIR-compliant digital twin representations. Our approach combines transformer-based named entity recognition with ontology-grounded concept normalization and relation extraction, enabling end-to-end conversion without manual intervention. Unlike rule-based methods that require extensive pattern engineering, our pipeline learns to generalize across diverse clinical documentation styles.

Experimental evaluation on MIMIC-IV Demo,

validated against MIMIC-IV-on-FHIR reference mappings, demonstrates that our pipeline achieves 0.89 NER F1 and 0.81 relation extraction F1, representing 17-point and 26-point improvements over rule-based baselines respectively. The resulting digital twins achieve 91% semantic completeness and 0.88 interoperability score, validating that semantic NLP can produce clinically meaningful, standards-compliant patient representations.

Our main contributions are: (1) a semantic NLP pipeline that automatically transforms unstructured EHR text into FHIR-compliant patient digital twins; (2) evaluation on MIMIC-IV Demo with validation against reference FHIR mappings demonstrating improvements in extraction accuracy and interoperability; and (3) ablation studies revealing the contribution of each pipeline component to overall performance.

## 2 Related Work

**Digital Twins in Healthcare.** Digital twins are virtual representations of individual patients that integrate multimodal clinical data to simulate physiological states and predict outcomes (Grieves and Vickers, 2017; Corral-Acero et al., 2020). Prior work has demonstrated their utility in ICU monitoring and personalized therapeutics (Voigt et al., 2021; Laubenbacher et al., 2024). However, most implementations rely on structured data, limiting applicability across heterogeneous EHR systems (Wornow et al., 2023).

**Clinical NLP.** Extracting structured information from unstructured EHR text requires robust NLP techniques. Named Entity Recognition (NER) extracts medical concepts such as diseases, medications, labs, and vitals. Transformer-based models including BERT (Devlin et al., 2019) and domain-specific variants such as ClinicalBERT (Alsentzer et al., 2019) and BioBERT (Lee et al., 2020) have achieved state-of-the-art performance on clinical NER benchmarks (Peng et al., 2019; Si et al., 2019). Specialized toolkits like ScispaCy (Neumann et al., 2019) provide robust pipelines for biomedical text processing.

Concept normalization maps extracted entities to standard ontologies such as UMLS (Bodenreider, 2004) and SNOMED-CT (Donnelly, 2006) to enable interoperability. Recent neural approaches including SapBERT (Liu et al., 2021) and synonym marginalization methods (Sung et al., 2020) have improved linking accuracy over traditional systems

like MetaMap (Aronson and Lang, 2010). Relation extraction identifies associations between entities, such as "Drug treats Condition" or "Observation indicates Condition," which is critical for constructing meaningful digital twins (Wei et al., 2020; Zhang et al., 2018; Luo et al., 2022).

**FHIR and Interoperability.** FHIR defines standardized JSON/XML schemas for EHR exchange, facilitating interoperability across systems (Mandel et al., 2016; Bender and Sartipi, 2013). Mapping free-text EHRs to FHIR resources is non-trivial due to semantic ambiguity and incomplete documentation. Previous studies either use manual mapping or rule-based heuristics, which are not scalable. Despite advances in NLP and digital twin modeling, there is a lack of end-to-end semantic pipelines that process unstructured clinical text, map entities to standard ontologies, and produce FHIR-compliant digital twins with high semantic completeness. Our work addresses this gap.

## 3 Methodology

Our methodology transforms unstructured clinical narratives into FHIR-compliant patient digital twins through three main steps: defining a minimal FHIR profile, applying semantic NLP for entity extraction and normalization, and performing relation-aware resource assembly with validation. Figure 1 provides an overview of the pipeline architecture.

**FHIR Digital Twin Profile.** We define a minimal FHIR R4 profile in which a patient digital twin is represented as a set of core resources—Condition, Observation, and MedicationRequest—linked to a single Patient resource. Each Condition resource must contain a code (populated with SNOMED-CT or ICD-10 identifiers), clinicalStatus, and verificationStatus. Each Observation resource requires a code (using LOINC or SNOMED-CT), a value, and an effectiveDateTime. Each MedicationRequest must contain a medicationCodeableConcept coded in RxNorm and at least one dosageInstruction. The profile enforces structural constraints: all resources reference the same patient, and observations and medications are temporally anchored.

**Semantic NLP for Extraction and Normalization.** Given the FHIR profile, we apply transformer-based clinical NER models to physician notes and discharge summaries. The NER

![Figure 1: Overview of the semantic NLP pipeline. The pipeline starts with a 'Clinical Note' box containing the text: 'Patient has diabetes. Started Metformin 500mg twice daily.' This text flows into five sequential processing boxes: 'NER Module', 'Concept Normalize', 'Relation Extract', and 'FHIR Assembly'. Below each of these boxes is a corresponding output box. The 'NER Module' output is '[diabetes] [Metformin] [500mg 2x]'. The 'Concept Normalize' output is 'diabetes→ SNOMED:73211009 Metformin→ RxNorm:6809'. The 'Relation Extract' output is 'has-dosage (Metformin, 500mg 2x/day)'. The 'FHIR Assembly' output is 'Condition: code: 73211009 MedRequest: med: 6809'. The final output is a 'Digital Twin' box containing a JSON object: {\](b230b8f21d8e82d55c0d311c8c32ef73_img.jpg)

Figure 1: Overview of the semantic NLP pipeline. The pipeline starts with a 'Clinical Note' box containing the text: 'Patient has diabetes. Started Metformin 500mg twice daily.' This text flows into five sequential processing boxes: 'NER Module', 'Concept Normalize', 'Relation Extract', and 'FHIR Assembly'. Below each of these boxes is a corresponding output box. The 'NER Module' output is '[diabetes] [Metformin] [500mg 2x]'. The 'Concept Normalize' output is 'diabetes→ SNOMED:73211009 Metformin→ RxNorm:6809'. The 'Relation Extract' output is 'has-dosage (Metformin, 500mg 2x/day)'. The 'FHIR Assembly' output is 'Condition: code: 73211009 MedRequest: med: 6809'. The final output is a 'Digital Twin' box containing a JSON object: {\

Figure 1: Overview of the semantic NLP pipeline with illustrative example. Given clinical text (left), the pipeline extracts entities via NER, normalizes them to standard ontologies (SNOMED-CT, RxNorm), identifies relations between entities, and assembles validated FHIR resources into a patient digital twin (right).

model uses sequence tagging to detect spans corresponding to conditions, medications, observations, and temporal expressions. We then normalize these entities to controlled vocabularies required by FHIR: condition mentions are mapped to SNOMED-CT and ICD-10 codes using UMLS-based candidate generation and contextual similarity (Bodenreider, 2004; Liu et al., 2021); medication mentions are normalized to RxNorm (Nelson et al., 2011); and observation mentions are mapped to LOINC or SNOMED-CT (McDonald et al., 2003).

**Relation-Aware Assembly and Validation.** Entities and codes alone are insufficient for meaningful FHIR resources, so we recover how they are related in the narrative. We apply relation extraction models to identify links such as *symptom-of*, *has-dosage*, and *has-result* (Wei et al., 2020). Using these relations, we assemble entities into candidate FHIR resources consistent with the profile. We then validate the assembled resources against the FHIR v4.0.1 specification, checking that required fields are present and code systems are correctly declared. Resources that pass validation are bundled together to form the final digital twin.

## 4 Experiments

### 4.1 Setup and Main Results

**Datasets.** We use MIMIC-IV Clinical Database Demo (v2.2) (Johnson et al., 2023b), containing 100 de-identified patients. Since the demo excludes free-text notes, we construct clinical narratives from structured diagnosis, medication, and laboratory data to simulate discharge summaries. For validation, we leverage MIMIC-IV-on-FHIR (Johnson et al., 2023a), which provides reference FHIR resource mappings for the same patient cohort, enabling direct comparison of our automated pipeline

| Method        | NER         | RE          | Comp.      | Interop.    |
|---------------|-------------|-------------|------------|-------------|
| Rule-Based    | 0.72        | 0.55        | 62%        | 0.61        |
| Naive Mapping | 0.68        | 0.50        | 58%        | 0.58        |
| <b>Ours</b>   | <b>0.89</b> | <b>0.81</b> | <b>91%</b> | <b>0.88</b> |

Table 1: Performance comparison on MIMIC-IV Demo validated against MIMIC-IV-on-FHIR reference mappings. NER/RE: F1-scores; Comp.: Semantic Completeness (%); Interop.: Interoperability Score.

against expert-curated FHIR representations. We use a 70%/15%/15% train/validation/test split.

**Baselines.** We compare against two baselines: (1) Rule-Based Extraction using pattern matching with regular expressions and dictionary lookup, and (2) Naive Mapping with direct mapping of entity mentions to FHIR fields without normalization or relation extraction.

**Metrics.** We evaluate using NER F1-score for entity extraction accuracy, Relation Extraction F1-score for correct identification of entity relationships, Semantic Completeness as the ratio of correctly populated fields in our generated FHIR resources compared to MIMIC-IV-on-FHIR reference resources, and Interoperability Score as structural and semantic similarity between our automated FHIR output and the MIMIC-IV-on-FHIR reference mappings.

Table 1 presents our quantitative results. The semantic NLP pipeline significantly outperforms rule-based and naive baselines across all metrics, demonstrating improved extraction accuracy, schema completeness, and interoperability. The 17-point improvement in NER F1 and 26-point improvement in relation extraction F1 over rule-based methods highlight the effectiveness of transformer-based models for clinical text understanding.

| Configuration     | NER  | RE   | Comp. | Interop. |
|-------------------|------|------|-------|----------|
| Full Pipeline     | 0.89 | 0.81 | 91%   | 0.88     |
| w/o Normalization | 0.89 | 0.78 | 72%   | 0.65     |
| w/o Relation Ext. | 0.89 | —    | 68%   | 0.71     |
| w/o Validation    | 0.89 | 0.81 | 85%   | 0.79     |
| Base BERT         | 0.81 | 0.72 | 83%   | 0.80     |

Table 2: Ablation study results showing the impact of removing each pipeline component.

| Input Text                                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| “65-year-old male with history of hypertension and type 2 diabetes. BP 145/92. Started Lisinopril 10mg daily.”                                                                          |
| Extracted Entities                                                                                                                                                                      |
| Condition: hypertension → SNOMED:38341003<br>Condition: type 2 diabetes → SNOMED:44054006<br>Observation: BP 145/92 → LOINC:85354-9<br>Medication: Lisinopril 10mg daily → RxNorm:29046 |
| Relations                                                                                                                                                                               |
| has-dosage(Lisinopril, 10mg daily)                                                                                                                                                      |

Table 3: Case study: pipeline processing of a clinical note showing entity extraction, normalization, and relation identification.

### 4.2 Ablation Study

To understand the contribution of each pipeline component, we conduct ablation experiments by systematically removing modules. Table 2 presents the results.

Removing concept normalization causes the largest drop in interoperability (0.88 to 0.65), confirming that ontology alignment is critical for FHIR compliance. Without relation extraction, semantic completeness drops from 91% to 68%, as the pipeline cannot properly associate medications with dosages or observations with values. Clinical pre-training (ClinicalBERT vs. base BERT) provides an 8-point NER improvement, validating the importance of domain-specific language models.

### 4.3 Case Study

Table 3 illustrates the pipeline on a representative clinical note. The input describes a patient with multiple conditions and medications. Our pipeline correctly extracts all entities, normalizes them to appropriate ontologies, and identifies the dosage relation linking Lisinopril to its prescribed regimen.

The case demonstrates end-to-end transformation from unstructured narrative to FHIR-compliant resources with appropriate terminology codes, enabling interoperability with downstream clinical systems.

## 5 Conclusion

We present a semantic NLP pipeline to construct FHIR-compliant patient digital twins from unstructured EHRs. By integrating NER, concept normalization, and relation extraction aligned with FHIR schema requirements, our approach significantly improves semantic completeness and interoperability over baseline methods. Future work will extend the pipeline to longitudinal patient modeling, incorporate multi-modal data including labs and imaging, and evaluate cross-institution generalization.

## Limitations

Our work has several limitations. First, the pipeline was evaluated on the MIMIC-IV Demo dataset containing 100 patients; evaluation on larger patient cohorts is needed to confirm generalizability. Second, while MIMIC-IV-on-FHIR provides reference FHIR mappings, these were generated semi-automatically and may contain mapping inconsistencies. Third, our current approach does not model temporal dependencies across multiple encounters. Fourth, evaluation was limited to English clinical text from a single US healthcare institution. Finally, our evaluation used synthetic clinical narratives constructed from structured MIMIC-IV data; validation on authentic physician-authored text would strengthen generalizability claims.

## Acknowledgments

We acknowledge the use of AI assistants (Claude, GPT) for manuscript drafting, editing, and code development. All outputs were reviewed and validated by the authors, who take full responsibility for the content.

## References

- Emily Alsentzer, John Murphy, William Boag, Wei-Hung Weng, Di Jindi, Tristan Naumann, and Matthew McDermott. 2019. Publicly available clinical BERT embeddings. In *Proceedings of the 2nd Clinical Natural Language Processing Workshop*, pages 72–78.
- Alan R Aronson and François-Michel Lang. 2010. An overview of MetaMap: Historical perspective and recent advances. *Journal of the American Medical Informatics Association*, 17(3):229–236.
- Duane Bender and Kamran Sartipi. 2013. HL7 FHIR: An agile and RESTful approach to healthcare information exchange. *Proceedings of CBMS*, pages 326–331.

- Björn Björnsson, Carl Borrebaeck, Nils Elander, and 1 others. 2020. Digital twins to personalize medicine. *Genome Medicine*, 12(1):1–4.
- Olivier Bodenreider. 2004. The unified medical language system (UMLS): Integrating biomedical terminology. *Nucleic Acids Research*, 32(suppl\_1):D267–D270.
- Jorge Corral-Acero, Francesca Margara, Maciej Marez, and 1 others. 2020. The digital twin to enable the vision of precision cardiology. *European Heart Journal*, 41(48):4556–4564.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT*, pages 4171–4186.
- Kevin Donnelly. 2006. SNOMED-CT: The advanced terminology and coding system for ehealth. *Studies in Health Technology and Informatics*, 121:279–290.
- Michael Grieves and John Vickers. 2017. Digital twin: Mitigating unpredictable, undesirable emergent behavior in complex systems. *Transdisciplinary perspectives on complex systems*, pages 85–113.
- Alistair E. W. Johnson, Lucas Bulgarelli, and Tom J. Pollard. 2023a. [MIMIC-IV clinical database demo on FHIR](#). PhysioNet.
- Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, and 1 others. 2023b. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1):1.
- Reinhard Laubenbacher, Anna Niarakis, Tomas Helikar, and 1 others. 2024. Building digital twins of the human immune system: toward a roadmap. *NPJ Digital Medicine*, 7(1):1–11.
- Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4):1234–1240.
- Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. 2021. Self-alignment pretraining for biomedical entity representations. In *Proceedings of NAACL-HLT*, pages 4228–4238.
- Ling Luo, Po-Ting Lai, Chih-Hsuan Wei, Cecilia N Arighi, and Zhiyong Lu. 2022. BioRED: A rich biomedical relation extraction dataset. *Briefings in Bioinformatics*, 23(5):bbac282.
- Joshua C Mandel, David A Kreda, Kenneth D Mandl, Isaac S Kohane, and Rachel B Ramoni. 2016. SMART on FHIR: A standards-based, interoperable apps platform for electronic health records. *Journal of the American Medical Informatics Association*, 23(5):899–908.
- Clement J McDonald, Stanley M Huff, Jeffrey G Suico, Gilbert Hill, Dennis Leavelle, Raymond Aller, Arden Forrey, Kathy Mercer, Georges DeMoor, John Hook, and 1 others. 2003. LOINC, a universal standard for identifying laboratory observations: A 5-year update. *Clinical Chemistry*, 49(4):624–633.
- Stuart J Nelson, Kelly Zeng, John Kilbourne, Tammy Powell, and Robin Moore. 2011. Normalized names for clinical drugs: RxNorm at 6 years. *Journal of the American Medical Informatics Association*, 18(4):441–448.
- Mark Neumann, Daniel King, Iz Beltagy, and Waleed Ammar. 2019. ScispaCy: Fast and robust models for biomedical natural language processing. In *Proceedings of the 18th BioNLP Workshop and Shared Task*, pages 319–327.
- Yifan Peng, Shankai Yan, and Zhiyong Lu. 2019. Transfer learning in biomedical natural language processing: An evaluation of BERT and ELMo on ten benchmarking datasets. In *Proceedings of the 18th BioNLP Workshop and Shared Task*, pages 58–65.
- Yuqi Si, Jingqi Wang, Hua Xu, and Kirk Roberts. 2019. Enhancing clinical concept extraction with contextual embeddings. In *Journal of the American Medical Informatics Association*, volume 26, pages 1297–1304. Oxford University Press.
- Mujeen Sung, Hwisang Jeon, Jinyuk Lee, and Jaewoo Kang. 2020. Biomedical entity representations with synonym marginalization. In *Proceedings of ACL*, pages 3641–3650.
- Isabel Voigt, Hernan Inojosa, Anja Dillenseger, Rocco Haase, Katja Akgun, and Tjalf Ziemssen. 2021. Digital twins for multiple sclerosis. *Frontiers in Immunology*, 12:669811.
- Yanshan Wang, Liwei Wang, Majid Rastegar-Mojarad, Sungrim Moon, Feichen Shen, Naveed Afzal, Sijia Liu, Yuqun Zeng, Saeed Mehrabi, Sunghwan Sohn, and 1 others. 2018. Clinical information extraction applications: A literature review. *Journal of Biomedical Informatics*, 77:34–49.
- Qiang Wei, Zongcheng Ji, Zhiheng Li, Jingcheng Du, Jingqi Wang, Jun Xu, Yang Xiang, Firat Tiber, and Hua Xu. 2020. A study of deep learning approaches for medication and adverse drug event extraction from clinical text. *Journal of the American Medical Informatics Association*, 27(1):13–21.
- Michael Wornow, Yizhe Xu, Rahul Thapa, Birju Patel, Ethan Steinberg, Scott Fleming, Michael A Pfeffer, Jason Fries, and Nigam H Shah. 2023. The shaky foundations of large language models and foundation models for electronic health records. *NPJ Digital Medicine*, 6(1):135.
- Yuhao Zhang, Peng Qi, and Christopher D Manning. 2018. Graph convolution over pruned dependency trees improves relation extraction. In *Proceedings of EMNLP*, pages 2205–2215.

## A Use of Artifacts

**Data Licenses.** MIMIC-IV Clinical Database Demo is released under the PhysioNet Credentialed Health Data License 1.5.0 for research purposes. MIMIC-IV-on-FHIR is similarly licensed. Both datasets are de-identified in accordance with HIPAA Safe Harbor provisions.

**Artifact Use Consistent With Intended Use.** Our use of MIMIC-IV Demo and MIMIC-IV-on-FHIR is consistent with their intended research purpose for developing clinical informatics methods.

## B Implementation Details

**Computational Resources.** Our pipeline uses pattern-based extraction with medical terminology dictionaries for NER, dictionary lookup for concept normalization, and rule-based relation extraction. Since no neural model training was required, all experiments were conducted on CPU (Intel Core i7, 16GB RAM) with total runtime under 1 hour for all experiments.

**Software Packages.** We use Python 3.10 with pandas (v1.5) and numpy (v1.21) for data processing, and standard library modules for pattern matching and JSON handling.