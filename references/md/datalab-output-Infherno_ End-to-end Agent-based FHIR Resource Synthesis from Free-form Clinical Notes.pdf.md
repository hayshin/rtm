

# Inferno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes

Johann Frei<sup>1</sup> Nils Feldhus<sup>2,3,4</sup> Lisa Raithel<sup>2,3,4</sup>  
Roland Roller<sup>4</sup> Alexander Meyer<sup>2,5</sup> Frank Kramer<sup>1</sup>

<sup>1</sup>IT-Infrastructure for Translational Medical Research, University of Augsburg

<sup>2</sup>BIFOLD – Berlin Institute for the Foundations of Learning and Data <sup>3</sup>Technische Universität Berlin

<sup>4</sup>German Research Center for Artificial Intelligence (DFKI), Berlin

<sup>5</sup>DHZC Medical Data Science, Charité - Universitätsmedizin Berlin

## Abstract

For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources rely on modular, rule-based systems or LLMs with instruction tuning and constrained decoding. Since they frequently suffer from limited generalizability and structural in conformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Inferno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions.

## 1 Introduction

Large language models (LLMs) have demonstrated strong performance in clinical and biomedical domains, as they have been shown to encode domain-specific knowledge (Singhal et al., 2023; Moor et al., 2023). They are increasingly used to answer clinical questions by processing relevant documents at inference time (Zakka et al., 2024; Chen et al., 2025a; Wang et al., 2024). However, this retrieval-based incurs significant latency and computational cost, as documents must be reprocessed for every query. This limits usability for tasks such as retrospective analysis or study planning, where multiple queries over the same data are common (Coromilas et al., 2021; Leibig et al., 2022).

A more scalable solution is to extract structured representations from clinical text in advance. If the extracted structure preserves the relevant information, it can be queried and reused instantly across

![Diagram illustrating the Inferno framework. It shows a workflow starting with a 'Discharge Letter' (cyan box) on the left. This is followed by 'SNOMED CT Tools' (light blue box) and 'Code Search Loop' (green box) in the middle. On the right, there are 'fhir.resources Code Loop' (purple box) and a 'Final Answer (FHIR resource)' (red box) at the bottom. Arrows indicate the flow of information between these components. The 'fhir.resources Code Loop' contains Python-like code for importing FHIR resources and creating a bundle. The 'Final Answer' shows a JSON representation of a FHIR bundle with patient and condition resources.](68dad113f9a15ab01945110cb50cdcfb_img.jpg)

Diagram illustrating the Inferno framework. It shows a workflow starting with a 'Discharge Letter' (cyan box) on the left. This is followed by 'SNOMED CT Tools' (light blue box) and 'Code Search Loop' (green box) in the middle. On the right, there are 'fhir.resources Code Loop' (purple box) and a 'Final Answer (FHIR resource)' (red box) at the bottom. Arrows indicate the flow of information between these components. The 'fhir.resources Code Loop' contains Python-like code for importing FHIR resources and creating a bundle. The 'Final Answer' shows a JSON representation of a FHIR bundle with patient and condition resources.

Figure 1: Illustrative example of how Inferno, an agentic approach for FHIR resource synthesis, processes a discharge letter (top left, cyan) using SNOMED CT tools (light blue) and code search (green) and fhir.resources code loops (purple, right). After a few iterations including tool calls and observations from a Python executor, the LLM agent proceeds to produce a final answer (red) in a FHIR/JSON format, representing the clinical information on patients and medications.

multiple applications. This is particularly important for healthcare service providers and clinical data integration efforts (Leroux et al., 2017; Hong et al., 2019; Pimenta et al., 2023). Here, the FHIR (Fast Healthcare Interoperability Resources)<sup>1</sup> standard provides a flexible and interoperable format for representing healthcare data and is increasingly adopted to support standardized access to complex medical information.

Conventional information extraction (IE) methods, such as classical named entity recognition, are typically designed for narrowly defined tasks and fixed schemata. As such, they lack the flexibility

<sup>1</sup><https://fhir.org/>

to adapt to complex, variable clinical contexts and often fail to produce complete, structured clinical representations. In contrast, LLMs have shown promise for IE methods when they are framed as structured prediction tasks (Dagdelen et al., 2024). A recurring challenge in such tasks is ensuring that the generated output adheres to a specified schema, particularly when downstream components require well-structured inputs (Tavanaei et al., 2024). This is especially true in domains like healthcare, where semantic correctness and schema compliance are essential. Various approaches have been proposed to guide LLM outputs toward structural conformity, including fine-tuning, pre-training, instruction tuning, or constrained decoding (Shin et al., 2021; Geng et al., 2023). Some of them have dealt with text-to-FHIR translation (Sharma et al., 2023; Li et al., 2024; Tabari et al., 2025; Pope and Patooghy, 2025), but often encountered inconsistencies with the desired schema. Agentic LLM approaches that “reason” through intermediate steps using external tools, have emerged as a promising solution. Inspired by frameworks like Toolformer (Schick et al., 2023) and ReAct (Yao et al., 2023), models perform multiple tool-augmented reasoning steps, with validation and retry mechanisms to ensure correct output.

To fill the current gaps, we propose an end-to-end framework that transforms unstructured clinical text into rich, semantically accurate FHIR representations using an agentic LLM-approach. This enables holistic information extraction (Zhang et al., 2025; Shao et al., 2025), supports integration of both legacy and new data, and fosters interoperability across institutions.

Our contributions are:

- An agent-based implementation for text-to-FHIR translation with SNOMED CT terminology integration and FHIR schema validation;
- Evaluation on a synthetic benchmark, including detailed error analysis to characterize failures and their severity;
- A lightweight demonstrator with front-end functionality, supporting both local and proprietary state-of-the-art LLMs.

## 2 Background

FHIR (Fast Healthcare Interoperability Resources) is a widely adopted standard for exchanging

healthcare-related data, developed by the HL7 organization. FHIR defines resources as nested documents, often encoded in JSON, with well-defined types, required fields, enumerations, and references to other resources. A single FHIR resource can represent a broad range of entities, from patients and conditions to administrative structures like coverage or questionnaires.<sup>2</sup> Due to the richness and expressiveness of the standard, it enables the structured encoding of complex medical information and data in an interoperable fashion. A key feature of FHIR is the integration of internal and external code systems, composed as ValueSets to reference specific entities and concepts. Certain data elements may be constrained to a fixed, FHIR-internal code system to define the set of valid data values.<sup>3</sup> For certain fields, concepts can be referenced from external coding systems such as SNOMED CT<sup>4</sup> or LOINC<sup>5</sup>, and the set of valid data values can be further constrained by individual ValueSets, e.g., in order to limit data entries for body site to the subset of SNOMED CT concepts that only refer to body structures. To search for codes and terms in a specific ValueSet, FHIR terminology servers provide a standardized interface for querying valid concepts. These servers commonly support multiple external code systems like SNOMED CT and LOINC in addition to the FHIR-internal code systems.

While the FHIR schema is capable to accurately and verbosely capture complex clinical situations, it may not be used by individual actors to its fullest extent, and only a subset of data elements may be used effectively in some situations depending on use cases and context. In addition, the standard does not always enforce the encoding of certain information into an unambiguous representation. For instance, a bone fracture of the left limb may be expressed as a *Fracture of bone* SNOMED CT concept along with the *bodySite* element referring to the *Structure of left hand* concept, or purely by referring to the *Fracture of bone of left hand* concept. Dosage information could be phrased only by a free-form text element, or by fully utilizing all relevant structured elements, rendering both approaches valid. Beyond this aspect, clinical notes

<sup>2</sup>See an example for a Patient resource object at: <https://hl7.org/fhir/R4/patient-example.json.html>

<sup>3</sup>For instance, `Condition.clinicalStatus` only allows the values active, recurrence, relapse, inactive, remission, and resolved.

<sup>4</sup><https://www.snomed.org/what-is-snomed-ct>

<sup>5</sup><https://loinc.org/get-started/what-loinc-is/>

may also be rather imprecise or ambiguous and require additional and subjective interpretation to fully infer the intended meaning, yet this issue also affects other, non-FHIR-based IE systems. Therefore, comparing predicted and ground truth FHIR data for semantic equivalence and correctness remains a non-trivial task.

**Related Work** [Sharma et al. \(2023\)](#) presented a pipeline for automated digitization of prescription images and focus on populating fields of the FHIR prescription schema. While they used separate components for extraction, normalization, entity recognition and linking, our system is based on an end-to-end approach powered by a single LLM agent. [Li et al. \(2024\)](#) were the first to use LLMs for transforming clinical narratives into HL7 FHIR resources. While their analysis also included human-annotated ground truth data<sup>6</sup>, their scope is limited to MedicationStatement resources. Additionally, they encountered JSON parsing issues due to the lack of constraints, whereas our approach’s validation loop ensures format conformity. [Tabari et al. \(2025\)](#) integrated a syntactic validator and zero- and few-shot strategies into their text-to-FHIR pipeline. Their setup is constrained to sentence-level conversion and exhibits less transparency due to the separation between the OpenAI model and the validator. In contrast, InFherno’s tool-calling approach offers a higher degree of transparency and a larger variety of model choices. [Pope and Patoogh \(2025\)](#) explored a variety of FHIR-related tasks as a benchmark, but simplified them to short QA-style problems and also did not consider any elaborate pipeline with tools. [Riquelme Tornel et al. \(2025\)](#) used GPT-4o and Llama-3.2 alongside clustering and retrieval generation approaches to perform automated FHIR mappings on MIMIC-IV (instead of free text), but missed out on evaluating the results manually.

Finally, [Schmiedmayer et al. \(2025\)](#) aimed for an inverse perspective on the translation task by developing a mobile application that allows users to interact with FHIR resources via an LLM, while [Ehtesham et al. \(2025\)](#) presented an MCP-based agent for summarization and interpretation. Both represent a FHIR-to-text scenario which is focused on patient understanding.

<sup>6</sup>The human-annotated FHIR-GPT data has not been open-sourced to the best of our knowledge.

## 3 InFherno, an Agentic Approach

Building on recent work on LLM agents in the medical domain ([Liao et al., 2025](#); [Rose et al., 2025](#); [Chen et al., 2025b](#); [Wang et al., 2025](#)), we propose an agentic framework that incorporates tool calls and coding to generate structured FHIR output from unstructured clinical text.

The core task is to transform an unstructured clinical text into semantically corresponding FHIR representations. Our approach follows the Thought-Code-Observation structure proposed as the ReAct framework by [Yao et al. \(2023\)](#), and is implemented using the Smolagents ([Roucher et al., 2025](#)) library which supports multi-step LLM agents with Python-code execution. Figure 1 presents a simplified example of the InFherno pipeline<sup>7</sup>: Given a discharge letter, InFherno which is equipped with tools accessing SNOMED CT, performing Code Search, and executing Python code, is tasked to extract information pertinent to patients and medications. In the following, we describe each component:

**Prompt Structure** To guide the agent’s behavior, we include relevant contextual information into the prompt (Figure 4, top left). This includes the unstructured input text, a list of target FHIR resource types, supported ValueSets, example code snippets demonstrating FHIR object creation, and a set of instructions on desired behaviors and constraints.

**Code Search** To integrate FHIR-specific codes that conform to its specification, we provide our agentic system with an external, retrieval-augmented generation-based function to query particular terms in a set of supported FHIR ValueSets. This enables the agent to rely on external code systems, in particular the SNOMED CT ontology, to retrieve potential search results and include them into its context window. The external function call binds to an external FHIR terminology server to obtain a valid query response.

**Structured Data as Code** Within the agent code execution stage, the agent is incentivized to use the `fhir.resources`<sup>8</sup> Python module to create FHIR-conform data instances in a object-oriented fashion. This approach is crucial as it is able to catch morphological and syntactic errors early within the life cycle of the agent loop, and avoids cumbersome data validation that may arise from a purely

<sup>7</sup>Figure 4 in Appendix A shows the extended version.

<sup>8</sup><https://github.com/nazrulworld/fhir-resources/tree/8.0.0>

![](5f18c728fc511750ffcaa626716b920e_img.jpg)

**Inferno**

Agent Chat Log Replay

## Agent Chat

Chat with the agent. Returns a FHIR resource.

.fhir Agent

Patient presents with a headache.

```
{  
  "resourceType": "Bundle",  
  "type": "collection",  
  "entry": [  
    {  
      "fullUrl": "Patient/pat-001",  
      "resource": {  
        "resourceType": "Patient",  
        "id": "pat-001"  
      }  
    },  
    {  
      "fullUrl": "Condition/cond-001",  
      "resource": {  
        "resourceType": "Condition",  
        "id": "cond-001",  
        "clinicalStatus": {  
          "system": "http://terminology.hl7.org/CodeSystem/  
code",  
          "code": "active",  
          "display": "Active"  
        }  
      }  
    }  
  ]  
}
```

▶

Figure 2: Front end of `Inferno` showing a short input text and the final answer as given by Gemini-2.5-Pro in the *Agent Chat* function.

JSON-centric FHIR document generation by the LLM. Since the library can directly provide error feedback, it can also facilitate the recovery from erroneous code predicted initially by the agent.

**Output Formatting** As part of the Smolagents framework, the code agent can stop the agent loop by the `final_answer` function call. Hereby, the agent is instructed to use the JSON-based object serialization of the `fhir.resources` module. This ensures that the response provides a structurally valid, FHIR-compliant JSON output. To deliver all generated FHIR resources to the user, the agent is instructed to aggregate them into a FHIR *Bundle* that encapsulates the complete set.

**Front end** The visual interface of Inferno is built on top of Gradio<sup>9</sup> and allows the user to enter arbitrary clinical notes or select pre-defined examples from our synthetic dataset (Figure 2). Intermediate steps including the tool calls and tool responses as well as the reasoning processes (“thoughts” and “observations”) of the ReAct framework (Yao et al.,

![](af8b95b7fc833cebe89ba6c8ed839984_img.jpg)

Figure 3: Front end of *Inferno* showing an intermediate step (Code Search) during the text-to-FHIR translation with the *Log Replay* function.

2023) are shown at inference time. A *Log Replay* tab (Figure 3) also enables to simulate the execution of already conducted experiments at custom speed without the need of an API key. It supports the default Gemini API and OpenAI API via LiteLLM<sup>10</sup> and both API and local Hugging Face models – a feature inherited from Smolagents. The front-end app is available on <https://github.com/j-frei/Inferno>.

## 4 Experiments

To validate our agentic approach, we apply the agent to a set of medical documents to transform the unstructured text into a set of FHIR resources. Such individual comparison is highly complex due to the depth and richness of the FHIR schema and the complexity of clinical language. Therefore, we conduct a manual inspection and interpretation to assess the prediction quality.

**Experimental Setup** To lower the complexity of the IE task, we limit the set of supported

<sup>9</sup><https://www.gradio.app/>

---

<sup>10</sup> <https://docs.litellm.ai/docs/>

| Level of equivalence                                              | Inferno (w/ Gemini-2.5-Pro) |                    |                                               |                                                       | Human Baseline             |
|-------------------------------------------------------------------|-----------------------------|--------------------|-----------------------------------------------|-------------------------------------------------------|----------------------------|
| <b>Worse than HB</b><br>(field not referenced)<br>(hallucination) | [ +?]                       | Condition.bodySite | "text": "Stirnbereich"                        | "display": "Forehead structure"                       | "code": "52795006",<br>N/A |
|                                                                   | [X+?]                       | Condition.category | "code": "symptom"                             |                                                       |                            |
| <b>Neutral</b><br>(optional field missing)<br>(total equivalence) | [-?]                        | MS.dosage          | N/A                                           | "code": "ordered",<br>"display": "Ordered"            |                            |
|                                                                   | [=?]                        | .doseAndRate.type  | N/A                                           |                                                       |                            |
|                                                                   |                             | Condition.severity | "code": "255604002",<br>"display": "Mild"     | "code": "255604002",<br>"display": "Mild"             |                            |
| <b>Better than HB</b><br>(inaccurate reference)                   | [/+!]                       | Condition.code     | "code": "422400008",<br>"display": "Vomiting" | "code": "422587007",<br>"display": "Nausea (finding)" |                            |

Table 1: Examples for level of equivalence and the manual validation between system output and human baseline.

FHIR resources to Patient, Condition, and MedicationStatement, as we consider these resource types to fit best to key clinical entities. The FHIR R4 release is targeted as it currently is the latest *normative* release version.

For the manual evaluation, we compare the manual annotation and the generated annotation by verifying individual *items* of each FHIR object. We define an item as a single unit of information, that may refer to, for instance, a single `birthDate` field but could also refer to a nested object item that describes a reference to a concept from an external coding system. Since the internal structure of certain objects is only meaningful in its entirety, we consider them as monolithic items in the evaluation, rather than decomposing their components.

**Models** For our agentic approach, we use Gemini-2.5-Pro<sup>11</sup> due to its support for long-context sequences, which is crucial for complex agent loops (Zhang et al., 2024; Jiang et al., 2025), and its strong performance in long context tasks, and pragmatic reasons involving easy and low-cost applicability. While we considered the option to also evaluate open-weights models, we were constrained in terms of hardware resources, rendering the use of larger models with long context sizes infeasible. Nevertheless, we regard this as an important direction for future investigation.

**Data** The use of a remote, commercial LLM limits the set of evaluable text data, as clinical data cannot be easily shared due to privacy regulations and data use agreements. In addition, the de-identification of most public datasets impedes the evaluation of certain aspects of the Patient resource, such as the extraction of names or birth-dates, as these fields are typically removed. We solve this problem by relying on synthetic data. We synthesize a dataset of German discharge letters us-

ing ChatGPT, following related work in synthetic EHR generation (Lin et al., 2025). The raw texts required manual editing. Obvious placeholder names, such as "Max Mustermann", were replaced in the synthetic texts to ensure realistic, non-repetitive patient names and addresses to improve the authenticity of the data.<sup>12</sup> To obtain suitable reference FHIR data, we annotate the first 10 documents from the corpus by manually extracting the relevant corresponding FHIR resources, referred to as human baseline (HB).

## 5 Results & Discussion

**Manual Validation** We manually compared the items of each FHIR object in the human baseline annotation with their corresponding agent-generated equivalents. Missing items were tracked using + for those absent in the human baseline and - for those missing in the prediction. Potentially equivalent values were categorized as exact matches (==), semantically equivalent (=), or different (+-).

Items were tagged with / if the prediction value was preferable to the baseline, and with | if the baseline value was preferable. Items were left untagged when no clear preference could be determined or justified.

To distinguish essential from less relevant features during evaluation, we annotated core FHIR objects of major importance, such as patient information or the main diagnosis extracted from the input text, with !, and less critical elements, like vaguely described symptoms, with ?. The importance level of a FHIR object determined the default importance of its internal items, unless overridden by manually applied, item-specific tags. These overrides were primarily used to demote non-essential items such as `Condition.verificationStatus`

<sup>11</sup><https://ai.google.dev/gemini-api/docs/models>

<sup>12</sup>The documents are publicly available on GitHub at <https://github.com/j-frei/Inferno>.

| Sign    | Description                         | Worse than HB | Neutral | Better than HB |    |    |    |
|---------|-------------------------------------|---------------|---------|----------------|----|----|----|
| ! and ? | (cruciality)                        | !             | ?       | !              | ?  | !  | ?  |
| =       | (semantically related)              | 0             | 4       | 0              | 4  | 0  | 0  |
| ==      | (completely identical)              | 0             | 0       | 121            | 83 | 0  | 0  |
| +       | (lacking in HB)                     | 0             | 10      | 0              | 23 | 13 | 67 |
| -       | (lacking in PD)                     | 6             | 15      | 0              | 67 | 0  | 0  |
| +-      | (value difference)                  | 0             | 10      | 0              | 12 | 5  | 1  |
| X       | (semantic hallucination or invalid) | 1             | 9       | 0              | 0  | 0  | 0  |
| total   |                                     | 46            | 314     | 86             |    |    |    |

Table 2: Quantitative analysis between predicted (PD) and human baseline (HB) indicating the success and failure cases of InFherno.

or `Patient.name.use`. Conversely, items from otherwise crucial FHIR resources, e.g., `Condition.subject` or `Condition.code`, were generally considered essential, as they carry core information. Some examples are shown in Table 1.

**Quantitative Analysis** Table 2 shows the statistics of the human annotation and the equivalence of predictions by Gemini-2.5-Pro (PD) with that human baseline. The results show that most frequent items are completely identical between the human baseline and prediction, in particular for the crucial items. Greater divergence is observed in the non-crucial items. This outcome is expected, as these items typically pertain to aspects that are less distinctly expressed in the input text, such as vague symptom descriptions or ambiguous phrasing, in contrast to crucial elements like the main diagnosis, which are generally stated more explicitly. In general, the scores indicate the general trend that the human annotation performs inferior to the InFherno agent. As our approach enforces the FHIR syntax through the code-driven FHIR object creation, hallucinations occur only on the semantic level, and in less than 2.3% of all items.

**Key Findings** The manual validation highlights several important observations. First, the phrasing of the input text plays a critical role in annotation consistency. Vague or ambiguous expressions frequently lead to disagreements between the predicted and reference annotations, particularly for non-crucial items. In contrast, plainly stated and well-structured information is more reliably and consistently captured.

Second, many divergences can be attributed to the partially subjective nature of FHIR in fringe cases. Minor or nonspecific health issues often fall into a gray area. These may either be excluded or encoded in different ways, such as as

a `Condition` or an `Observation`. Since the experimental setup allowed only the use of `Patient`, `MedicationStatement`, and `Condition` resource types, the agent was not permitted to use the `Observation` resource, which limited some of its encoding options.

Furthermore, the InFherno agent appears to be more cautious when deciding whether to encode uncertain symptoms. At the same time, it demonstrates stronger recall for clearly stated information that human annotators sometimes overlook. For example, the agent successfully included an address field that was missing in the human annotation. It also inferred an `onsetDateTime` by subtracting six weeks from the encounter date, which is a detail the human annotator did not encode.

These findings indicate that while human annotations are prone to fatigue and inconsistency, especially in repetitive and detail-oriented tasks, automated agents benefit from their ability to process dense text data using their large context as receptive field. As a result, they can achieve more reliable and comprehensive structured data extraction from our clinical text samples.

## 6 Conclusion

In conclusion, InFherno presents a robust and effective framework and interface for transforming unstructured clinical data into standardized FHIR resources. Its agentic design, integrating external knowledge and validation, addresses critical challenges in clinical information extraction, paving the way for improved data interoperability in healthcare. Future work includes the fine-tuning of smaller language models on the text-to-FHIR task and the integration of more FHIR resource types while further strengthening the robustness, and evaluate the approach on more diverse datasets.

## Limitations

Our current work has several limitations that warrant consideration. First, our evaluation predominantly utilized a strong commercial LLM (Gemini-2.5-Pro) due to its long-context support and performance, and thus does not include an assessment of open-weights models, which might exhibit different performance characteristics and resource requirements. Second, the validation data is synthetic and limited in quantity. While synthetic data addresses privacy concerns with real-world clinical notes, it may not fully capture the complexities and loose structuring often found in genuine clinical documentation, potentially affecting generalizability. Manual evaluation of a larger, more diverse dataset was infeasible given the labor intensity and expertise required. Third, the evaluation lacked medical expert input, relying instead on a single annotator for the human baseline, which inherently introduces a degree of subjectivity. Finally, our scope was intentionally limited to a subset of FHIR resource types (Patient, Condition, MedicationStatement). Expanding to a broader range of FHIR resources would likely necessitate more verbose guidance in the system prompt, potentially increasing computational cost and latency. Finally, from a legal perspective, it's important to note that Inferno interacts with a FHIR terminology server that includes a loaded SNOMED CT ontology. Therefore, a SNOMED CT license may be required if self-hosting a FHIR terminology server is desired.

## Ethics Statement

As the preferred solution is to choose state-of-the-art proprietary models, we want to emphasize that users should be careful in selecting what data they enter. Most of the real-world medical datasets have licences and usage restrictions, so we recommend to use synthetic data only. Users should acknowledge the risk of leaking private data and de-identification.

## References

Hanjie Chen, Zhouxiang Fang, Yash Singla, and Mark Dredze. 2025a. [Benchmarking large language models on answering and explaining challenging medical questions](#). In *Proceedings of the 2025 Conference of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 3563–3599, Albuquerque, New Mexico. Association for Computational Linguistics.

Shan Chen, Pedro Moreira, Yuxin Xiao, Sam Schmidgall, Jeremy Warner, Hugo Aerts, Thomas Hartvigsen, Jack Gallifant, and Danielle S. Bitterman. 2025b. [Medbrowsecomp: Benchmarking medical deep research and computer use](#). *arXiv*, abs/2505.14963.

Ellie J. Coromilas, Stephanie Kochav, Isaac Goldenthal, Angelo Biviano, Hasan Garan, Seth Goldbarg, Joon-Hyuk Kim, Ilhwan Yeo, Cynthia Tracy, Shant Ayanian, Joseph Akar, Avinainder Singh, Shashank Jain, Leandro Zimerman, Maurício Pimentel, Stefan Osswald, Raphael Twerenbold, Nicolas Schaeli, Lia Crotti, Daniele Fabbri, Gianfranco Parati, Yi Li, Felipe Atienza, Eduardo Zatarain, Gary Tse, Keith Sai Kit Leung, Milton E. Guevara-Valdivia, Carlos A. Rivera-Santiago, Kyoko Soejima, Paolo De Filippo, Paola Ferrari, Giovanni Malanchini, Prapa Kanagaratnam, Saud Khawaja, Ghada W. Mikhail, Mauricio Scanavacca, Ludhmila Abrahão Hajjar, Brenno Rizério, Luciana Sacilotto, Reza Mollazadeh, Masoud Eslami, Vahideh Laleh far, Anna Vittoria Mattioli, Giuseppe Boriani, Federico Migliore, Alberto Cipriani, Filippo Donato, Paolo Compagnucci, Michela Casella, Antonio Dello Russo, James Coromilas, Andrew Aboyne, Connor Galen O'Brien, Fatima Rodriguez, Paul J. Wang, Aditi Naniwadekar, Melissa Moy, Chia Siang Kow, Wee Kooi Cheah, Angelo Auricchio, Giulio Conte, Jongmin Hwang, Seongwook Han, Pietro Enea Lazzerini, Federico Franchi, Amato Santoro, Pier Leopoldo Capecci, Jose A. Joglar, Anna G. Rosenblatt, Marco Zardini, Serena Bricoli, Rosario Bonura, Julio Echarte-Morales, Tomás Benito-González, Carlos Minguito-Carazo, Felipe Fernández-Vázquez, and Elaine Y. Wan. 2021. [Worldwide survey of covid-19-associated arrhythmias](#). *Circulation: Arrhythmia and Electrophysiology*, 14(3):e009458.

John Dagdelen, Alexander Dunn, Sanghoon Lee, Nicholas Walker, Andrew S Rosen, Gerbrand Ceder, Kristin A Persson, and Anubhav Jain. 2024. [Structured information extraction from scientific text with large language models](#). *Nature Communications*, 15(1):1418.

Abul Ehtesham, Aditi Singh, and Saket Kumar. 2025. [Enhancing clinical decision support and ehr insights through llms and the model context protocol: An open-source mcp-fhir framework](#). *arXiv*, abs/2506.13800.

Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. 2023. [Grammar-constrained decoding for structured NLP tasks without finetuning](#). In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 10932–10952, Singapore. Association for Computational Linguistics.

Na Hong, Andrew Wen, Feichen Shen, Sunghwan Sohn, Chen Wang, Hongfang Liu, and Guoqian Jiang. 2019.

- Developing a scalable fhir-based clinical data normalization pipeline for standardizing and integrating unstructured and structured electronic health record data. *JAMIA Open*, 2(4):570–579.
- Mingjian Jiang, Yangjun Ruan, Luis Lastras, Pavan Kapanipathi, and Tatsunori Hashimoto. 2025. [Putting it all into context: Simplifying agents with lclms. arXiv, abs/2505.08120.](#)
- Christian Leibig, Moritz Brehmer, Stefan Bunk, Danylyn Byng, Katja Pinker, and Lale Umutlu. 2022. [Combining the strengths of radiologists and ai for breast cancer screening: a retrospective analysis. The Lancet Digital Health](#), 4(7):e507–e519.
- Hugo Leroux, Alejandro Metke-Jimenez, and Michael J Lawley. 2017. [Towards achieving semantic interoperability of clinical study data with fhir. Journal of biomedical semantics](#), 8:1–14.
- Yikuan Li, Hanyin Wang, Halid Z. Yerebakan, Yoshihisa Shinagawa, and Yuan Luo. 2024. [FHIR-GPT enhances health interoperability with large language models. NEJM AI](#), 1(8):A1cs2300301.
- Yusheng Liao, Shuyang Jiang, Yanfeng Wang, and Yu Wang. 2025. [Reflectool: Towards reflection-aware tool-augmented clinical agents. arXiv, abs/2410.17657.](#)
- Yihan Lin, Zhirong Bella Yu, and Simon Lee. 2025. [A case study exploring the current landscape of synthetic medical record generation with commercial llms. arXiv, abs/2504.14657.](#)
- Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumbholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. 2023. [Foundation models for generalist medical artificial intelligence. Nature](#), 616(7956):259–265.
- Nuno Pimenta, António Chaves, Regina Sousa, António Abelho, and Hugo Peixoto. 2023. [Interoperability of clinical data through fhir: A review. Procedia Computer Science](#), 220:856–861. The 14th International Conference on Ambient Systems, Networks and Technologies Networks (ANT) and The 6th International Conference on Emerging Data and Industry 4.0 (EDI40).
- Tia Pope and Ahmad Patoogh. 2025. [Comparative evaluation of gpt models in fhir proficiency. ACM Trans. Intell. Syst. Technol.](#) Just Accepted.
- Álvaro Riquelme Tornel, Pedro Costa del Amo, and Catalina Costa Martínez. 2025. [Large language models for automating clinical data standardization: HI7 fhir use case. arXiv, abs/2507.03067.](#)
- Daniel Rose, Chia-Chien Hung, Marco Lepri, Israa Alqassem, Kiril Gashteovski, and Carolin Lawrence. 2025. [Meddxagent: A unified modular agent framework for explainable automatic differential diagnosis. arXiv, abs/2502.19175.](#)
- Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. 2025. ‘smolagents’: a smol library to build great agentic systems. <https://github.com/huggingface/smolagents>.
- Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. [Toolformer: Language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems.](#)
- Paul Schmiedmayer, Adrit Rao, Philipp Zagar, Lauren Aalami, Vishnu Ravi, Aydin Zahedivash, Dong han Yao, Arash Fereydooni, and Oliver Aalami. 2025. [Llmonfhir. JACC: Advances](#), 4(6\_Part\_1):101780.
- Chong Shao, Douglas Snyder, Chiran Li, Bowen Gu, Kerry Ngan, Chun-Ting Yang, Jiageng Wu, Richard Wyss, Kueiyu Joshua Lin, and Jie Yang. 2025. [Scalable medication extraction and discontinuation identification from electronic health records using large language models. arXiv, abs/2506.11137.](#)
- Megha Sharma, Tushar Vatsal, Srujana Merugu, and Aruna Rajan. 2023. [Automated digitization of unstructured medical prescriptions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics \(Volume 5: Industry Track\)](#), pages 794–805, Toronto, Canada. Association for Computational Linguistics.
- Richard Shin, Christopher Lin, Sam Thomson, Charles Chen, Subhro Roy, Emmanouil Antonios Platanios, Adam Pauls, Dan Klein, Jason Eisner, and Benjamin Van Durme. 2021. [Constrained language models yield few-shot semantic parsers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing](#), pages 7699–7715, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. [Large language models encode clinical knowledge. Nature](#), 620(7972):172–180.
- Parinaz Tabari, Alfonso Piscitelli, Gennaro Costagliola, and Mattia de Rosa. 2025. [Assessing the potential of an llm-powered system for enhancing fhir resource validation. In Intelligent Health Systems—From Technology to Data and Knowledge](#), pages 803–807. IOS Press.
- Amir Tavanaei, Kee Kiat Koo, Hayreddin Ceker, Shaobai Jiang, Qi Li, Julien Han, and Karim Bouyarmane. 2024. [Structured object language modeling \(SO-LM\): Native structured objects generation conforming to complex schemas with self-supervised denoising. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track](#), pages 821–828, Miami, Florida, US. Association for Computational Linguistics.

- Bowen Wang, Jiuyang Chang, Yiming Qian, Guoxin Chen, Junhao Chen, Zhouqiang Jiang, Jiahao Zhang, Yuta Nakashima, and Hajime Nagahara. 2024. [Di-reCT: Diagnostic reasoning for clinical notes via large language models](#). In *The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track*.
- Wenxuan Wang, Zizhan Ma, Zheng Wang, Chenghan Wu, Jiaming Ji, Wenting Chen, Xiang Li, and Yixuan Yuan. 2025. [A survey of llm-based agents in medicine: How far are we from baymax?](#) *arXiv*, abs/2502.11211.
- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. [React: Synergizing reasoning and acting in language models](#). In *The Eleventh International Conference on Learning Representations*.
- Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R. Dalal, Jennifer L. Kim, Michael Moor, Robyn Fong, Curran Phillips, Kevin Alexander, Euan Ashley, Jack Boyd, Kathleen Boyd, Karen Hirsch, Curt Langlotz, Rita Lee, Joanna Melia, Joanna Nelson, Karim Sallam, Stacey Tullis, Melissa Ann Vogelsong, John Patrick Cunningham, and William Hiesinger. 2024. [Almanac — retrieval-augmented language models for clinical medicine](#). *NEJM AI*, 1(2):AIoa2300068.
- Xiao Yu Cindy Zhang, Carlos R. Ferreira, Francis Rossignol, Raymond T. Ng, Wyeth Wasserman, and Jian Zhu. 2025. [Casereportbench: An llm benchmark dataset for dense information extraction in clinical case reports](#). *arXiv*, abs/2505.17265.
- Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Ö. Arik. 2024. [Chain of agents: Large language models collaborating on long-context tasks](#). In *Advances in Neural Information Processing Systems*, volume 37, pages 132208–132237. Curran Associates, Inc.

## A Examples

Figure 4 illustrates a complete example of a text-to-FHIR translation flow.

## B Example of a Synthetic Clinical Document

The Figure 5 shows the first document from our synthesized text corpus. All documents are accessible on GitHub at the following url: <https://github.com/j-frei/Inferno>.



Betreff: Arztberichtsbrief – Patienteninformationen

Sehr geehrter Dr. Peters,

hiermit möchte ich Ihnen einen aktuellen Bericht über den Gesundheitszustand von Herrn Uwe Jaeger, geboren am 10. Februar 1975, vorlegen. Herr Jaeger wurde am 20. Juni 2023 in unserer Klinik, dem St. Ursula Krankenhaus, zur weiteren Untersuchung und Behandlung aufgenommen.

Anamnese:

Herr Jaeger suchte unsere Notaufnahme mit anhaltenden Beschwerden im Magen-Darm-Bereich auf. Er berichtete über starke Bauchschmerzen, Übelkeit, Erbrechen und Gewichtsverlust in den letzten vier Wochen. Er verneinte jegliche vorherige Operationen oder relevante Vorerkrankungen. Herr Jaeger ist Nichtraucher und konsumiert keinen Alkohol.

Klinischer Befund:

Bei der körperlichen Untersuchung zeigten sich eine allgemeine Schwäche und ein mäßig abgeschwächter Allgemeinzustand. Der Bauch war diffus druckempfindlich, ohne spürbare Vergrößerungen der Organe. Keine Zeichen einer Peritonitis waren erkennbar. Die übrige körperliche Untersuchung ergab keine auffälligen Befunde.

Diagnostische Maßnahmen:

Um die Ursache der Beschwerden zu ermitteln, wurden bei Herrn Jaeger verschiedene diagnostische Tests durchgeführt. Eine Blutuntersuchung ergab eine erhöhte Anzahl weißer Blutkörperchen und eine leichte Anämie. Der Leberfunktionstest zeigte normale Werte. Ein abdominales Ultraschall wurde durchgeführt, das keine strukturellen Abnormalitäten zeigte. Eine Endoskopie des oberen Verdauungstrakts wurde ebenfalls durchgeführt, bei der eine erosive Gastritis festgestellt wurde.

Diagnose:

Basierend auf den klinischen Symptomen, den Laborergebnissen und der Endoskopie wurde bei Herrn Jaeger die Diagnose einer erosiven Gastritis gestellt.

Therapie:

Um die Symptome zu lindern und die Schleimhaut im Magen zu heilen, wurde Herr Jaeger eine Kombinationstherapie verschrieben. Er erhält eine Protonenpumpenhemmer (PPI) für acht Wochen, um die Magensäureproduktion zu reduzieren. Zusätzlich wurde ihm ein Antazidum verschrieben, um den sofortigen Effekt einer schnellen Symptomlinderung zu erzielen. Er erhielt auch Anweisungen zur Vermeidung von auslösenden Nahrungsmitteln, wie scharfe und säurehaltige Lebensmittel.

Verlauf und Prognose:

Herr Jaeger hat die empfohlene Therapie begonnen und wurde über mögliche Nebenwirkungen und Maßnahmen zur Verbesserung seines Gesundheitszustands aufgeklärt. Wir werden ihn in regelmäßigen Abständen zu Follow-up-Terminen einladen, um den Verlauf seiner Symptome zu überwachen und gegebenenfalls weitere Untersuchungen durchzuführen.

Abschließend möchte ich Ihnen versichern, dass wir die bestmögliche Versorgung für Herrn Jaeger sicherstellen und eng mit ihm zusammenarbeiten werden, um eine schnelle Genesung zu erreichen.

Bei weiteren Fragen stehe ich Ihnen gerne zur Verfügung.

Mit freundlichen Grüßen,

Dr. Anna Karolin Vogel

Fachärztin für Innere Medizin  
St. Ursula Krankenhaus

Figure 5: The full text from the first document from the synthetic corpus.