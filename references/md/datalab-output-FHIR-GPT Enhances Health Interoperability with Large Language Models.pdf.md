

![HHS Public Access logo featuring a stylized eagle and the text 'DEPARTMENT OF HEALTH & HUMAN SERVICES, USA'](2dfa6ac3edfe874f68aa0cbccaa42322_img.jpg)

HHS Public Access logo featuring a stylized eagle and the text 'DEPARTMENT OF HEALTH & HUMAN SERVICES, USA'

## HHS Public Access

Author manuscript

*NEJM AI*. Author manuscript; available in PMC 2025 July 31.

Published in final edited form as:

*NEJM AI*. 2024 August ; 1(8): . doi:10.1056/aics2300301.

# FHIR-GPT Enhances Health Interoperability with Large Language Models

Yikuan Li, M.S.<sup>1,2</sup>, Hanyin Wang, Ph.D.<sup>1</sup>, Halid Z. Yerebakan, Ph.D.<sup>2</sup>, Yoshihisa Shinagawa, Ph.D.<sup>2</sup>, Yuan Luo, Ph.D.<sup>1</sup>

<sup>1</sup>Division of Health and Biomedical Informatics, Department of Preventive Medicine, Feinberg School of Medicine, Northwestern University, Chicago

<sup>2</sup>Siemens Healthineers, Malvern, PA

## Abstract

Advancing health data interoperability can significantly benefit research, including phenotyping, clinical trial support, and public health surveillance. Federal agencies such as the Office of the National Coordinator of Health Information Technology, the Centers for Disease Control and Prevention, and the Centers for Medicare & Medicaid Services are collectively promoting interoperability by adopting the Fast Healthcare Interoperability Resources (FHIR) standard. However, the heterogeneous structures and formats of health data present challenges when transforming electronic health record data into FHIR resources. This challenge is exacerbated when critical health information is embedded in unstructured rather than structured data formats. Previous studies relied on separate rule-based or deep learning-based natural language processing (NLP) tools to complete the FHIR transformation, leading to high development costs, the need for extensive training data, and the complex integration of various NLP tools. In this study, we assessed the ability of large language models (LLMs) to convert clinical narratives into FHIR resources. The FHIR-generative pretrained transformer (GPT) was developed specifically for the transformation of clinical texts into FHIR medication statements. In experiments involving 3671 snippets of clinical texts, FHIR-GPT achieved an exact match rate of more than 90%, surpassing the performance of existing methods. FHIR-GPT improved the exact match rates of existing NLP pipelines by 3% for routes, 12% for dose quantities, 35% for reasons, 42% for forms, and more than 50% for timing schedules. These findings provide confirmation of the potential for leveraging LLMs to enhance health data interoperability. (Funded by the National Institutes of Health and by an American Heart Association Predoctoral Fellowship.)

## Introduction

Interoperability enhances health care providers' ability to deliver safe, effective, and patient-focused care. It also opens new avenues for individuals and caregivers to access electronic health data for care coordination and management.<sup>1</sup> The promotion of interoperability has become an integral aspect of various health initiatives, from ensuring health equity

---

Dr. Luo can be contacted at yuan.luo@northwestern.edu or at 750 N. Lake Shore Drive, 11th Floor, Chicago, IL 60611.

Author disclosures and other supplementary materials are available at [ai.nejm.org](https://www.nejm.org).

to responding to public health emergencies.<sup>2</sup> Federal agencies, including the Office of the National Coordinator of Health Information Technology, the Centers for Disease Control and Prevention, and the Centers for Medicare & Medicaid Services, collaborate to promote interoperability through the adoption of Fast Healthcare Interoperability Resources (FHIR).<sup>1,3,4</sup> Created by the Health Level Seven (HL7) standards development organization, FHIR is designed to enable the swift and efficient exchange of health data.<sup>5</sup> Increasingly adopted for modeling and integrating both structured and unstructured data, FHIR supports a wide range of health research applications, including computational phenotyping, clinical trial support, and the development of surveillance systems.<sup>6–14</sup>

Transforming health data into the FHIR format presents a major challenge, because health organizations have unique infrastructures, standards, and formats for generating, storing, and organizing health data.<sup>15</sup> This challenge is exacerbated when critical health information is embedded in unstructured rather than structured data formats. Efforts to facilitate the transformation of unstructured data into FHIR resources are ongoing in both the academic and commercial sectors. In the academic sector, Hong et al.<sup>16</sup> integrated clinical natural language processing (NLP) tools, including cTAKES, MedXN, and MedTime, to extract clinical entities from corresponding document sections and standardize them into FHIR resources.<sup>17,18</sup> Opioid2FHIR, developed by Wang et al.,<sup>19</sup> comprises a system that uses multiple deep learning–based NLP techniques for opioid information extraction and normalization. In the commercial sector, Google Cloud has released the Healthcare Natural Language (HNL) application programming interface (API), capable of converting medical text input into FHIR resources.<sup>20</sup> Azure Health Data is proficient at converting semistructured data into FHIR resources but does not handle free-text unstructured input.<sup>21</sup> All of these solutions require the use of multiple NLP tools in sequence. Creating such a pipeline often demands substantial computational resources, annotated data, and human effort. Moreover, as data progresses through the pipeline, compounding errors from each NLP tool can reduce overall accuracy.

We therefore proposed leveraging pretrained large language models (LLMs) and efficient prompt engineering to facilitate the transformation of free-text input into FHIR resources. To this end, we manually annotated a dataset of free text to FHIR MedicationStatement resource transformation pairs and subsequently evaluated the transformation accuracy of our model, the FHIR–generative pretrained transformer (FHIR-GPT), against existing NLP pipelines using the annotated dataset.

## Methods

### DATA ANNOTATION

To the best of our knowledge, there is no publicly available dataset with corresponding text and FHIR resource pairs. We therefore annotated a dataset containing pairs of free-text input and corresponding FHIR MedicationStatement resource output. The FHIR resource of MedicationStatement is a record of “a medication being taken by a patient or that a medication has been given to a patient, where the record is the result of a report from the patient or another clinician, or derived from supporting information.”<sup>22</sup> This transformation holds particular significance because many medication-related details, such as the reasons

for administration and dosage instructions, often remain absent in structured data. Clinical notes within the electronic health record system frequently serve as the primary source for this medication-related information. Table 1 provides detailed examples of the elements in MedicationStatement.

The clinical text input was obtained from the 2018 medication extraction challenge of the National NLP Clinical Challenges (n2c2).<sup>23</sup> The text snippets, each containing mentions of one medication and all its associated entities, were extracted from the discharge summaries (Fig. 1). These extracted snippets, each tied to a specific medication, served as input for both annotations and transformations. Our human annotation consisted of three key steps. We started by standardizing the entities from free text into clinical terminology coding systems. To achieve this, we leveraged the word spans of entities provided in the n2c2 dataset and manually looked up the HL7 CodeSets, SNOMED CT Browser, and the RxNav for standardization.<sup>24,25</sup> We then assembled the identifiers, codes, texts, elements, and structures into a complete MedicationStatement resource in JSON format as per the FHIR v6.0.0: R6 implementation guide.<sup>22</sup> Finally, the human-converted MedicationStatement resources underwent validation using the official FHIR validator to ensure compliance with FHIR standards.<sup>26</sup> More details of the annotation process are provided in the Supplementary Appendix.

### LLM DEPLOYMENT

We experimented with three LLMs: OpenAI GPT-4, Llama-2-70B, and Falcon 180B.<sup>27–29</sup> We accessed GPT-4 (model: gpt-4-32k as of 2023-05-15) through the Azure OpenAI API service. Multiple asynchronous API calls were made to enhance efficiency. Both Llama-2-70B and Falcon 180B were deployed on the Health Insurance Portability and Accountability Act–compliant firewalled local servers with multiple graphics processing units. GPTQ was used to accelerate the inference time for Llama-2-70B and Falcon 180B.<sup>30</sup>

These LLMs were used to transform the free-text entries into MedicationStatements conforming to the FHIR standard, using the few-shot in-context learning that includes four to five examples of transformations in the prompts. Each clinical text snippet was fed into the LLM individually to generate the MedicationStatement resource. Five separate prompts were leveraged to instruct the LLM to transform the free-text input into the elements of a MedicationStatement resource, including medication details (drug name, strength, and form), route, timing, dosage, and reason. All few-shot prompts followed a template consisting of task instructions, expected output FHIR templates in JSON format, four to five examples of transformations, a comprehensive list of codes from which the model could make selections, and the input text to be transformed.

Because there was no fine-tuning or domain-specific adaptation in our experiments, we initially had the LLMs generate the FHIR resource for a small trial dataset (N=100). We then manually reviewed the discrepancies between the LLM-generated FHIR output and our human annotations. Common mistakes were identified and used to iteratively refine the prompts. Once we were satisfied with the performance on the trial set, we applied the prompt to the remaining testing set (N=3571) and reported the results. There were slight variations in the prompts for each LLM, because different LLMs may be sensitive to

particular prompt wording. The LLMs were not instructed to look up the SNOMED codes for the “medication” and “reason” elements, because there are thousands of SNOMED CT Medication and Finding codes, exceeding the token limits of the LLMs. Instead, the LLMs were instructed to identify the contexts mentioned in the input text and convert them into the appropriate FHIR format. We instructed the LLMs to look up other elements, such as routes and forms, directly from the code set (numbering in hundreds). More prompting details, including example prompts, are provided in the Supplementary Appendix. Our annotated dataset is available on PhysioNet (<https://physionet.org/>).

### PERFORMANCE EVALUATION

The LLMs were compared with two existing NLP pipelines: NLP2FHIR and Google HNL API.<sup>16,20</sup> NLP2FHIR was built based on a previous version of the FHIR implementation guide R5; the Google HNL API primarily standardized concepts to Unified Medical Language System Concept Unique Identifiers, and the latest guide R6 and SNOMED were used in our annotations and the LLM transformations. We therefore made adaptations and conversions to ensure a fair comparison. We deployed the NLP2FHIR pipeline on our firewalled local servers and accessed the Google HNL API through the Google Cloud Healthcare API.

When evaluating the FHIR resources generated by the LLMs, a format validation check was first conducted to ensure that the output was in a valid JSON format. On passing the validation, we evaluated the generated resources with an exact match rate. This strict criterion required that the resources generated by the LLMs exactly matched the human annotations in all aspects, including structures, codes, and cardinality. The results of other metrics are reported in the Supplementary Appendix.

## Results

We annotated a total of 3671 pairs of free-text input and FHIR MedicationStatement resource output (Table 1). The free-text input was derived from discharge summaries for 280 admissions. The annotated resources encompass 625 distinct medications in 26 different forms and are associated with 354 different reasons, as well as 16 administration routes. These elements display varying levels of availability, ranging from approximately 30% for reasons to 65% for timing schedules. The annotated resources in the JSON structure have an average number of objects of 58.2 (standard deviation, 16.2) and an average depth of 6.7 (standard deviation, 0.5).

Transformations using FHIR-GPT achieved an exact match rate of more than 90% for all elements, outperforming both baseline models and all other LLMs (Table 2). Specifically, compared with existing NLP pipelines, FHIR-GPT improved the exact match rate by 3% for routes, 12% for dose quantities, 35% for reasons, 42% for forms, and more than 50% for timing schedules. Among all LLMs, a trend was observed of increasing accuracy as the parameter size increased. FHIR-GPT, with approximately 1.7 trillion parameters, surpassed the 180-billion-parameter Falcon model and further improved upon the 70-billion-parameter Llama-2 model.

The reproducibility of using LLMs for FHIR transformation was examined through two experiments conducted 6 months apart, in September 2023 and March 2024. No updates were applied to the weights of Falcon and Llama during this period. In the March 2024 experiment of FHIR-GPT, the latest gpt-4-turbo model was used; this is an upgrade from the gpt-4-32k model used in the previous September 2023 run. The gpt-4-turbo model boasts an expanded context window, stretching from 32k to 128k tokens, and incorporates additional training data spanning from September 2021 to April 2023. The results of reproducibility are presented in Table 3. Although slight fluctuations were observed across various models and elements, none exceeded a decrease of 2%. This indicates a relative stability in using LLMs for FHIR transformation, even with the update to the foundation model. We also conducted external validation and confirmed that our proposed methods retain a robust level of generalizability when applied to an alternative data source.<sup>31</sup> The results of the external validation are provided in the Supplementary Appendix.

An error analysis was conducted to investigate instances in which FHIR-GPT and human annotation diverge, with a particular focus on drug routes as an example. The 204 disagreements in transforming drug routes were categorized into five types of errors: false-negative errors, false-positive errors, mismatched errors, syntax errors, and content filter rejections. A comprehensive breakdown, along with descriptions, examples, and distribution of these errors, is provided in Table 4. False-negative errors primarily result from FHIR-GPT's insensitivity to certain medical abbreviations, such as 'gtt,' or its failure to associate medical terms such as 'lumen' with the intravenous route. Conversely, false-positive errors occur when FHIR-GPT inaccurately introduces nonexistent information or identifies annotation errors, such as the oversight of 'IVIG,' which was unrecognized in the n2c2 expert annotation and therefore omitted from our human annotation. The mismatched error category presents a combination of false-negative and false-positive errors. We posit that such errors could be mitigated through the incorporation of more domain-specific knowledge or examples in the prompt, or through the injection of knowledge bases with retrieval-augmented generations. In addition, there are rare instances in which output cannot be parsed as JSON. These parsing issues can be easily rectified with a simple format adjustment, replacing all double quotes with a single quote or using more advanced JSON parsing tools. Finally, the Azure platform occasionally rejects requests with content filters to avoid harmful content in the prompts, although such filters can be opted out of if necessary.

## Discussion

The current study compared three potential pathways for transforming free-text clinical input into FHIR resources. Although human annotation is the gold standard for transformation, its dependence on extensive human efforts poses scalability limitations. Existing NLP pipelines can automate these transformation processes but demand substantial training data and resources, with challenges in generalizability and transferability. Conversely, a new pipeline must undergo training or fine-tuning for even minimal changes in the code set or expansion to new resources. In addition, the multistep transformation process incurs considerable maintenance costs, demanding meticulous tracing for effective error debugging across all steps. FHIR-GPT, harnessing the power of pretrained LLMs, eliminates the need for high-cost training and depends on only minimal human annotation for the

few-shot examples in the prompts. FHIR-GPT also achieves a superior level of accuracy compared with the approach using NLP pipelines. Moreover, by changing the template and the corresponding code set in the prompt, FHIR-GPT has the potential to generalize to other resources without the requirement for resource-specific training. We believe that leveraging FHIR-GPT has the potential to greatly enhance interoperability, given its ease of implementation, high accuracy, and broad generalizability.

We recognize several limitations in our study. First, although FHIR-GPT exhibited superior performance compared with Llama and Falcon, its significant computational resource requirements present a challenge. Moreover, its commercial nature may give rise to ethical and data security considerations, impeding smooth integration into local electronic health record systems. We aim to investigate alternative lightweight and open-source foundation models to address these issues while maintaining comparable performance. Second, our evaluation of FHIR-GPT was confined to medication-related FHIR resources, potentially limiting the generalizability of our findings. Future work will include extending these methods to a broader range of FHIR resources. Third, our current approach primarily involves prompt engineering through trial-and-error with existing LLMs without enhancement of the architecture of the foundational models. To improve accuracy and reasoning in the transformation process, we aim to adopt techniques such as chain-of-thoughts, retrieval-augmented generation, or LLM agents in future endeavors.<sup>32</sup> Finally, we recognize that LLMs can assist in other FHIR-related transformations such as FHIR version upgrades and tabular-to-FHIR transformations. Although we perceive these tasks as potentially less complex than our text-to-FHIR transformations, we encourage fellow researchers to explore the efficacy of LLMs in these alternative pathways to enhance interoperability.

## Conclusions

This study lays the groundwork for harnessing LLMs to improve health data interoperability through the transformation of free-text input into the FHIR resources. The FHIR-GPT model not only streamlines the process, but also enhances transformation accuracy compared with existing NLP pipelines. Building upon these results, our future investigations will expand to encompass additional FHIR resources, aiming to advance the practical applications of LLMs in enhancing health data interoperability.

## Supplementary Material

Refer to Web version on PubMed Central for supplementary material.

## Disclosures

This work was supported by the National Institutes of Health (grants R01LM013337 and U01TR003528) and by an American Heart Association Predoctoral Fellowship (grant number 23PRE1010660).

## References

1. Office of the National Coordinator for Health Information Technology (ONC). Interoperability (<https://www.healthit.gov/topic/interoperability>).

2. Office of the National Coordinator for Health Information Technology. U.S. Core Data for Interoperability (USCDI) and USCDI+ Quality. November 29, 2023 (<https://mmshub.cms.gov/sites/default/files/MMS-InfoSession-USCDI-Slides-20231129.pdf>).
3. Centers for Disease Control and Prevention. Public health surveillance and data. Advancing interoperability for public health. December 27, 2023 (<https://www.cdc.gov/surveillance/policy-standards/interoperability.html>).
4. CMS Health Informatics and Interoperability Group (HIIG). Federal interoperability. April 25, 2024 (<https://www.cms.gov/priorities/key-initiatives/burden-reduction/interoperability/federal-interoperability>).
5. HL7.org. FHIR overview. March 26, 2023 (<https://hl7.org/fhir/overview.html>).
6. Bauer DC, Metke-Jimenez A, Maurer-Stroh S, et al. Interoperable medical data: the missing link for understanding COVID-19. *Transbound Emerg Dis* 2021;68:1753–1760. DOI: 10.1111/tbed.13892. [PubMed: 33095970]
7. Brandt PS, Pacheco JA, Rasmussen LV. Development of a repository of computable phenotype definitions using the clinical quality language. *JAMIA Open* 2021;4:o0ab094. DOI: 10.1093/jamiaopen/o0ab094.
8. Zong N, Sharma DK, Yu Y, et al. Developing a FHIR-based framework for phenome wide association studies: a case study with a pan-cancer cohort. *AMIA Jt Summits Transl Sci Proc* 2020;2020:750–759. [PubMed: 32477698]
9. Metke-Jimenez A, Hansen D. FHIRCap: transforming REDCap forms into FHIR resources. *AMIA Jt Summits Transl Sci Proc* 2019;2019:54–63. [PubMed: 31258956]
10. Pffiffer PB, Pinyol I, Natter MD, Mandl KD. C3-PRO: connecting ResearchKit to the health system using i2b2 and FHIR. *PLoS One* 2016;11:e0152722. DOI: 10.1371/journal.pone.0152722.
11. Reinecke I, Gulden C, Kümmel M, Nassirian A, Blasini R, Sedlmayr M. Design for a modular clinical trial recruitment support system based on FHIR and OMOP. *Stud Health Technol Inform* 2020;270: 158–162. DOI: 10.3233/SHTI200142. [PubMed: 32570366]
12. Zong N, Stone DJ, Sharma DK, et al. Modeling cancer clinical trials using HL7 FHIR to support downstream applications: a case study with colorectal cancer data. *Int J Med Inform* 2021;145:104308. DOI: 10.1016/j.ijmedinf.2020.104308.
13. Lee H-A, Kung H-H, Lee Y-J, et al. Global infectious disease surveillance and case tracking system for COVID-19: development study. *JMIR Med Inform* 2020;8:e20567. DOI: 10.2196/20567.
14. Wang X, Lehmann H, Botsis T. Can FHIR support standardization in post-market safety surveillance? *Stud Health Technol Inform* 2021;281:33–37. DOI: 10.3233/SHTI210115. [PubMed: 34042700]
15. Dash S, Shakyawar SK, Sharma M, Kaushik S. Big data in healthcare: management, analysis and future prospects. *J Big Data* 2019;6:1–25. DOI: 10.1186/s40537-019-0217-0.
16. Hong N, Wen A, Shen F, et al. Developing a scalable FHIR-based clinical data normalization pipeline for standardizing and integrating unstructured and structured electronic health record data. *JAMIA Open* 2019;2:570–579. DOI: 10.1093/jamiaopen/ooz056. [PubMed: 32025655]
17. Savova GK, Masanz JJ, Ogren PV, et al. Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES): architecture, component evaluation and applications. *J Am Med Inform Assoc* 2010;17:507–513. DOI: 10.1136/jamia.2009.001560. [PubMed: 20819853]
18. Sohn S, Clark C, Halgrim SR, Murphy SP, Chute CG, Liu H. MedXN: an open source medication extraction and normalization tool for clinical text. *J Am Med Inform Assoc* 2014;21:858–865. DOI: 10.1136/amiadjnl-2013-002190. [PubMed: 24637954]
19. Wang J, Mathews WC, Pham HA, Xu H, Zhang Y. Opioid2FHIR: a system for extracting FHIR-compatible opioid prescriptions from clinical text. 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). Piscataway, NJ: IEEE, 2020:1748–1751.
20. Google Cloud. Use the Healthcare Natural Language API. June 20, 2024 (<https://cloud.google.com/healthcare-api/docs/how-tos/nlp>).
21. Microsoft. Azure Health Data Services (<https://azure.microsoft.com/en-us/products/health-data-services>).

22. HL7.org. Resource MedicationStatement — Content (<https://build.fhir.org/medicationstatement.html>).
23. Henry S, Buchan K, Filannino M, Stubbs A, Uzuner O. 2018 n2c2 Shared task on adverse drug events and medication extraction in electronic health records. *J Am Med Inform Assoc* 2020;27:3–12. DOI: 10.1093/jamia/ocz166. [PubMed: 31584655]
24. SNOMED International. SNOMED International SNOMED CT Browser (<https://browser.ihtsdotools.org/>).
25. Zeng K, Bodenreider O, Kilbourne J, Nelson SJ. RxNav: a web service for standard drug information. *Proceedings of AMIA Annual Symposium 2006*. Washington, DC: American Medical Informatics Association, 2006: 1156.
26. HL7.org. Validate resources (<https://validator.fhir.org>).
27. Achiam J, Adler S, Agarwal S, et al. GPT-4 technical report. March 15, 2023 (<https://arxiv.org/abs/2303.08774>). Preprint.
28. Touvron H, Martin L, Stone K, et al. Llama 2: open foundation and fine-tuned chat models. July 18, 2023 (<https://arxiv.org/abs/2307.09288>). Preprint.
29. Penedo G, Malartic Q, Hesslow D, et al. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. June 1, 2023 (<https://arxiv.org/abs/2306.01116>). Preprint.
30. Frantar E, Ashkboos S, Hoefer T, Alistarh D. GPTQ: accurate post-training quantization for generative pre-trained transformers. October 31, 2022 (<https://arxiv.org/abs/2210.17323>). Preprint.
31. Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: expert-curated oncology reports to advance language model inference. *NEJM AI* 2024;1(4). DOI: 10.1056/AIdbp2300110.
32. Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits reasoning in large language models. *Adv Neural Inf Process Syst* 2022;35:24824–24837.

![Workflow diagram showing the transformation from Free Text to FHIR Resources. It starts with 'Free-text input' and 'Prompts' as inputs. 'Free-text input' goes through 'n2c2 Annotation' and 'Existing NLP Pipelines' (NLP2FHIR16, Google HNL API20) to produce 'Entity Annotations23'. 'Prompts' go through 'Large Language Models' (FHIR-GPT, Llama-2-70B29, Falcon 180B28) to produce 'Our Annotation'. Both 'Entity Annotations23' and 'Our Annotation' feed into 'Transformation' to produce 'FHIR Resources'. 'FHIR Resources' are then generated by 'Large Language Models'.](dbe553cf16dd14073b89a8263a428664_img.jpg)

**A Discharge Summary**

Discharge Summary:  
Discharge Medications:  
...  
[7. senna 8.6 mg Tablet Sig: One (1) Tablet PO BID P.R.N. Constipation]  
...  
Patient was discharged to long-term care facility.

**B Entity Annotations**

| Medication | senna        |
|------------|--------------|
| Reason     | Constipation |
| Route      | PO           |
| Timing     | BID          |
| Dose       | One (1)      |
| Form       | Tablet       |
| Strength   | 8.6 mg       |
| asNeeded   | P.R.N.       |

**C FHIR MedicationStatement**

```

{
  'resourceType': 'MedicationStatement',
  'id': '100035T133',
  'subject': {'reference': 'hadm_id100035'},
  'medication': {'reference': '#med100035T133'},
  'reason': [
    {
      'concept': {'text': 'Constipation',
        'coding': [{'system': 'http://snomed.info/sct',
          'code': '14760008',
          'display': 'Constipation'}]}],
  'dosage': [
    {
      'route': {'text': 'PO',
        'coding': [{'system': 'http://snomed.info/sct',
          'code': '26643006',
          'display': 'Oral route'}]},
      'timing': {'repeat': {'frequency': 2, 'period': 1.0, 'periodUnit': 'd'},
        'code': {'coding': [{'system': 'http://terminology.hl7.org/',
          'code': 'BID',
          'display': 'BID'}]}},
      'asNeeded': True,
      'doseAndRate': [{'doseQuantity': {'value': 1.0}}]},
  'contained': [
    {
      'resourceType': 'Medication',
      'id': 'med100035T133',
      'code': {'coding': [
        {'system': 'National Drug Code',
          'code': '00904516561',
          'display': 'sennosides, USP 8.6 MG Oral Tablet'},
        {'system': 'RxNorm',
          'code': '312935',
          'display': 'sennosides, USP 8.6 MG Oral Tablet'},
        {'system': 'senna 8.6 mg Tablet'},
      ]},
    {
      'doseForm': {'text': 'Tablet',
        'coding': [{'system': 'http://snomed.info/sct',
          'code': '385055001',
          'display': 'Tablet'}]},
      'ingredient': [{'item': {'concept': {'text': 'senna'}}}],
      'strengthQuantity': {
        'value': 8.6, 'unit': 'milligram',
        'system': 'http://unitsofmeasure.org/',
        'code': 'mg'}]}]
  }

```

**D Prompts for LLMs**

**[INSTRUCTIONS]**  
You are a helpful assistant that can help with medication data extraction. User will paste a short narrative that describes the administration of a drug. Please extract the drug route (How drug should enter body), e.g. PO, IV. < *Collapsed for more instructions* >

**[TEMPLATE]**  
{"text": "<string>", // the original text mention of drug route  
"coding": []//optional, but MUST lookup from the table below  
{"system": "http://snomed.info/sct",  
"code": "<code>", # SNOMED code  
"display": "<display>" # the display of the code}}

**[EXAMPLES]**  
For example, the narrative  
"Oxycodone-Acetaminophen 5-325 mg Tablet  
Sig: 1-2 Tablets PO/nQ4-6H (every 4 to 6 hours) as needed"  
You should return a jsonformat:  
{"text": "PO", "coding": [{"system": "http://snomed.info/sct", "code": "26643006", "display": "Oral route"}]}  
< *Collapsed for 4 more examples* >

**[TERMINOLOGIES]**  
Code Display  
6064005 Topical route  
10547007 Oticroute  
< *Collapsed for 143 more SNOMED CT Codes* >

**E Workflow**

```

graph TD
    subgraph Inputs
        direction TB
        A[Free-text input]
        B[Prompts]
    end
    A --> C[n2c2 Annotation]
    A --> D[Existing NLP Pipelines]
    B --> E[Large Language Models]
    C --> F[Entity Annotations23]
    D --> F
    E --> G[Our Annotation]
    F --> H[Transformation]
    G --> H
    H --> I[FHIR Resources]
    E --> J[Generation]
    I --> J
  
```

Workflow diagram showing the transformation from Free Text to FHIR Resources. It starts with 'Free-text input' and 'Prompts' as inputs. 'Free-text input' goes through 'n2c2 Annotation' and 'Existing NLP Pipelines' (NLP2FHIR16, Google HNL API20) to produce 'Entity Annotations23'. 'Prompts' go through 'Large Language Models' (FHIR-GPT, Llama-2-70B29, Falcon 180B28) to produce 'Our Annotation'. Both 'Entity Annotations23' and 'Our Annotation' feed into 'Transformation' to produce 'FHIR Resources'. 'FHIR Resources' are then generated by 'Large Language Models'.

**Figure 1.**

Overview of the Transformation from Free Text to FHIR Resource.

Example of a snippet from the discharge summary, which is the free-text input for the Fast Healthcare Interoperability Resources (FHIR) resource generation (Panel A). The National NLP Clinical Challenges (n2c2) expert annotation of medication-related entities in the discharge summary (Panel B). Example of the transformed FHIR MedicationStatement resource based on our annotations, serving as the ground truth transformation (Panel C). The same color shading from Panel B is used. Example prompt used to instruct large

language models (LLMs) in generating FHIR resources (Panel D). The workflow details how we annotated the dataset and compared the performance of LLMs with existing natural language processing (NLP) pipelines in transforming free-text inputs into associated FHIR resources (Panel E). HNL API denotes Healthcare Natural Language application programming interface; BID, twice daily; IV, intravenously; PO, orally; and PRN, as needed.

Author Manuscript

Author Manuscript

Author Manuscript

Author Manuscript

Descriptions, Examples, and Statistics of Human Annotation for the FHIR MedicationStatement Resource.\*  
**Table 1.**

| Medication Statement Elements | Type              | Card. | Example                                                                                                                                                                                                                                                                                                                         | Description                                      | Code Set                                 | N (%)          | N, Unique Entries | N, Unique Codes                       |
|-------------------------------|-------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|------------------------------------------|----------------|-------------------|---------------------------------------|
| Identifier                    | String            | 1..1  | 1000357133                                                                                                                                                                                                                                                                                                                      | External identifier                              | MIMIC+h2e2                               | 3671 (100)     | 3671              | 3671                                  |
| Subject                       | CodeableReference | 1..1  | {'reference': 'hadm_id/64366'}                                                                                                                                                                                                                                                                                                  | Who is/was taking the medication                 | MIMIC                                    | 3671 (100)     | 280               | 280                                   |
| Medication                    | CodeableConcept   | 0..1  | {'coding': [{'system': 'NDC', 'code': '51079088120', 'display': 'clonazepam 0.5 MG Oral Tablet'}, {'system': 'RxNORM', 'code': '197527', 'display': 'Clonazepam 500 microgram oral tablet'}, {'system': 'SNOMED', 'code': '322897008', 'display': 'Clonazepam 500 microgram oral tablet'}, {'text': 'clonazepam 0.5 mg Tablet'} | What medication                                  | NDC<br>RxNorm<br>SNOMED CT<br>Medication | 3671 (100)     | 1762              | NDC:625<br>RxNorm: 520<br>SNOMED: 210 |
| medication                    | CodeableConcept   | 0..1  | {'text': 'Tablet', 'coding': [{'system': 'SNOMED', 'code': '385055001', 'display': 'Tablet'}]}                                                                                                                                                                                                                                  | Powder   tablets   capsule +                     | SNOMED CT<br>Dose Form                   | 1478<br>(40.5) | 176               | 26                                    |
| ingredient. Strength          | Quantity          | 0..1  | {'value': 0.5, 'unit': 'milligram', 'code': 'mg', 'system': 'http://unitsofmeasure.org'}                                                                                                                                                                                                                                        | Quantity of ingredient presents                  | unitsofmeasure.org                       | 2383<br>(64.9) | 188               | 16                                    |
| Reason                        | CodeableConcept   | 0..*  | [{'concept': {'text': 'headache', 'coding': [{'system': 'SNOMED', 'code': '25064002', 'display': 'Headache'}]}]                                                                                                                                                                                                                 | Reason for why the medication is being/was taken | SNOMED CT<br>Finding                     | 1106<br>(30.1) | 619               | 354                                   |
| Dosage                        | 0..*              |       |                                                                                                                                                                                                                                                                                                                                 | Take "as needed"                                 |                                          | 3671 (100)     | 2                 |                                       |
| asNeeded                      | Boolean           | 0..1  | True                                                                                                                                                                                                                                                                                                                            | How medication enters the body                   | SNOMED CT<br>Route of Admin.             | 2011<br>(54.8) | 64                | 15                                    |
| route                         | CodeableConcept   |       | {'text': 'PO', 'coding': [{'system': 'SNOMED', 'code': '26643006', 'display': 'Oral route'}]}                                                                                                                                                                                                                                   | Timing schedule                                  | hl7.org/fhir/                            | 2393<br>(65.2) | 177               | 6                                     |
| timing.repeat                 | Element           | 0..1  | {'frequency': 1, 'period': 4.0, 'periodMax': 6.0, 'periodUnit': 'h', 'duration': 3.0, 'durationUnit': 'd'}                                                                                                                                                                                                                      | Code for timing schedule (e.g., "BID")           | hl7.org/fhir/                            | 2287<br>(62.5) | 17                | 17                                    |
| timing.code                   | CodeableConcept   | 0..1  | {'coding': [{'system': 'HL7', 'code': 'Q4H', 'display': 'Q4H'}]}                                                                                                                                                                                                                                                                | Amount or range of medication per dose           |                                          | 1378<br>(37.5) | 53                |                                       |
| dose-Quantity                 | Quantity          | 0..1  | {'doseQuantity': {'value': 5.0, 'unit': 'ML'}}                                                                                                                                                                                                                                                                                  |                                                  |                                          |                |                   |                                       |
| dose-Range                    | Range             | 0..1  | {'doseRange': {'low': {'value': 1.0}, 'high': {'value': 3.0}}}                                                                                                                                                                                                                                                                  |                                                  |                                          | 11 (0.30)      | 7                 |                                       |

\* FHIR denotes Fast Healthcare Interoperability Resources—generative pretrained transformer.

Table 2.

Comparison of LLMs and Existing NLP Pipelines for Transforming Free-Text Input into FHIR MedicationStatement Resources.\*

| Elements of MedicationStatement | LLMs                   |                           |                           | Existing NLP Pipelines |                              |  |
|---------------------------------|------------------------|---------------------------|---------------------------|------------------------|------------------------------|--|
|                                 | FHIR-GPT <sup>27</sup> | Falcon 180B <sup>28</sup> | Llama-2-70B <sup>29</sup> | NLP2FHIR <sup>6†</sup> | Google HNL API <sup>20</sup> |  |
| Medication                      |                        |                           |                           |                        |                              |  |
| medication                      | 0.968                  | 0.899                     | 0.859                     | 0.862                  | 0.963                        |  |
| doseForm                        | 0.976                  | 0.790                     | 0.633                     | 0.556                  | —                            |  |
| ingredient.Strength             | 0.980                  | 0.921                     | 0.792                     | —                      | —                            |  |
| Reason                          | 0.902                  | 0.593                     | 0.169                     | 0.645                  | —                            |  |
| Dosage                          |                        |                           |                           |                        |                              |  |
| route                           | 0.902                  | 0.457                     | 0.516                     | —                      | 0.871                        |  |
| timing.repeat                   | 0.947                  | 0.268                     | 0.221                     | 0.403                  | —                            |  |
| timing.code                     | 0.952                  | 0.818                     | 0.600                     | 0.424                  | —                            |  |
| doseQuantity/Range              | 0.973                  | 0.864                     | 0.823                     | 0.724                  | 0.854                        |  |

\* Performance is evaluated by using the exact match rate, which requires that the resources generated by the models precisely match human annotations in all aspects, including structure, codes, and cardinality. The best-performing model for each element is indicated in bold, and the second-place model is in italics. FHIR denotes Fast Healthcare Interoperability Resources; FHIR-GPT, Fast Healthcare Interoperability Resources–generative pretrained transformer; HNL API, Healthcare Natural Language application programming interface; LLMs, large language models; and NLP, natural language programming.

† Due to version and implementation differences, the existing NLP pipelines cannot generate all the elements included in our dataset. They have therefore been left as blanks (dashes).

**Table 3.**  
Reproducibility in Using LLMs for Transforming FHIR Resources.\*

| Elements of MedicationStatement | LLMs                   |            |                           |            |                           |            |
|---------------------------------|------------------------|------------|---------------------------|------------|---------------------------|------------|
|                                 | FHIR-GPT <sup>27</sup> |            | Falcon 180B <sup>28</sup> |            | Llama-2-70B <sup>29</sup> |            |
|                                 | Sept 2023              | March 2024 | Sept 2023                 | March 2024 | Sept 2023                 | March 2024 |
| Medication                      |                        |            |                           |            |                           |            |
| medication                      | 0.968                  | 0.970      | 0.899                     | 0.897      | 0.859                     | 0.849      |
| doseForm                        | 0.976                  | 0.974      | 0.790                     | 0.785      | 0.633                     | 0.634      |
| ingredient.Strength             | 0.980                  | 0.981      | 0.921                     | 0.912      | 0.792                     | 0.792      |
| Reason                          | 0.902                  | 0.908      | 0.593                     | 0.617      | 0.169                     | 0.172      |
| Dosage                          |                        |            |                           |            |                           |            |
| route                           | 0.902                  | 0.944      | 0.457                     | 0.471      | 0.516                     | 0.518      |
| timing.repeat                   | 0.947                  | 0.938      | 0.268                     | 0.264      | 0.221                     | 0.218      |
| timing.code                     | 0.952                  | 0.946      | 0.818                     | 0.810      | 0.600                     | 0.580      |
| doseQuantity/Range              | 0.973                  | 0.972      | 0.864                     | 0.864      | 0.823                     | 0.814      |

\* Two separate experiments were conducted 6 months apart using the same prompts, with identical model weights used for the Falcon and Llam-2a models. Fast Healthcare Interoperability Resources–generative pretrained transformer (FHIR-GPT) used the gpt-4-32k model in the September 2023 experiment, which was upgraded to the gpt-4-turbo (128k) model in March 2024. LLMs denotes large language models; and Sept, September.

**Table 4.**  
Discrepancies between FHIR-GPT–Generated Resources and Human Annotations.\*

| Category                 | Explanation                                                                                                   | Input Example                                                                                                                                                                           | Expected Output                                                                                                             | Generated Output                                                                                                            | N (%)     |
|--------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-----------|
| False negative           | Despite the presence of drug route information in the input, FHIR-GPT overlooks it                            | 2 ml of 100 U/ml heparin (200 units heparin) each lumen Daily and PRN                                                                                                                   | {'text': 'each lumen', 'coding': [{'system': 'snomed.info/sct', 'code': '47625008', 'display': 'Intravenous route'}]}       | {}                                                                                                                          | 95 (45.7) |
|                          |                                                                                                               | PEs and HIT at the referring institution, and was continued on an argatroban gtt                                                                                                        | {'text': 'gtt'}                                                                                                             | {}                                                                                                                          |           |
| False positive           | FHIR-GPT fabricates drug route information that does not exist in the input                                   | Pneumonia was suspected and patient was started on vancomycin                                                                                                                           | {}                                                                                                                          | {'text': 'IV', 'coding': [{'system': 'snomed.info/sct', 'code': '47625008', 'display': 'Intravenous route'}]}               | 62 (29.8) |
|                          |                                                                                                               | Joint pain: medication side effect (IVIG, hydralazine)                                                                                                                                  | {}                                                                                                                          | {'text': 'IVIG', 'coding': [{'system': 'snomed.info/sct', 'code': '47625008', 'display': 'Intravenous route'}]}             |           |
| Mismatched error         | Although FHIR-GPT generates drug route resource, it does not align with the actual data provided in the input | Heparin lock flush (porcine) [heparin lock flush] 10 U/ml 2 ml to PICC line Flush daily                                                                                                 | {'text': 'PICC line Flush', 'coding': [{'system': 'snomed.info/sct', 'code': '417989007', 'display': 'Intraductal route'}]} | {'text': 'PO', 'coding': [{'system': 'snomed.info/sct', 'code': '284009009', 'display': 'Route of Administration values'}]} | 46 (22.1) |
|                          |                                                                                                               | Artificial Tears 1–2 DROP OU PRN                                                                                                                                                        | {'text': 'OU', 'coding': [{'system': 'snomed.info/sct', 'code': '54485002', 'display': 'Ophthalmic route'}]}                | {'text': 'DROP', 'coding': [{'system': 'snomed.info/sct', 'code': '372473007', 'display': 'Oromucosal use'}]}               |           |
| Syntax error             | The generated content is not in a valid FHIR resource or JSON format                                          | COPD flare with vancomycin 1 gm IV                                                                                                                                                      | {'text': 'IV', 'coding': [{'system': 'snomed.info/sct', 'code': '47625008', 'display': 'Intravenous route'}]}               | {'text': 'IV', 'coding': [{'system': 'snomed.info/sct', 'code': '47625008', 'display': 'Intravenous route'}]}               | 4 (1.9)   |
| Content filter rejection | FHIR-GPT fails to generate content due to the input failing to pass the Azure content filter                  | ... he developed delirium ... which manifested as inappropriate and sometimes violent actions with pt attempting to hit staff and spitting on staff. ... pt had to be given haloperidol | {}                                                                                                                          | NA                                                                                                                          | 1 (0.5)   |

\* A total of 204 disagreements in transforming routes were categorized into five types of errors. The percentage represents the proportion of this type of error among the total 204 errors. COPD denotes chronic obstructive pulmonary disease; FHIR-GPT, Fast Healthcare Interoperability Resources–generative pretrained transformer; gtt, drops (from Latin 'guttæ'); HIT, heparin-induced thrombocytopenia; IVIG, intravenous immunoglobulin; NA, not applicable; PEs, pulmonary embolisms; PICC, peripherally inserted central catheter; pt, patient; and PRN, as needed.