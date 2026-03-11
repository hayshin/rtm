

# LLM-based speaker diarization correction: A generalizable approach

Georgios Efsthathiadis<sup>a,b</sup>, Vijay Yadav<sup>a,c</sup>, Anzar Abbas<sup>a</sup>

<sup>a</sup>*Brooklyn Health, 11201 USA, Brooklyn, NY, USA*

<sup>b</sup>*Department of Biostatistics, Harvard T. H. Chan School of Public Health, 02115 USA, Boston, MA, USA*

<sup>c</sup>*School of Psychology, University of New South Wales, 2052 Australia, Sydney, NSW, Australia*

## --- Abstract

Speaker diarization is necessary for interpreting conversations transcribed using automated speech recognition (ASR) tools. Despite significant developments in diarization methods, diarization accuracy remains an issue. Here, we investigate the use of large language models (LLMs) for diarization correction as a post-processing step. LLMs were fine-tuned using the Fisher corpus, a large dataset of transcribed conversations. The ability of the models to improve diarization accuracy in a holdout dataset from the Fisher corpus as well as an independent dataset was measured. We report that fine-tuned LLMs can markedly improve diarization accuracy. However, model performance is constrained to transcripts produced using the same ASR tool as the transcripts used for fine-tuning, limiting generalizability. To address this constraint, an ensemble model was developed by combining weights from three separate models, each fine-tuned using transcripts from a different ASR tool. The ensemble model demonstrated better overall performance than each of the ASR-specific models, suggesting that a generalizable and ASR-agnostic approach may be achievable. We have made the weights of these models publicly available on HuggingFace at <https://huggingface.co/bklynhlth>.

*Keywords:* speaker diarization, speech enhancement/separation, multi-speaker, large language model

---

*Email addresses:* `georgios.efsthathiadis@brooklyn.health` (Georgios Efsthathiadis), `vijay.yadav@brooklyn.health` (Vijay Yadav), `anzar@brooklyn.health` (Anzar Abbas)

## --- 1. Introduction

Diarization refers to the identification of unique speakers in a conversation [1, 2]. It is often a component of automated speech recognition (ASR) tools [3, 4]. It is necessary in various contexts, such as medical transcriptions [5, 6], where separating patient and clinician speech is necessary for interpretation. Diarization accuracy is impacted by several factors such as audio quality, environmental noise, variability in speaker behavior, and overlapping speech [7, 8, 9]. Here, we present a method that uses a fine-tuned large language model (LLM) to improve diarization accuracy in conversational transcripts.

Several speaker diarization methods exist [10, 11]. Traditionally, clustering-based approaches have been widely used. One such method is the x-vector approach [12], which extracts speaker embeddings, or x-vectors, from fixed-length segments of audio using a time-delay neural network. The embeddings are clustered to assign speaker labels. Building on these foundations, earlier neural network-based methods further improved diarization performance. A popular open source method is Pyannote [13], which uses pre-trained neural networks to assign speaker labels through analysis of acoustic data. It processes audio into overlapping segments, extracting features such as mel-frequency cepstral coefficients to detect changes in speakers.

More recently, the field has shifted towards end-to-end diarization models [14, 15, 16, 17, 18, 19] which integrate speaker recognition with transcription, allowing for simultaneous transcription and diarization. These models map audio inputs directly to speaker-attributed transcriptions. Fujita et al. [14] introduced a self-attention mechanism to enhance the End-to-End Neural Diarization (EEND) model [20], showing improved performance over traditional BLSTM-based methods by directly optimizing the diarization error rate (DER), which significantly reduces errors in overlapping speech scenarios. Zhang et al. [15] proposed a fully supervised diarization system named unbounded interleaved-state recurrent neural network (UIS-RNN), which replaces traditional unsupervised clustering modules with an online generative process, leading to significant improvements in diarization error rates, particularly when trained with high-quality, time-stamped data. Kanda et al. proposed the transcribe-to-diarize model [16], which applies a neural network that processes audio to provide both the transcribed text and the speaker labels in one pass with the ability to estimate the start and end times of

![Diagram illustrating the process of speaker diarization correction. On the left, a conversation transcript is shown with incorrect speaker labels: spk1: 'hey are you free this', spk2: 'weekend yes why are', spk1: 'you', spk2: 'asking let's', and spk1: 'go hiking'. An arrow labeled 'Diarization correction' points to the right, where the corrected transcript is shown: spk1: 'hey are you free this weekend', spk2: 'yes why are you asking', and spk1: 'let's go hiking'. Above the arrow is a small icon of a hand writing on a notepad with a pencil.](b230b8f21d8e82d55c0d311c8c32ef73_img.jpg)

Diagram illustrating the process of speaker diarization correction. On the left, a conversation transcript is shown with incorrect speaker labels: spk1: 'hey are you free this', spk2: 'weekend yes why are', spk1: 'you', spk2: 'asking let's', and spk1: 'go hiking'. An arrow labeled 'Diarization correction' points to the right, where the corrected transcript is shown: spk1: 'hey are you free this weekend', spk2: 'yes why are you asking', and spk1: 'let's go hiking'. Above the arrow is a small icon of a hand writing on a notepad with a pencil.

Figure 1: Accurate speaker diarization is necessary for interpretation of important conversations.

each word by incorporating a minimal set of learnable parameters into the model’s internal state.

Diarization is often a component of popular ASR tools such as those offered by Amazon Web Services (AWS), Azure, or Google Cloud Platform (GCP). These ASRs tools have various distinguishing features such as transcription accuracy, language support, inference speed, and – relevant to our manuscript – the diarization method employed. While some popular ASR tools such as WhipserX are open source [21], others do not disclose details on diarization methods. In a given use case, depending on the ASR tool used, differences in diarization accuracy may influence downstream findings and interpretation [22, 23]. Where addressing lack of visibility into underlying methods may not be practical, methods to improve and consequently converge diarization accuracy in transcripts from different ASR tools would allow for ASR-agnostic applications.

Several methods have been explored to improve speaker diarization accuracy. Han et al. 2024 [24] proposes a network architecture to improve speaker diarization from an initial diarization attempt. Their model incorporates two parallel encoder networks: one processes the initial diarization results; the other analyzes the log-mel frequency spectrum of the audio. These are integrated by a decoder that merges the output embedding from both encoders. Paturi et al. 2023 [25] uses a similar approach. They fine tune an encoder network that utilizes the embedding. From a pre-trained language model combined with the original speaker ID embeddings to recalibrate the speaker IDs. Duke et al. 2022 [26] employs a theoretical approach by modeling the probability of an utterance being assigned to a speaker using Bayesian inference. The probability is calculated based on the semantic distance between an utterance and its preceding utterances.

Other methods have utilized LLMs to improve speaker diarization. Wang et al. 2024 [27] established the fine-tuning framework employed in this manuscript. They fine-tune PaLM 2-S [28] to correct speaker diarization mistakes from GCP’s Universal Speech Model [29], which uses Turn-to-Diarize [30] for speaker diarization. Their manuscript shows that the approach can notably enhance speaker diarization accuracy. Park et al. 2024 [31] fine-tuned an LLM to correct speaker diarization of an acoustic-only ASR tool, a version of the Multi-Scale Diarization Decoder model [32]. They combine LLM predictions with acoustic information to build a beam search decoding approach that showed improvement in speaker labeling. Adedji et al. 2024 [33] used pre-trained LLMs in a multi-step approach to improve speaker diarization in medical transcripts. By using a separate component to transcribe the text and an LLM to label speakers, they showed that using chain-of-thought speaker prompts can diarization ASR-transcribed text and match or surpass the performance of integrated ASR tools that conduct both transcription and diarization.

Here, we fine-tune LLMs to improve diarization accuracy in ASR transcripts as a post-processing step, visualized in Figure 1. To fine-tune our models, we used the English Fisher corpus [34], a large audio dataset containing 1,960 hours of phone conversations, and to test them we used a held out set of the Fisher corpus and the PriMock57 dataset [35] a smaller audio dataset containing mock primary care consultations. We consider the effects of using transcripts from different ASRs, tuning separate models for each AWS, Azure, and WhisperX transcript. Wang et al. (2024) demonstrated notable improvements in speaker diarization accuracy through fine-tuning. Our work investigates the generalizability of this method across different ASR systems. Specifically, we identify a key limitation in previous approaches: the variability in performance when applied to different ASRs. We find that LLM-based diarization correction models are most effective when correcting transcripts produced using the same ASR as the transcripts they were fine-tuned with, limiting their generalizability. To address this constraint, we build an ensemble model that combines the weights from each of the individual models, and demonstrate that such a model has the potential to serve as a reliable and ASR-agnostic diarization correction tool.

This model has been published on HuggingFace and is available at <https://huggingface.co/bklynhlth/WillisDiarize-v1>. We additionally released a function using the methods described here for a friendlier user interface on applying diarization correction in OpenWillis [36], a Python library for

![Flowchart of the LLM evaluation pipeline. The process starts with 'ASR output (JSON)', which is converted to a 'JSON Transcript' and then an 'ASR transcript'. This transcript is segmented into 'ASR transcript - segment 1', 'ASR transcript - segment 2', ..., 'ASR transcript - segment n'. These segments are used for 'Prompt Generation' to create 'Prompt 1', 'Prompt 2', ..., 'Prompt n'. These prompts are fed into a 'Fine-tuned LLM' to produce 'Completion 1', 'Completion 2', ..., 'Completion n'. The completions are then processed by the 'Completion Parser' stage. This stage includes 'Completion Transcript Extraction' to get 'Extracted words, labels 1', 'Extracted words, labels 2', ..., 'Extracted words, labels n'. These are then processed by 'TPST (transfer to original words)' to get 'Mapped labels 1', 'Mapped labels 2', ..., 'Mapped labels n'. Finally, these are combined into 'Corrected Labels', which are used to produce the 'Corrected ASR output'. The 'Data Preprocessing' stage is enclosed in a dashed box, and the 'Completion Parser' stage is also enclosed in a dashed box.](f9a14fbfecbd7d059226cc93677d721b_img.jpg)

```

graph TD
    A([ASR output (JSON)]) --> B[JSON Transcript]
    B --> C[ASR transcript]
    C --> D[ASR transcript - segment 1]
    C --> E[ASR transcript - segment 2]
    C --> F[ASR transcript - segment n]
    subgraph Data Preprocessing
        D --> G[Prompt Generation]
        E --> G
        F --> G
        G --> H[Prompt 1]
        G --> I[Prompt 2]
        G --> J[Prompt n]
    end
    H --> K[Fine-tuned LLM]
    I --> K
    J --> K
    K --> L[Completion 1]
    K --> M[Completion 2]
    K --> N[Completion n]
    subgraph Completion Parser
        L --> O[Completion Transcript Extraction]
        M --> O
        N --> O
        O --> P[Extracted words, labels 1]
        O --> Q[Extracted words, labels 2]
        O --> R[Extracted words, labels n]
        P --> S[TPST transfer to original words]
        Q --> S
        R --> S
        S --> T[Mapped labels 1]
        S --> U[Mapped labels 2]
        S --> V[Mapped labels n]
        T --> W[Corrected Labels]
        U --> W
        V --> W
    end
    W --> X([Corrected ASR output])
  
```

Flowchart of the LLM evaluation pipeline. The process starts with 'ASR output (JSON)', which is converted to a 'JSON Transcript' and then an 'ASR transcript'. This transcript is segmented into 'ASR transcript - segment 1', 'ASR transcript - segment 2', ..., 'ASR transcript - segment n'. These segments are used for 'Prompt Generation' to create 'Prompt 1', 'Prompt 2', ..., 'Prompt n'. These prompts are fed into a 'Fine-tuned LLM' to produce 'Completion 1', 'Completion 2', ..., 'Completion n'. The completions are then processed by the 'Completion Parser' stage. This stage includes 'Completion Transcript Extraction' to get 'Extracted words, labels 1', 'Extracted words, labels 2', ..., 'Extracted words, labels n'. These are then processed by 'TPST (transfer to original words)' to get 'Mapped labels 1', 'Mapped labels 2', ..., 'Mapped labels n'. Finally, these are combined into 'Corrected Labels', which are used to produce the 'Corrected ASR output'. The 'Data Preprocessing' stage is enclosed in a dashed box, and the 'Completion Parser' stage is also enclosed in a dashed box.

Figure 2: Overview of pipeline used to evaluate the LLM.

digital measurement of health, that can be found at <https://github.com/bklynhlth/openwillis>. An overview of the system used in a real-world application can be seen in Figure 2.

## 2. Methods

Code for all data, training and evaluation methods can be found at <https://github.com/GeorgeEfsthathiadis/LLM-Diarize-ASR-Agnostic>.

### 2.1. Data

The English Fisher corpus, available through the Linguistic Data Consortium, was used [34]. The dataset contains 11,699 audio recordings of telephone conversations between two participants. The recordings total 1,960 hours in length, averaging 10 minutes each. 11,971 participants (57% female, average age 36 years) were involved, each participating in 1.95 calls on average. Recordings in the Fisher corpus have corresponding transcripts with sentence-level timestamps. The transcribed words and speaker labels are referred to here as the **reference transcripts**. Methods for production of these transcripts can be found in Cieri et al. 2004 [34]. Data was split into a training and testing set using the same split as in past work [27, 37, 38, 39]: The training set had 11,527 recordings from 11,631 participants; the testing set had 172 recordings from 340 participants.

In addition we tested the model on an independent dataset. We used PriMock57 for evaluation [35], a dataset of 57 mock primary consultations between a clinician and a mock patient. The recordings total approximately 9 hours in length, averaging 9 minutes each. There were 7 clinicians and 57 actors portraying the patient involved. Same as in the Fisher corpus, the PriMock57 dataset had corresponding transcripts with sentence-level timestamps. Methods for how these mock consultations were conducted can be found in Korfiatis et al. 2022 [35].

### 2.2. Transcription

All recordings were transcribed into text using three ASR tools: AWS’s Transcribe [3]; Azure’s Speech to Text [4]; and WhisperX [21]. Each ASR conducts both transcription and diarization. WhisperX is a combination of Whisper, OpenAI’s open-source transcription model [40], and Pyannote’s open-source diarization model [13].

When running transcriptions through both AWS and Azure, language was prespecified as English, the maximum number of speakers was set to 2, and the option of speaker diarization was set to true. Version 3.2 of Azure’s Speech to Text API was used. When running transcriptions through WhisperX, the Whisper large-v2 model was used for transcription, Pyannote

version 3.1.1 was used for diarization, the language was prespecified as English, and the maximum number of speakers were set to 2. All transcriptions produced a JSON file for each recording, containing the transcribed text with speaker labels.

### 2.3. Data Preprocessing

Given that each JSON file had ASR-specific formatting, we extracted transcribed words and speaker labels into a common format, consistent with what was used in Wang et al. 2024 [27]. These transcripts are referred to here as the **ASR transcripts**. The ASR transcripts were further standardized by removing punctuation and transforming all text to lowercase. 11% of reference transcripts were missing transcripts for part of the recording, ranging from 30 seconds up to 8 minutes and 27 seconds. For these, we trimmed the ASR transcripts to correspond with their reference transcripts.

While all input audio recordings contained conversation of 2 speakers, in some cases ASRs may only detect 1 speaker. This can be due to poor audio quality, noise in the recording, volume of the speakers or other factors. Any ASR transcripts with only one or more than two detected speakers were removed from the dataset. Some ASR transcripts, particularly those from WhisperX, had continuously repeating word sequences [41]. For example, words such as ‘and’ or phrases such as ‘that’s right’ would repeat consecutively such that the sentence would lose meaning. To minimize the effect of repeating sequences on model fine-tuning and testing, transcripts that contained more than 10 repetitions of a sequence were removed.

To create the training data, a transcript-preserving speaker transfer algorithm (TPST) [27] was used to transfer speaker labels from the reference transcripts to the ASR transcripts shown in Figure 3. The TPST algorithm accepts speaker-labeled source text and speaker-labeled target text. It aligns the two so that the source text speaker labels match the target text. The resulting transcripts are referred to here as the **oracle transcripts**. Oracle transcripts have the same text as the ASR transcripts but the speaker labels of the reference transcripts. They allow for comparison of diarization accuracy between an ASR transcripts and the ground-truth speaker labeling in the reference transcripts by removing effects for differences in the transcription itself.

![Figure 3: Creation of oracle transcripts using the TPST algorithm. The diagram shows two input transcripts: a Reference transcript and an ASR transcript. The Reference transcript consists of three lines: 'spk1: hi i'm john', 'spk2: mike nice to meet you', and 'spk1: you as well'. The ASR transcript consists of three lines: 'spk1: hi hi i am', 'spk2: john mark glad to meet you', and 'spk1: you as well'. Both transcripts are processed by the TPST algorithm, which aligns word sequences and speaker labels. The output is an Oracle transcript with corrected speaker labels: 'spk1: hi hi i am john' and 'spk2: mark glad to meet you', while preserving the original words and the final line 'spk1: you as well'.](3121ebddccf183ca63bb9781be440a7e_img.jpg)

The diagram illustrates the TPST algorithm's process. It starts with two input transcripts: a Reference transcript and an ASR transcript. The Reference transcript contains three lines of text with speaker labels (spk1, spk2). The ASR transcript also contains three lines but with different speaker labels and some word mismatches. The TPST algorithm processes these two transcripts, aligning the word sequences and speaker labels. The output is an Oracle transcript, which has corrected speaker labels to match the Reference transcript while keeping the original words and the final line of the ASR transcript.

Figure 3: Creation of oracle transcripts using the TPST algorithm. The diagram shows two input transcripts: a Reference transcript and an ASR transcript. The Reference transcript consists of three lines: 'spk1: hi i'm john', 'spk2: mike nice to meet you', and 'spk1: you as well'. The ASR transcript consists of three lines: 'spk1: hi hi i am', 'spk2: john mark glad to meet you', and 'spk1: you as well'. Both transcripts are processed by the TPST algorithm, which aligns word sequences and speaker labels. The output is an Oracle transcript with corrected speaker labels: 'spk1: hi hi i am john' and 'spk2: mark glad to meet you', while preserving the original words and the final line 'spk1: you as well'.

Figure 3: Creation of oracle transcripts using the TPST algorithm. Words and speaker labels are extracted from each transcript. The algorithm aligns word sequences, such that the resulting speaker labels from the reference transcript match the text of the ASR transcript. This corrects speaker labeling in the ASR transcript without changing the underlying transcription.

### 2.4. Model Tuning

#### 2.4.1. Model Selection

We used Mistral AI’s Mistral 7b Instruct v0.2 as the base model for all fine-tuned models in this manuscript [42]. Compared to similar open-source large language models, Mistral 7b was chosen because of its size and relative baseline accuracy in diarization correction. The decision-making process behind model selection is presented in Appendix A.

#### 2.4.2. ASR-specific models

Given each ASR tool has a noticeably different transcription style, in terms of both words utilized and also diarization errors made, we hypothesized that a model fine-tuned on one ASR would perform best at correcting diarization for transcripts produced by the same ASR compared to transcripts produced by a different ASR. For this reason, three separate ASR-specific models were fine-tuned using AWS, Azure, and WhisperX transcripts.

Fine-tuning involved using the ASR transcripts in the prompt and the oracle transcripts in the completion, using the same method as Wang et al. 2024 [27]. The fine-tuning prompt and completion template is shown below; the exact template can be found in <https://github.com/GeorgeEfsthathiadis/LLM-Diarize-ASR-Agnostic>.

##### **Prompt:**

In the speaker diarization transcript below, some words are potentially misplaced. Please correct those words and move them to the right speaker. Directly show the corrected transcript without explaining what changes were made or why you made those changes:

[ASR transcript]

##### **Completion:**

[Oracle transcript]

Though the context length of Mistral 7b is 32 thousand tokens, we segmented transcripts in the training set to achieve prompt-completion pairs totaling 8,192 tokens. This was because smaller segments ensure efficient memory usage and align better with practical use-case scenarios.

We fine-tuned each of the three ASR-specific models using Amazon SageMaker on an ml.g5.8xlarge instance (1 NVIDIA A10G Tensor Core GPU with 24 GiB of memory) for 2 epochs, with a batch size of 6 and 5 gradient accumulation steps. We used Quantized Low-Ranked Adaption of Language Models [43] and Flash attention [44] for efficient fine-tuning, balancing computational resources with model performance improvement. The three resulting fine-tuned models are referred to here as the **AWS model**, **Azure model**, and **WhisperX model**, or collectively as the **ASR-specific models**.

#### 2.4.3. Ensemble model

To develop an ASR-agnostic ensemble model, TIES-Merging [45] from merge-kit [46] was used to combine parameters from each ASR-specific model. This computes a weighted average of each parameter, such that only the most significant changes of each parameter are used, with the rest set back to the base model value, and conflicts between different models are solved by using the most significant change across different versions. For hyperparameters, we set all models to approximately equal weight (AWS: 0.34, Azure: 0.33, WhisperX: 0.33) and the density for each model to 0.8.

### 2.5. Completion parser

A completion parser was developed to post-process the model output. It extracts only the relevant part from the model output, removing pre- and post-fixes if present, extracts speaker labels from the model output, and transfers those labels back to input text. Since LLMs can hallucinate, it is common that the LLM may alter original wording in the transcript in addition to speaker labeling. To mitigate this, the TPST algorithm—the same algorithm as seen in creating the oracle transcripts during training—was used to ensure speaker labels outputted by the model matched the original wording. This is done by transferring the completion speaker labels to the input ASR transcription wording, thus mitigating word changes made by the LLM in the completion wording. This completion parser was integrated as part of the model’s methods to ensure no word substitutions or modifications were made through use of the model.

### 2.6. Model evaluation

#### 2.6.1. Measurement of accuracy

Diarization accuracy was measured using two established metrics: delta concatenated minimum-permutation word error rate (deltaCP) [47] and delta speaker-attributed word error rate (deltaSA) [48]. For each, word error rate (WER) first needed to be calculated. WER was measured as the percentage of transcription errors divided by the total number of words. When measuring accuracy in this manuscript, deltaCP and deltaSA were calculated with the reference transcripts serving as ground truth. Together, deltaCP and deltaSA quantify how much additional error is introduced by speaker labeling, allowing us to differentiate between transcription errors and those stemming from misattributed speakers.

To calculate deltaCP, concatenated minimum-permutation word error rate (cpWER) was first calculated. To do this, each speaker’s words were separated into their own transcripts. For each possible permutation of the speaker’s words, the WER was calculated against the reference speaker transcript. The smallest observed WER for each of the two speakers was averaged. cpWER is considered the minimum possible WER among all permutations of the two speakers. deltaCP is simply cpWER subtracted by the original WER (i.e.  $\text{deltaCP} = \text{cpWER} - \text{WER}$ ). The difference allows us to analyze errors introduced by speaker labeling, independent of the original WER.

To calculate deltaSA, speaker-attributed word error rate (SA-WER) was first calculated. To do this, each speaker’s words were separated into their own transcripts. WER was calculated against the reference speaker-transcripts. The WER of each of the two speakers was averaged. deltaSA is simply SA-WER subtracted by the WER (i.e.  $\text{deltaSA} = \text{SA-WER} - \text{WER}$ ).

#### 2.6.2. Assessing performance

Diarization correction was performed on the ASR transcripts in the testing set. This was done separately for transcripts from each of the three ASRs. The resulting transcripts are referred to here as the **corrected transcripts**. Before diarization correction, transcripts were segmented into smaller chunks to meet the 4,096 token threshold (8,192 tokens for combined prompt-completion).

Model performance was assessed by calculating deltaCP and deltaSA before and after diarization correction. Seven different models for diarization correction were used:

1. Mistral 7b, with no fine-tuning
2. Mistral 8x7b, with no fine-tuning [49]
3. DiarizationLM 8b Fisher v2 [27]
4. Mistral 7b fine-tuned on AWS transcripts (AWS model)
5. Mistral 7b fine-tuned on Azure transcripts (Azure model)
6. Mistral 7b fine-tuned on WhisperX transcripts (WhisperX model)
7. The ensemble model built using the three ASR-specific models

DiarizationLM 8b Fisher v2 is the best performing model from the models that are made openly available, proposed by Wang et al. 2024 [27]. The framework they employed is the one we based our training strategy on for our ASR-specific models. The base model used was Llama3 with 8b parameters which is similar in size with the base model we used which had 7b parameters.

When assessing performance of the fine-tuned models, the same prompt from model training was used. For the zero-shot performance assessment i.e. for Mistral 7b and Mistral 8x7b with no-finetuning, the following prompt was used:

##### **Prompt:**

<s> [INST]

In the speaker diarization transcript below, some words are potentially misplaced. Please correct those words and move them to the right speaker. Directly show the corrected transcript without explaining what changes were made or why you made those changes:

[ASR transcript] [/INST]

Here is the corrected transcript with the words moved to the right speaker:

In addition, when evaluating DiarizationLM 8b Fisher v2 which was proposed and open-sourced in Wang et al. 2024 [27], the following prompt template was used (as shown at <https://huggingface.co/google/DiarizationLM-8b-Fisher-v2>):

##### **Prompt:**

[ASR transcript] →

## 3. Results

### 3.1. *Transcripts used*

After removing transcripts with only one or more than two detected speakers, we were left with 11,519 transcripts in the AWS training set (8 removed), 11,525 transcripts in the Azure training set (2 removed), and 11,464 transcripts in the WhisperX training set (63 removed).

After splitting the transcripts into smaller segments to fit the 4,096 token context window, the training set included 52,293 AWS transcripts, 58,192 Azure transcripts, and 53,997 WhisperX transcripts, each with a corresponding oracle transcript.

Then, the transcripts with repeating word sequences were removed. This was done after the segmentation to avoid removing more data than necessary. The final training set included 52,287 AWS transcripts (6 removed), 58,184

Azure transcripts (8 removed), and 52,982 WhisperX transcripts (1,015 removed).

The same steps were carried out on the Fisher testing set and the PriMock57 dataset. After preprocessing, for the Fisher testing set we were left with 172 AWS transcripts, 172 Azure transcripts (both AWS and Azure did not have any transcripts removed), and 162 WhisperX transcripts (1 removed due to a single speaker and 9 removed due to repeating word sequences). While for the PriMock57 dataset we were left with 57 AWS transcripts (0 removed), 57 Azure transcripts (0 removed) and 43 WhisperX transcripts (14 removed due to single or more than 2 speakers).

### 3.2. Word error rate

We looked at WER across the three ASRs. The WER across ASRs is shown in Table 1. The PriMock57 dataset had considerably better audio quality, given the mock sessions were carefully conducted in an isolated environment, which explains the improved transcription WER compared to the Fisher test set. Azure’s Speech to Text model performed best in terms of transcription accuracy in the Fisher test set, while AWS’s Transcribe performed the best in the PriMock57 dataset. WhisperX performed the worst in both datasets.

### 3.3. Baseline diarization accuracy

Diarization accuracy was compared between ASR transcripts and corresponding reference transcripts without any diarization correction. The baseline diarization accuracy across ASRs is shown in Table 1. In the Fisher dataset, AWS performed best, while Azure was next and WhisperX performed worst. On the PriMock57 dataset Azure performed the best and AWS was next. WhisperX performed worst on the PriMock57 dataset as well. The difference in diarization performance between AWS’s Transcribe and Azure’s Speech to Text is indicative of the fact that different ASRs may work better for different use cases.

Table 1: Baseline transcription and diarization accuracy measures across ASRs.

|                          | AWS transcripts |         |       | Azure transcripts |         |       | WhisperX transcripts |         |       |
|--------------------------|-----------------|---------|-------|-------------------|---------|-------|----------------------|---------|-------|
|                          | deltaCP         | deltaSA | WER   | deltaCP           | deltaSA | WER   | deltaCP              | deltaSA | WER   |
| <b>Fisher test set</b>   | 0.93            | 2.5     | 22.04 | 1.91              | 3.06    | 16.99 | 4.46                 | 5.77    | 22.39 |
| <b>PriMock57 dataset</b> | 2.18            | 3.56    | 12.43 | 0.82              | 1.97    | 14.28 | 4.96                 | 6.56    | 18.69 |

### 3.4. Zero-shot model performance

The performance of the Mistral 7b and Mistral 7x8b models in improving diarization accuracy was measured against baseline, i.e., transcripts with no diarization correction. This was done using the Fisher test dataset only. The results are shown in Table 2.

Table 2: Zero-shot evaluation of base models with no fine-tuning on correction of diarization errors in the Fisher test set.

|              | AWS transcripts |          |         |          | Azure transcripts |          |         |          | WhisperX transcripts |          |         |          |
|--------------|-----------------|----------|---------|----------|-------------------|----------|---------|----------|----------------------|----------|---------|----------|
|              | deltaCP         |          | deltaSA |          | deltaCP           |          | deltaSA |          | deltaCP              |          | deltaSA |          |
|              | Value           | $\Delta$ | Value   | $\Delta$ | Value             | $\Delta$ | Value   | $\Delta$ | Value                | $\Delta$ | Value   | $\Delta$ |
| Baseline     | 0.93            |          | 2.5     |          | 1.91              |          | 3.06    |          | 4.46                 |          | 5.77    |          |
| Mistral 7b   | 16.53           | +1677%   | 19.28   | +671%    | 16.85             | +782%    | 19.43   | +535%    | 21.26                | +377%    | 24.09   | +318%    |
| Mistral 7x8b | 12.01           | +1191%   | 14.44   | +478%    | 12.88             | +574%    | 15      | +390%    | 17.5                 | +292%    | 19.94   | +246%    |

The base models, with no fine-tuning, i.e., the zero-shot models, instead of improving diarization accuracy, significantly reduced diarization accuracy compared to baseline.

### 3.5. Fine-tuned model performance

The performance of the ASR-specific models and the ensemble model in improving diarization accuracy was measured against baseline in the Fisher test set. We also evaluated DiarizationLM 8b Fisher v2 [27]. The results are shown in Table 3.

Table 3: Evaluation of fine-tuned models on correction of diarization errors in the Fisher test set. ASR-specific models performed best on transcripts produced using the same ASR as the transcripts used for fine-tuning. The ensemble model performed best across the board, regardless of the ASR used to produce the transcript.

|                | AWS transcripts |          |         |          | Azure transcripts |          |         |          | WhisperX transcripts |          |         |          |
|----------------|-----------------|----------|---------|----------|-------------------|----------|---------|----------|----------------------|----------|---------|----------|
|                | deltaCP         |          | deltaSA |          | deltaCP           |          | deltaSA |          | deltaCP              |          | deltaSA |          |
|                | Value           | $\Delta$ | Value   | $\Delta$ | Value             | $\Delta$ | Value   | $\Delta$ | Value                | $\Delta$ | Value   | $\Delta$ |
| Baseline       | 0.93            |          | 2.5     |          | 1.91              |          | 3.06    |          | 4.46                 |          | 5.77    |          |
| DiarizationLM  | 3.94            | +324%    | 5.71    | +128%    | 5.97              | +213%    | 7.3     | +139%    | 6.05                 | +36%     | 7.46    | +29%     |
| AWS model      | 0.5             | -46%     | 2.05    | -18%     | 1.3               | -32%     | 2.41    | -22%     | 4.11                 | -8%      | 5.39    | -7%      |
| Azure model    | 1.04            | +12%     | 2.6     | +4%      | 0.87              | -54%     | 1.92    | -37%     | 3.89                 | -13%     | 5.12    | -11%     |
| WhisperX model | 1.71            | +84%     | 3.44    | +38%     | 1.46              | -24%     | 2.57    | -16%     | 3.37                 | -24%     | 4.61    | -20%     |
| Ensemble model | 0.63            | -32%     | 2.15    | -14%     | 0.82              | -57%     | 1.88    | -39%     | 3.15                 | -29%     | 4.31    | -25%     |

DiarizationLM 8b Fisher v2 is unable to improve diarization performance in unseen ASRs. Fine-tuned models markedly improved diarization accuracy compared to baseline in most cases.

As hypothesized, each of the ASR-specific models improved diarization accuracy most significantly on transcripts that were produced using the same ASR tool as the transcripts in the training set. The ASR-specific models did not perform as well on transcripts derived from a different ASR, with some worsening diarization accuracy.

The ensemble model achieved the best overall performance. In the AWS transcripts, it had marginally worse performance compared to the AWS model. However, in the Azure and WhisperX transcripts, it performed better than even the ASR-specific models for Azure and WhisperX. Ablation studies on ensembles of two ASR-specific models indicate that while partial improvements can be achieved by combining subsets of ASR-specific models, the full three-model ensemble provides the best generalization across different ASR outputs (see Appendix C for details).

### 3.6. Performance on an independent dataset

We also evaluated the fine-tuned models on the PriMock57 dataset. The results are shown in Table 4.

Table 4: Evaluation of fine-tuned models on correction of diarization errors in the PriMock57 dataset. ASR-specific models performed best on transcripts produced using the same ASR as the transcripts used for fine-tuning in most cases. The ensemble model performed best across the board, regardless of the ASR used to produce the transcript.

|                | AWS transcripts |             |             |             | Azure transcripts |             |             |             | WhisperX transcripts |             |             |             |
|----------------|-----------------|-------------|-------------|-------------|-------------------|-------------|-------------|-------------|----------------------|-------------|-------------|-------------|
|                | deltaCP         |             | deltaSA     |             | deltaCP           |             | deltaSA     |             | deltaCP              |             | deltaSA     |             |
|                | Value           | $\Delta$    | Value       | $\Delta$    | Value             | $\Delta$    | Value       | $\Delta$    | Value                | $\Delta$    | Value       | $\Delta$    |
| Baseline       | 2.18            |             | 3.56        |             | 0.82              |             | 1.97        |             | 4.96                 |             | 6.56        |             |
| AWS model      | 1.79            | -18%        | 3.1         | -13%        | 0.52              | -37%        | 1.63        | -17%        | 4.64                 | -6%         | 6.11        | -7%         |
| Azure model    | 1.35            | -38%        | 2.59        | -27%        | 0.46              | -44%        | 1.55        | -21%        | 3.18                 | -36%        | 4.53        | -31%        |
| WhisperX model | 1.33            | -39%        | 2.56        | -28%        | 0.52              | -37%        | 1.62        | -18%        | <b>2.31</b>          | <b>-53%</b> | <b>3.61</b> | <b>-45%</b> |
| Ensemble model | <b>1.32</b>     | <b>-39%</b> | <b>2.56</b> | <b>-28%</b> | <b>0.37</b>       | <b>-55%</b> | <b>1.44</b> | <b>-27%</b> | 2.68                 | -46%        | 3.99        | -39%        |

As seen previously in the Fisher test set, the ASR-specific models from Azure and WhisperX performed the best in the datasets transcribed with the same ASR. The exception was the AWS-specific model which performed the worst in the AWS dataset across ASR-specific models.

The ensemble model achieved the best overall performance in this dataset as well. In the WhisperX transcripts, it had comparable performance to the WhisperX model, which performed the best. However, in the Azure and AWS transcripts, it performed better than all ASR-specific models.

## 4. Discussion

Our findings demonstrate that fine-tuning is necessary for LLM-based correction of speaker diarization, as demonstrated previously in Wang et al. 2024 [27]. Zero-shot model performance was poor, leading to worse diarization accuracy than present at baseline. This may be attributable to a lack of specific adaptation to the task at hand and the varied characteristics of each ASR output. Performance may be improved through use of examples in the prompt, few-shot techniques, or other prompt engineering techniques, as demonstrated in [33].

Though fine-tuning led to improved performance, this performance was constrained to transcripts obtained from the same ASR as the transcripts that were used for fine-tuning. This demonstrates that transcripts from different ASR tools are sufficiently varied to impact model performance. The nature of this variance is beyond the scope of this manuscript. However, qualitative observation showed that different ASR tools had different types of errors, e.g., one ASR tool would frequently show inaccurate speaker labeling at the end of sentences, whereas another would mislabel small phrases in the middle of longer speech segments.

In examining the results reported in Wang et al. 2024 [27] (using USM + turn-to-diarize as their ASR), their best performing model (PaLM 2-S finetuned (hyp2ora)) achieves an impressive 75% improvement in deltaCP, which significantly outperforms our ASR-specific models. However the best performing open-source model (Llama 3 8B finetuned (mixed) v2; named DiarizationLM 8b Fisher v2 in HuggingFace) achieves approximately 49% improvement in deltaCP, which is more in line with the performance gains seen in our AWS and Azure models when evaluated on their own test sets, that show 46% and 54% improvements respectively.

However, when evaluating DiarizationLM 8b Fisher v2 on transcriptions from unseen ASRs (using our test sets transcribed with AWS, Azure, and WhisperX) the model fails to generalize, leading to a deterioration in diarization performance. This decline in accuracy may be attributed to overfitting to source ASR, large differences in ASR transcription or differences in the

amount of diarization errors. Notably, the WhisperX model, which shows the least degradation in performance, had the most comparable baseline diarization error ( $\text{deltaCP} = 4.46$ ) to the ASR used in Wang et al. ( $\text{deltaCP} = 5.71$ ), suggesting that diarization similarity plays a key role in the model’s ability to maintain performance.

The performance of the fine-tuned models was influenced by the distribution of speaker diarization errors in the training dataset. When evaluated on an independent dataset, the model’s performance could be negatively affected if the quantity of diarization errors differed significantly from the training dataset, even if transcribed by the same ASR. For instance, the AWS model exhibited the smallest performance improvement on the AWS-transcribed PriMock57 dataset compared to other models. This was likely because AWS-transcribed PriMock57 had a substantially higher baseline rate of speaker diarization errors ( $\text{deltaCP}: 2.18$ ,  $\text{deltaSA}: 3.56$ ) than the AWS-transcribed Fisher test set ( $\text{deltaCP}: 0.93$ ,  $\text{deltaSA}: 2.5$ ). Further analysis revealed that the AWS-specific model became more conservative in the number of speaker changes it implemented, which hindered its performance when evaluated on a dataset with higher error rates.

The ensemble model, with combined weights from each of the individual ASR-specific models, demonstrated better overall performance across ASRs. Evidence suggests the ensemble model may have better adaptability to the different types of speaker mislabeling present across ASRs. Hence, it may be taking a more balanced approach in the amount of speaker corrections it should produce. In the case of correcting diarization errors in the Azure and WhisperX Fisher test transcripts, the ensemble model performed better than the ASR-specific models. It may be that different ASR platforms have common error types present in different distributions. A model that is aware of all such types would perform better than an ASR-specific model tuned to identify the more commonly occurring error types in its ASR.

Improved accuracy through the ensemble model was achieved without compromising on inference time. The merging approach did not change the original architecture. Hence, the combined model had the same size as the individual models. We believe an additional advantage of the ensemble model will be that it may generalize better in transcripts from unfamiliar ASRs, as investigated in Appendix B. This will be useful in real-world applications, where a system may be constrained to a specific cloud platform or have a preference for specific ASR tools. The model may be able to improve diarization accuracy as a post-processing step regardless of the ASR tool

used to produce the transcript.

We also show that the model is useful on an independent dataset. Notably, this was a clinical dataset, which highlights a major application of such a model i.e. clinical transcriptions. Given the efforts to automate clinical note-taking [50, 51, 52], such a model could improve those tools significantly, particularly if future efforts involve further fine-tuning specific to conversations that consider the use case.

We acknowledge certain limitations, opening up opportunities for future work. First, a more thorough analysis of the diarization errors produced by each ASR would clarify the need for generalized approaches and help us understand better the types of errors each LLM is more probable to correct. In particular, future work should explore how the results of different ASR models are similar or different, and how LLMs use these variations to improve diarization performance. Especially as it becomes easier to quantify these errors, investigating these differences could provide deeper insights into model adaptability and generalization across ASR outputs. Second, the project was limited to transcripts in English. This restricts diarization correction to a single language. Future work could test this model on transcripts in different languages and also include additional languages in the dataset used for fine-tuning. Third, our experimentation with varied prompts was not extensive [53]. Given we were building on work from Wang et al. 2024 [27], we used the same prompts utilized in their work. Future work exploring the transferability of the model could include contextual information (e.g., this is a conversation between a doctor and a patient; this is a phone conversation between a salesperson and a customer) [54]. Finally, though our project focused on such models serving as post-processing steps, integrating multimodal data will allow for more robust diarization systems. Combining acoustic information with semantic insights from the LLM will enhance the system’s ability to label speech more effectively [31], especially in complex and noisy environments.

## 5. Conclusion

We investigated the use of fine-tuned LLMs to improve speaker diarization in conversational transcripts with existing speaker labels. Our findings demonstrate that LLMs can markedly improve diarization accuracy when specifically trained to do so. However, if such training is specific to one ASR tool or using a single dataset, it may constrain the model’s perfor-

mance to transcripts only from that ASR tool and with similar diarization error rates. An ensemble model with knowledge across ASR tools allows for a generalizable and ASR-agnostic diarization correction tool. We believe such a model can be a useful post-processing step in applications that depend on transcription and diarization of conversations with multiple speakers. We have made these models publicly available at HuggingFace at <https://huggingface.co/bklynhlth> and built a Python function for easier use of the model available at <https://github.com/bklynhlth/openwillis>.

## 6. Acknowledgements

Though referenced throughout the manuscript, the authors would like to acknowledge the authors of Wang et al. 2024 [27], whose work formed the foundation of this manuscript. Much of the code used during this work, including the TPST algorithm, critical to model fine-tuning and testing, was originally developed by their group and kindly shared with us for replication.

## 7. Author contributions

**Georgios Efstathiadis:** Conceptualization, Methodology, Software, Validation, Writing - Original Draft, Visualization **Vijay Yadav:** Data curation, Software, Investigation, Writing - Review & Editing **Anzar Abbas:** Writing - Review & Editing, Supervision, Project administration

## Appendix A. Evaluation of zero-shot pre-trained LLMs

To determine the optimal model for fine-tuning, we conducted an evaluation of various LLMs using a small sample from the test set. A mix of 15 Fisher test transcriptions was randomly selected, consisting of 5 transcriptions from each ASR (AWS, Azure and WhisperX). After preprocessing and segmenting these transcriptions into manageable prompt-completion pairs, we generated 64 pairs. We focused on small open-source models (fewer than 13 billion parameters), to keep the fine-tuning process computationally feasible and to ensure our results could be made accessible.

We evaluated the performance of Llama2 Chat models (7b and 13b sizes) and Mistral Instruct (7b size v0.2). The following prompt formats were used for each model:

For the Llama2 models:

#### **Prompt:**

<s>[INST] <<SYS>>

In the speaker diarization transcript below, some words are potentially misplaced. Please correct those words and move them to the right speaker. Directly show the corrected transcript without explaining what changes were made or why you made those changes:

<</SYS>>

[ASR transcript] [/INST]

Here is the corrected transcript with the words moved to the right speaker:

For the Mistral model:

#### **Prompt:**

<s>[INST]

In the speaker diarization transcript below, some words are potentially misplaced. Please correct those words and move them to the right speaker. Directly show the corrected transcript without explaining what changes were made or why you made those changes:

[ASR transcript] [/INST]

Here is the corrected transcript with the words moved to the right speaker:

The results are presented in Table A.5.

Despite the Llama2 13b model demonstrating the best performance, we opted to proceed with the Mistral 7b model for fine-tuning. The Mistral 7b model requires less computational power, making it a more cost-effective solution. Although the Llama2 13b model showed superior performance metrics, the marginal gains did not justify the significantly higher resource demands, especially in a production environment where efficiency and scalability are crucial. By choosing Mistral 7b, we aimed to maintain a high standard of

Table A.5: Zero-shot evaluation of smaller versions of Llama2 and Mistral base models with no fine-tuning on correction of diarization errors.

|            | Mixed transcripts |         |       |
|------------|-------------------|---------|-------|
|            | deltaCP           | deltaSA | WER   |
| Baseline   | 0.76              | 2.59    | 27.81 |
| Llama2 7b  | 21.13             | 23.29   | -     |
| Llama2 13b | 11.79             | 14.52   | -     |
| Mistral 7b | 15.08             | 17.75   | -     |

performance while optimizing for ease of deployment.

## Appendix B. Generalization to unseen ASR

To assess the generalization capabilities of our models, we transcribed the test set using a fourth ASR, without fine-tuning an ASR-specific model on its training data. We transcribed the Fisher test set using the Google Cloud Platform (GCP) transcription service. For the transcription we used the GCP speech to text API (speech\_v1p1beta) with the *latest\_long* model enabling speaker diarization capabilities on, setting the maximum speakers to 2 and specifying the language to English. We filtered the testing data by excluding transcriptions with a single speaker (excluded 8) or those with repeated words issue (excluded 0), resulting in 164 transcriptions.

We evaluated all our fine-tuned expert models and the ASR-specific model on the GCP test data, with the results shown in Table B.6.

Table B.6: Evaluation of fine-tuned models on correction of diarization errors in the GCP Fisher test set.

|                | GCP transcripts |          |         |          |
|----------------|-----------------|----------|---------|----------|
|                | deltaCP         | $\Delta$ | deltaSA | $\Delta$ |
| Baseline       | 1.83            |          | 2.98    |          |
| AWS model      | 1.82            | -1%      | 2.98    | -0%      |
| Azure model    | 1.93            | +5%      | 3.09    | +4%      |
| WhisperX model | 2.04            | +11%     | 3.21    | +8%      |
| Ensemble model | 1.72            | -6%      | 2.86    | -4%      |

The ASR-specific fine-tuned models did not show any improvement in performance. In fact the Azure and WhisperX models deteriorated baseline performance, while the AWS model achieved a marginal 1% improvement in deltaCP and 0% improvement in deltaSA. On the other hand, the ensemble

model demonstrated a 6% improvement in deltaCP and a 4% improvement in deltaSA.

The GCP test set exhibited a WER of 25.88, the worst transcription performance compared to the other ASRs, leading to higher text incoherence. This likely contributed to the marginal improvements in speaker diarization observed in this analysis.

Despite the small improvement shown by the ensemble model, it represented a significant relative improvement compared to the results from the individual experts that comprised the ensemble. This indicates that while no ASR-specific model was effective at correcting speaker diarization in GCP transcribed data, the ensemble model was able to generalize better and improve performance in an unseen transcription service, indicating the potential of our model to be applied across various ASR services.

## Appendix C. Ensemble model - Ablation study

The ensemble model was constructed by combining all three ASR-specific models (AWS, Azure, and WhisperX). Each individual model was fine-tuned on transcripts generated by a specific ASR, making them specialized for correcting diarization errors inherent to that ASR. The ensemble approach aimed to leverage the strengths of all three models to improve generalization across different ASRs.

To further analyze the contribution of each ASR-specific model to the ensemble’s performance, we conducted an ablation study evaluating ensembles composed of only two ASR-specific models at a time. The following combinations were tested:

- AWS + Azure
- AWS + WhisperX
- Azure + WhisperX

Each of these two-model ensembles was evaluated on the Fisher test set from all three ASRs to determine their effectiveness compared to the ASR-specific models and the full three-model ensemble. The results are shown in Table C.7.

The two-ASR ensembles performed better in the datasets which were included in their training sets compared to the third left out ASR, in some

Table C.7: Evaluation of ASR-specific and ensemble models on correction of diarization errors in the Fisher test set. Two-ASR ensemble models showed partial improvements but lacked full generalization. The full ensemble model, incorporating all three ASR-specific models, achieved the best overall performance across different ASRs, demonstrating improved robustness in diarization correction.

|                             | AWS transcripts |          |         |          | Azure transcripts |          |         |          | WhisperX transcripts |          |         |          |
|-----------------------------|-----------------|----------|---------|----------|-------------------|----------|---------|----------|----------------------|----------|---------|----------|
|                             | deltaCP         |          | deltaSA |          | deltaCP           |          | deltaSA |          | deltaCP              |          | deltaSA |          |
|                             | Value           | $\Delta$ | Value   | $\Delta$ | Value             | $\Delta$ | Value   | $\Delta$ | Value                | $\Delta$ | Value   | $\Delta$ |
| <b>Baseline</b>             | 0.93            |          | 2.5     |          | 1.91              |          | 3.06    |          | 4.46                 |          | 5.77    |          |
| <b>AWS model</b>            | 0.5             | -46%     | 2.05    | -18%     | 1.3               | -32%     | 2.41    | -22%     | 4.11                 | -8%      | 5.39    | -7%      |
| <b>Azure model</b>          | 1.04            | +12%     | 2.6     | +4%      | 0.87              | -54%     | 1.92    | -37%     | 3.89                 | -13%     | 5.12    | -11%     |
| <b>WhisperX model</b>       | 1.71            | +84%     | 3.44    | +38%     | 1.46              | -24%     | 2.57    | -16%     | 3.37                 | -24%     | 4.61    | -20%     |
| <b>AWS+Azure model</b>      | 0.54            | -42%     | 2.04    | -18%     | 0.91              | -52%     | 1.99    | -35%     | 3.45                 | -23%     | 4.64    | -20%     |
| <b>AWS+WhisperX model</b>   | 0.55            | -41%     | 2.01    | -20%     | 0.97              | -49%     | 2.05    | -33%     | 3.01                 | -33%     | 4.17    | -28%     |
| <b>Azure+WhisperX model</b> | 1               | +8%      | 2.54    | +2%      | 0.77              | -60%     | 1.81    | -41%     | 2.95                 | -34%     | 4.08    | -29%     |
| <b>Ensemble model</b>       | 0.63            | -32%     | 2.15    | -14%     | 0.82              | -57%     | 1.88    | -39%     | 3.15                 | -29%     | 4.31    | -25%     |

cases even surpassing the full ensemble model. This further underscores the challenge of achieving broad generalizability. Notably, the Azure+WhisperX ensemble performed the best in both Azure and WhisperX transcripts but failed to generalize to AWS transcripts, where it not only underperformed but also degraded baseline performance. This suggests that AWS transcripts exhibit distinctive characteristics compared to Azure and WhisperX transcripts, which appear to be more similar to each other. These structural differences likely contributed to the observed performance degradations.

Overall, the full ensemble model (AWS + Azure + WhisperX) provided a more balanced approach to diarization correction, reducing overfitting to any single ASR’s error patterns and ensuring better adaptability across different ASR outputs. Interestingly, the AWS+WhisperX ensemble demonstrated comparable or even superior performance in most cases, suggesting that the significant disparity between AWS and WhisperX transcripts in baseline diarization performance (deltaCP: 0.93 vs. 4.46; deltaSA: 2.5 vs. 5.77) may have encouraged complementary error correction.

These findings highlight the importance of incorporating diverse ASR-specific models into the ensemble that can compensate for each other’s weaknesses. While two-ASR ensembles offered performance gains over individual models, they lacked the full generalization benefits observed in the three-model ensemble. Future work could explore alternative fusion methods and incorporating more ASR-specific models to further enhance diarization cor-

rection in unseen ASR scenarios

## References

- [1] X. Anguera, S. Bozonnet, N. Evans, C. Fredouille, G. Friedland, O. Vinyals, Speaker Diarization: A Review of Recent research, *IEEE transactions on audio, speech and language processing/IEEE transactions on audio, speech, and language processing* 20 (2012) 356–370. URL: <https://doi.org/10.1109/tasl.2011.2125954>. doi:10.1109/tasl.2011.2125954.
- [2] S. Tranter, K. Yu, D. Reynolds, G. Evermann, D. Kim, P. C. Woodland, An investigation into the the interactions between speaker diarisation systems and automatic speech transcription, *Cambridge University Publications* (2003). URL: <http://publications.eng.cam.ac.uk/327929/>.
- [3] Speech to text - Amazon Transcribe - AWS, 2024. URL: <https://aws.amazon.com/transcribe/>.
- [4] Speech to Text – Audio to text Translation — Microsoft Azure, 2024. URL: <https://azure.microsoft.com/en-us/products/ai-services/speech-to-text>.
- [5] G. P. Finley, E. Edwards, A. Robinson, N. Sadoughi, J. Fone, M. Miller, D. Suendermann-Oeft, M. Brenndoerfer, N. Axtmann, An automated assistant for medical scribes., *Interspeech 2018* (2018) 3212–3213. URL: <https://dblp.uni-trier.de/db/conf/interspeech/interspeech2018.html#FinleyERSFOSBA18>.
- [6] B. Mirheidari, D. Blackburn, K. Harkness, T. Walker, A. Venneri, M. Reuber, H. Christensen, Toward the Automation of Diagnostic Conversation Analysis in Patients with Memory Complaints, *Journal of Alzheimer’s disease* 58 (2017) 373–387. URL: <https://pubmed.ncbi.nlm.nih.gov/28436388/>. doi:10.3233/jad-160507.
- [7] O. R. Khazaleh, L. A. Khrais, An investigation into the reliability of speaker recognition schemes: analysing the impact of environmental factors utilising deep learning techniques, *Journal of Engineering and Applied Science* 71 (2024). URL: <https://doi.org/10.1186/s44147-023-00351-0>. doi:10.1186/s44147-023-00351-0.

- [8] K. W. Godin, J. H. L. Hansen, Physical task stress and speaker variability in voice quality, EURASIP Journal on Audio, Speech and Music Processing 2015 (2015). URL: <https://doi.org/10.1186/s13636-015-0072-7>. doi:10.1186/s13636-015-0072-7.
- [9] D. Charlet, C. Barras, J. Liénard, Impact of overlapping speech detection on speaker diarization for broadcast news and debates, 2013 IEEE International Conference on Acoustics, Speech and Signal Processing (2013). URL: <https://doi.org/10.1109/icassp.2013.6639163>. doi:10.1109/icassp.2013.6639163.
- [10] T. J. Park, N. Kanda, D. Dimitriadis, K. J. Han, S. Watanabe, S. Narayanan, A review of speaker diarization: Recent advances with deep learning, Computer speech & language 72 (2022) 101317. URL: <https://doi.org/10.1016/j.csl.2021.101317>. doi:10.1016/j.csl.2021.101317.
- [11] L. Serafini, S. Cornell, G. Morrone, E. Zovato, A. Brutti, S. Squartini, An experimental review of speaker diarization methods with application to two-speaker conversational telephone speech recordings, Computer speech & language 82 (2023) 101534. URL: <https://doi.org/10.1016/j.csl.2023.101534>. doi:10.1016/j.csl.2023.101534.
- [12] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, S. Khudanpur, X-Vectors: robust DNN embeddings for speaker recognition, ICASSP 2018 (2018). URL: <https://doi.org/10.1109/icassp.2018.8461375>. doi:10.1109/icassp.2018.8461375.
- [13] H. Bredin, R. Yin, J. M. Coria, G. Gelly, P. Korshunov, M. Lavechin, D. Fustes, H. Titeux, W. Bouaziz, M.-P. Gill, Pyannote.Audio: Neural Building Blocks for Speaker Diarization, ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2020). URL: <https://doi.org/10.1109/icassp40776.2020.9052974>. doi:10.1109/icassp40776.2020.9052974.
- [14] Y. Fujita, N. Kanda, S. Horiguchi, Y. Xue, K. Nagamatsu, S. Watanabe, End-to-End Neural Speaker Diarization with Self-Attention, 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) (2019). URL: <https://doi.org/10.1109/asru46091.2019.9003959>. doi:10.1109/asru46091.2019.9003959.

- [15] A. Zhang, Q. Wang, Z. Zhu, J. Paisley, C. Wang, Fully supervised speaker diarization, ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2019) 6301–6305. URL: <https://doi.org/10.1109/icassp.2019.8683892>. doi:10.1109/icassp.2019.8683892.
- [16] N. Kanda, X. Xiao, Y. Gaur, X. Wang, Z. Meng, Z. Chen, T. Yoshioka, Transcribe-to-Diarize: Neural speaker diarization for unlimited number of speakers using End-to-End Speaker-Attributed ASR, ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2022). URL: <https://doi.org/10.1109/icassp43922.2022.9746225>. doi:10.1109/icassp43922.2022.9746225.
- [17] X. Zheng, C. Zhang, P. Woodland, Tandem multitask training of speaker diarisation and speech recognition for meeting transcription, Interspeech 2022 (2022). URL: <https://doi.org/10.21437/interspeech.2022-11368>. doi:10.21437/interspeech.2022-11368.
- [18] L. Meng, J. Kang, M. Cui, H. Wu, X. Wu, H. Meng, Unified Modeling of Multi-Talker Overlapped Speech Recognition and Diarization with a Sidecar Separator, Interspeech 2023 (2023). URL: <https://doi.org/10.21437/interspeech.2023-1422>. doi:10.21437/interspeech.2023-1422.
- [19] S. Cornell, J.-W. Jung, S. Watanabe, S. Squartini, One Model to Rule Them All ? Towards End-to-End Joint Speaker Diarization and Speech Recognition, HSCMA Satellite Workshop at ICASSP 2024 (2024). URL: <https://doi.org/10.1109/icassp48485.2024.10447957>. doi:10.1109/icassp48485.2024.10447957.
- [20] Y. Fujita, N. Kanda, S. Horiguchi, K. Nagamatsu, S. Watanabe, End-to-End Neural Speaker Diarization with Permutation-Free Objectives, Interspeech 2019 (2019). URL: <https://doi.org/10.21437/interspeech.2019-2899>. doi:10.21437/interspeech.2019-2899.
- [21] M. Bain, J. Huh, T. Han, A. Zisserman, WhisperX: Time-Accurate Speech Transcription of Long-Form Audio, Interspeech 2023 (2023). URL: <https://doi.org/10.21437/interspeech.2023-78>. doi:10.21437/interspeech.2023-78.

- [22] A. Ferraro, A. Galli, V. La Gatta, M. Postiglione, Benchmarking open source and paid services for speech to text: an analysis of quality and input variety, *Frontiers in big data* 6 (2023). URL: <https://doi.org/10.3389/fdata.2023.1210559>. doi:10.3389/fdata.2023.1210559.
- [23] B. Xu, C. Tao, Y. Raqui, S. Ranwez, A Benchmarking on Cloud based Speech-To-Text Services for French Speech and Background Noise Effect, *arXiv (Cornell University)* (2021). URL: <https://arxiv.org/abs/2105.03409>. doi:10.48550/arxiv.2105.03409.
- [24] J. Han, F. Landini, J. Rohdin, M. Diez, L. Burget, Y. Cao, H. Lu, J. Černocký, Diacorrect: Error Correction Back-End for Speaker Diarization, *ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (2024). URL: <https://doi.org/10.1109/icassp48485.2024.10446968>. doi:10.1109/icassp48485.2024.10446968.
- [25] R. Paturi, S. Srinivasan, X. Li, Lexical Speaker Error Correction: Leveraging Language Models for Speaker Diarization Error Correction, *INTERSPEECH 2023* (2023). URL: <https://doi.org/10.21437/interspeech.2023-1982>. doi:10.21437/interspeech.2023-1982.
- [26] R. Duke, A. Doboli, Top-down approach to solving speaker diarization errors in DiaLogic System, *2022 IEEE International Symposium on Smart Electronic Systems (iSES)* (2022). URL: <https://doi.org/10.1109/ises54909.2022.00051>. doi:10.1109/ises54909.2022.00051.
- [27] Q. Wang, Y. Huang, G. Zhao, E. Clark, W. Xia, H. Liao, DiarizationLM: Speaker Diarization Post-Processing with Large Language Models, *arXiv (Cornell University)* (2024). URL: <https://arxiv.org/abs/2401.03506>. doi:10.48550/arxiv.2401.03506.
- [28] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Paschos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, E. Chu, J. H. Clark, L. E. Shafey, Y. Huang, K. Meier-Hellstern, G. Mishra, E. Moreira, M. Omernick, K. Robinson, S. Ruder, Y. Tay, K. Xiao, Y. Xu, Y. Zhang, G. H. Abrego, J. Ahn, J. Austin, P. Barham, J. Botha, J. Bradbury, S. Brahma, K. Brooks, M. Catasta, Y. Cheng, C. Cherry, C. A. Choquette-Cho, A. Chowdhery, C. Crepy, S. Dave, M. Dehghani, S. Dev, J. Devlin, M. Díaz, N. Du, E. Dyer, V. Feinberg,

- F. Feng, V. Fienber, M. Freitag, X. Garcia, S. Gehrmann, L. Gonzalez, G. Gur-Ari, S. Hand, H. Hashemi, L. Hou, J. Howland, A. Hu, J. Hui, J. Hurwitz, M. Isard, A. Ittycheriah, M. Jagielski, W. Jia, K. Kenealy, M. Krikun, S. Kudugunta, C. Lan, K. Lee, B. Lee, E. Li, M. Li, W. Li, Y. Li, J. Li, H. Lim, H. Lin, Z. Liu, F. Liu, M. Maggioni, A. Mahendru, J. Maynez, V. Misra, M. Moussalem, Z. Nado, J. Nham, E. Ni, A. Nystrom, A. Parrish, M. Pellat, M. Polacek, A. Polozov, R. Pope, S. Qiao, E. Reif, B. Richter, P. Riley, A. C. Ros, A. Roy, B. Saeta, R. Samuel, R. Shelby, A. Slone, D. Smilkov, D. R. So, D. Sohn, S. Tokumine, D. Valter, V. Vasudevan, K. Vodrahalli, X. Wang, P. Wang, Z. Wang, T. Wang, J. Wieting, Y. Wu, K. Xu, Y. Xu, L. Xue, P. Yin, J. Yu, Q. Zhang, S. Zheng, C. Zheng, W. Zhou, D. Zhou, S. Petrov, Y. Wu, PALM 2 Technical Report, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2305.10403>. doi:10.48550/arxiv.2305.10403.
- [29] Y. Zhang, W. Han, J. Qin, Y. Wang, A. Bapna, Z. Chen, N. Chen, B. Li, V. Axelrod, G. Wang, Z. Meng, K. Hu, A. Rosenberg, R. Prabhavalkar, D. S. Park, P. Haghani, J. Riesa, G. Perng, H. Soltau, T. Strohman, B. Ramabhadran, T. Sainath, P. Moreno, C.-C. Chiu, J. Schalkwyk, F. Beaufays, Y. Wu, Google USM: Scaling Automatic Speech Recognition beyond 100 Languages, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2303.01037>. doi:10.48550/arxiv.2303.01037.
- [30] W. Xia, H. Lu, Q. Wang, A. Tripathi, Y. Huang, I. L. Moreno, H. Sak, Turn-to-Diarize: online speaker diarization constrained by transformer transducer speaker turn detection, ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2022). URL: <https://doi.org/10.1109/icassp43922.2022.9746531>. doi:10.1109/icassp43922.2022.9746531.
- [31] T. J. Park, K. Dhawan, N. Koluguri, J. Balam, Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach, ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2024). URL: <https://doi.org/10.1109/icassp48485.2024.10446204>. doi:10.1109/icassp48485.2024.10446204.

- [32] T. J. Park, N. R. Koluguri, J. Balam, B. Ginsburg, Multi-scale Speaker Diarization with Dynamic Scale Weighting, Interspeech 2022 (2022). URL: <https://doi.org/10.21437/interspeech.2022-991>. doi:10.21437/interspeech.2022-991.
- [33] A. Adedeji, S. Joshi, B. Doohan, The Sound of Healthcare: Improving Medical Transcription ASR Accuracy with Large Language Models, arXiv (Cornell University) (2024). URL: <https://arxiv.org/abs/2402.07658>. doi:10.48550/arxiv.2402.07658.
- [34] C. Cieri, D. Miller, K. Walker, The Fisher Corpus: a Resource for the Next Generations of Speech-to-Text., Language Resources and Evaluation (2004). URL: <https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/lrec2004-fisher-corpus.pdf>.
- [35] A. P. Korfiatis, F. Moramarco, R. Sarac, A. Savkov, PriMock57: A Dataset Of Primary Care Mock Consultations, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (2022). URL: <https://doi.org/10.18653/v1/2022.acl-short.65>. doi:10.18653/v1/2022.acl-short.65.
- [36] M. Worthington, G. Efsthathiadis, V. Yadav, A. Abbas, 172. OpenWillis: An Open-Source Python Library for Digital Health Measurement, Biological Psychiatry 95 (2024) S169–S170. URL: <https://doi.org/10.1016/j.biopsych.2024.02.407>. doi:10.1016/j.biopsych.2024.02.407.
- [37] Q. Wang, Y. Huang, L. Han, G. Zhao, I. L. Moreno, Highly Efficient Real-Time Streaming and Fully On-Device Speaker Diarization with Multi-Stage Clustering, arXiv (Cornell University) (2022). URL: <https://arxiv.org/abs/2210.13690>. doi:10.48550/arxiv.2210.13690.
- [38] G. Zhao, Q. Wang, L. Han, Y. Huang, I. L. Moreno, Augmenting Transformer-Transducer based speaker change detection with Token-Level training loss, ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2023). URL: <https://doi.org/10.1109/icassp49357.2023.10094955>. doi:10.1109/icassp49357.2023.10094955.

- [39] G. Zhao, Y. Wang, J. Pelecanos, Y. Zhang, H. Liao, Y. Huang, L. Han, Q. Wang, USM-SCD: Multilingual speaker change detection based on large pretrained foundation models, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2309.08023>. doi:10.48550/arxiv.2309.08023.
- [40] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever, Robust speech recognition via Large-Scale Weak Supervision, arXiv (Cornell University) (2022). URL: <https://arxiv.org/abs/2212.04356>. doi:10.48550/arxiv.2212.04356.
- [41] A. Koenecke, A. Choi, K. Mei, H. Schellmann, M. Sloane, Careless Whisper: Speech-to-Text hallucination harms, arXiv (Cornell University) (2024). URL: <https://arxiv.org/abs/2402.08021>. doi:10.48550/arxiv.2402.08021.
- [42] A. Q. Jiang, A. Sablayrolles, M. Arthur, C. Bamford, D. S. Chaplot, D. De Las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. J. Wang, T. Lacroix, W. E. Sayed, Mistral 7B, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2310.06825>. doi:10.48550/arxiv.2310.06825.
- [43] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, QLORA: Efficient Finetuning of Quantized LLMs, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2305.14314>. doi:10.48550/arxiv.2305.14314.
- [44] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré, FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, arXiv (Cornell University) (2022). URL: <https://arxiv.org/abs/2205.14135>. doi:10.48550/arxiv.2205.14135.
- [45] A. K. Yadav, D. Tam, L. Choshen, C. Raffel, M. Bansal, TIES-Merging: Resolving interference when merging models, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2306.01708>. doi:10.48550/arxiv.2306.01708.
- [46] C. Goddard, S. Siriwardhana, M. Ehghaghi, L. Meyers, V. Karpukhin, B. Benedict, M. McQuade, J. Solawetz, Arcee’s MergeKit: a

- toolkit for merging large language models, arXiv (Cornell University) (2024). URL: <https://arxiv.org/abs/2403.13257>. doi:10.48550/arxiv.2403.13257.
- [47] S. Watanabe, M. Mandel, J. Barker, E. Vincent, A. Arora, X. Chang, S. Khudanpur, V. Manohar, D. Povey, D. Raj, D. Snyder, A. S. Subramanian, J. Trmal, B. B. Yair, C. Boeddeker, Z. Ni, Y. Fujita, S. Horiguchi, N. Kanda, T. Yoshioka, N. Ryant, CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings, Proc. 6th International Workshop on Speech Processing in Everyday Environments (CHiME 2020) (2020). URL: <https://doi.org/10.21437/chime.2020-1>. doi:10.21437/chime.2020-1.
- [48] S. Cornell, M. Wiesner, S. Watanabe, D. Raj, X. Chang, P. García, Y. Masuyama, Z. Wang, S. Squartini, S. Khudanpur, The CHiME-7 DASR Challenge: Distant Meeting Transcription with Multiple Devices in Diverse Scenarios, arXiv (Cornell University) (2023). URL: <https://arxiv.org/abs/2306.13734>. doi:10.48550/arxiv.2306.13734.
- [49] A. Q. Jiang, A. Sablayrolles, A. Roux, M. Arthur, B. Savary, C. Bamford, D. S. Chaplot, D. De Las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. J. Wang, T. Lacroix, W. E. Sayed, Mixtral of experts, arXiv (Cornell University) (2024). URL: <https://arxiv.org/abs/2401.04088>. doi:10.48550/arxiv.2401.04088.
- [50] J. Wang, J. Yang, H. Zhang, H. Lu, M. Skreta, M. Husić, A. Arbabi, N. Sultanum, M. Brudno, PhenoPad: Building AI enabled note-taking interfaces for patient encounters, npj Digital Medicine 5 (2022). URL: <https://doi.org/10.1038/s41746-021-00555-9>. doi:10.1038/s41746-021-00555-9.
- [51] A. Mani, S. Palaskar, S. Konam, Towards Understanding ASR Error Correction for Medical Conversations, Proceedings of the First Workshop on Natural Language Processing for Medical Conversations (2020). URL: <https://doi.org/10.18653/v1/2020.nlpmc-1.2>. doi:10.18653/v1/2020.nlpmc-1.2.

- [52] S. V. Blackley, J. Huynh, L. Wang, Z. Korach, L. Zhou, Speech recognition for clinical documentation from 1990 to 2018: a systematic review, *Journal of the American Medical Informatics Association* 26 (2019) 324–338. URL: <https://doi.org/10.1093/jamia/ocy179>. doi:10.1093/jamia/ocy179.
- [53] P. Sahoo, A. K. Singh, S. Saha, V. Jain, S. Mondal, A. Chadha, A Systematic survey of prompt engineering in large language Models: Techniques and applications, *arXiv* (Cornell University) (2024). URL: <https://arxiv.org/abs/2402.07927>. doi:10.48550/arxiv.2402.07927.
- [54] A. Raju, B. Hedayatnia, L. Liu, A. Gandhe, C. Khatri, A. Metallinou, A. Venkatesh, A. Rastrow, Contextual Language Model Adaptation for Conversational Agents, *Interspeech* (2018). URL: <https://doi.org/10.21437/interspeech.2018-1122>. doi:10.21437/interspeech.2018-1122.