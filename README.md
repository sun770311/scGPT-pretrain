# scLLM-pretrain

## Large Language Models
Generative artificial intelligence (GenAI) is a branch of artificial intelligence that creates real-world text, images, music, and other forms of media.

A large language model (LLM), such as GPT (Generative Pretrained Transformer), is a type of generative artificial intelligence model. LLM learns from extensive data sets to process and generate natural language for tasks such as translation and conversation.

## LLMs and Single-cell Data
An analogy can be drawn between language and cell biology. Words form sentences, similar to genes building cells. Therefore, large language models can be adapted to single-cell RNA sequencing studies.

However, current machine learning-based methods in single-cell research are quite fragmented and have limited data sets. To overcome this limitation, the project aims to develop a single-cell-based LLM. The model is pre-trained on vast amounts of data, and then fine-tuned and tested according to different analysis tasks.

Drawing on existing foundational model scGPT, we pretrained a customized version of a Transformer-based language model on 300,000 human blood cells, minimizing loss between real and predicted gene expression levels. This versatile model hopes to improve the accuracy and efficiency of single-cell analysis, thereby accelerating discovery and innovation in biology.

## Data Sampling
Retrieve all human blood cells via the CELLxGENE Census API. Select human blood cells by complete samples at random, ensuring that their total does not surpass 300,000.
![CensusAPI](https://github.com/sun770311/scLLM-pretrain/blob/main/Census_API.png)

## Preprocessing
Part 1: Based on single-cell
1. Filter out genes whose total number in the entire matrix is too low (set threshold)
2. Filter out cells with too few genes in the entire matrix (set threshold)
3. Screen the top 1200 highly variable genes in each sample: A highly variable gene is one that shows significant variability in its expression levels across different samples, conditions, or individuals. Highly variable genes can signal key changes in gene expression that drive disease processes, response to therapy, or developmental stages.
![HVG](https://github.com/sun770311/scLLM-pretrain/blob/main/HVG.png)
4. Normalize the total number of genes in each cell in the entire matrix
5. Discrete binning of continuous expressions in the entire matrix
![Binning](https://github.com/sun770311/scLLM-pretrain/blob/main/Binning.png)

Part 2: Further adjustments for training
1. Split data into 97% training and 3% validation
2. Exclude zero gene expressions, resulting in cells with differing numbers of non-zero genes
3. Align the information of all cells, adding special values "cls" and "pad", so that the total length of cells is 1201 (1200 highly variable genes + "cls"). "cls": We can extract the embedding of the cell from this position, with a value of 0. "pad": used to fill the empty space of the sequence, the value is -2
4. Tokenization: Convert gene names to numeric indexes: [0, total number of genes in the dataset - 1], special values: <cls> = total number of genes, <pad> = total number of genes + 1
5. Arbitrarily shield the gene expression level of each cell. The percentage of masking is selected from [0.25, 0.5, 0.75]. The masking position does not include cls and pad.
![Preprocess2](https://github.com/sun770311/scLLM-pretrain/blob/main/Preprocess2.png)

## Model Initialization
Traditional TransformerModel: Input embedding + location information as model input; part of the input to the decoder comes from the encoder

Modified TransformerModel for project goals: The embeddings of the gene name and gene expression level are used as model input, without adding position information. The traditional Decoder is removed because the main purpose of the project is not to generate but to understand single-cell data, while the Encoder is retained. The output of the Encoder is directly connected to fully connected neural networks, and different target tasks will activate different neural networks.

Neural Network Architecture: 
1. Number of layers = 12
2. Attention heads = 8
3. Embedding size = 512
4. Fully-connected layer size = 512
![TransformerModel](https://github.com/sun770311/scLLM-pretrain/blob/main/TransformerModel.png)

## Training
A total of 6 training cycles. One cycle contains ~9000 batches, and one batch contains 32 cells.

1. At the beginning of each cycle, the gene expression level of each cell is arbitrarily masked
2. Fill in the blanks: predict the expression level of blocked genes
3. Calculate the loss:
a. Mean squared error (MSE: Mean squared error) between the true value and the predicted value of the masking position
b. The mean square error between the true value and the predicted value based on cell embedding ("cls")
4. Update model parameters based on loss
5. Calculate the total loss of all batches in a cycle
![Training](https://github.com/sun770311/scLLM-pretrain/blob/main/Training.png)

## Saving Best Model
Save the best performing model as "best_model.pt", which saves all model parameters.

## References
The project draws on existing single-cell LLM scGPT's framework. Details can be found in the scGPT paper and codebase.

https://github.com/bowang-lab/scGPT/tree/main

https://www.nature.com/articles/s41592-024-02201-0

Link to the CELLxGENE Census API: https://chanzuckerberg.github.io/cellxgene-census/

## People
Hannah Sun

Mentor: Dr. Guan Tao Zheng

MobiDrop (Zhejiang) Co., Ltd.



