# Classificação de Imagens com GLCM (Gray Level Co-occurrence Matrix) +  KNN (Nearest Neighbors)
Este projeto implementa classificação de imagens utilizando **extração de características** seguida de **aprendizado de máquina**. O foco principal é o método **GLCM (Gray Level Co-occurrence Matrix)** para análise de textura. Os resultados de treinamento e classificação foram gerados em conjunto com o classificador **KNN (K-Nearest Neighbors)**. 

**Ambiente**: Google Colab com interface interativa

## Método de Extração: GLCM

### O que é GLCM?

A **Matriz de Co-ocorrência de Níveis de Cinza** (GLCM) é uma técnica clássica que analisa a **textura** das imagens medindo como pares de pixels com determinadas intensidades ocorrem em relações espaciais específicas.

### Como Funciona?

1. **Análise Espacial**: Examina a relação entre pixels vizinhos em diferentes direções
2. **Extração de Propriedades**: Calcula medidas estatísticas da textura
3. **Vetor de Características**: Gera um vetor numérico representando a textura da imagem

| Fonte: [Figura de github.com/ailich/GLCMTextures](https://github.com/ailich/GLCMTextures) | Fonte: [Figure de scikit-image.org](https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html) |
| ----- | ----- |
| ![Imagem 1](https://github.com/user-attachments/assets/63a300ac-245d-4b69-93cd-03198c2d0574) | ![Imagem 2](https://github.com/user-attachments/assets/dd99db8a-6c94-4de6-a36b-618c5ef24a0f)



### Propriedades Extraídas

| Propriedade | Descrição |
|-------------|-----------|
| **Contrast** | Mede a variação local de intensidade (diferença entre regiões claras e escuras) |
| **Dissimilarity** | Similar ao contraste, mas com peso linear na diferença |
| **Homogeneity** | Mede a uniformidade da textura (valores altos = textura suave) |
| **Energy** | Mede a uniformidade da distribuição de pares de pixels |
| **Correlation** | Mede a dependência linear entre níveis de cinza |
| **ASM** | Angular Second Moment - raiz quadrada da energia |

### Parâmetros de Implementação

- **Distâncias**: 1 pixel
- **Ângulos**: 0°, 45°, 90°, 135° (quatro direções)
- **Níveis de Cinza**: 32 (quantização de 256 para melhor desempenho)
- **Redução Dimensional**: PCA com 50 componentes principais

---

## Tecnologias

- **Python 3.x**
- **OpenCV**: Processamento de imagens
- **Scikit-image**: Implementação do GLCM
- **Scikit-learn**: Classificadores e métricas
- **NumPy**: Operações numéricas
---

## Datasets

- **COVID-19**: Raio-X de pulmões (COVID vs Normal)
[https://www.kaggle.com/datasets/tarandeep97/covid19-normal-posteroanteriorpa-xrays]
- **Fracture**: Detecção de fraturas ósseas
[https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images/data]
- **OCR**: Reconhecimento de dígitos (0-9)
[https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset/data]

---

## Classificadores Disponíveis

| Classificador | Descrição |
|---------------|-----------|
| **KNN** | **K-Nearest Neighbors (Usado na classificação dos datasets)** |
| Random Forest | Ensemble de árvores de decisão |
| SVM | Support Vector Machine com kernel linear |
| MLP | Multi-Layer Perceptron (rede neural) |

---

## Como Usar

1. Baixe este repositório ou faça o clone com: `git clone https://github.com/ovitorhilario/classificao-classica.git`
2. Envie o projeto para uma pasta dentro do seu Google Drive
3. No Google Colab, conecte sua conta Google Drive.
4. Abra o arquivo `janela_principal.ipynb` diretamente pelo Drive usando o Google Colab.
5. No notebook, ajuste os caminhos para apontarem corretamente para a pasta onde você colocou o projeto no seu Drive.
```py
# Altere as variáveis caminho_modulos e caminho_base para o caminho correto
/content/drive/MyDrive/Colab Notebooks/classificao-classica/modulos
```
6. Execute a célula principal para montar o Drive
7. **Aba DATASET**: Selecione o dataset
8. **Aba EXT. CARACTERÍSTICAS**: Marque "GLCM" e clique em "Extrair"
9. **Aba TREINAMENTO**: Selecione classificador e "KNN", clique em "Treinar"
10. **Aba CLASSIFICAÇÃO**: Selecione classificador e "KNN", clique em "Classificar"

---

## Métricas de Avaliação

Para cada experimento, o sistema gera:

- **Matriz de Confusão**: Visualização de acertos e erros
- **Precision**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos identificados
- **F1-Score**: Média harmônica entre precision e recall
- **Acurácia**: Percentual de predições corretas

## Resultados Obtidos

### Dataset: COVID-19 (GLCM + KNN)
| Relatório de Classificação | Matriz de Confusão |
| ------- | ------- |
| ![graph_1_0](https://github.com/user-attachments/assets/597959f4-6e5b-45f0-9600-160794eb9e89) | ![graph_2_0](https://github.com/user-attachments/assets/5e4a83c3-c1bb-452a-b963-1148b334a46f) |

---

### Dataset: Fracture (GLCM + KNN)
| Relatório de Classificação | Matriz de Confusão |
| ------- | ------- |
| ![graph_1_1](https://github.com/user-attachments/assets/8df0227e-57a9-4c4c-99de-8ebc02fbe92a) | ![graph_2_1](https://github.com/user-attachments/assets/a9cf6d2e-fed2-462b-8a20-fd225d129733) |

---

### Dataset: OCR (GLCM + KNN)
| Relatório de Classificação | Matriz de Confusão |
| ------- | ------- |
| ![graph_1_2](https://github.com/user-attachments/assets/9d90e74b-39bb-480d-82ec-d63a52ef9b81) | ![graph_2_2](https://github.com/user-attachments/assets/45f40594-1def-4db3-b7dc-ebbaef5a85dd) |

---
