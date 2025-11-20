# Classificação de Imagens com GLCM
Este projeto implementa classificação de imagens utilizando **extração de características** seguida de **aprendizado de máquina**. O foco principal é o método **GLCM (Gray Level Co-occurrence Matrix)** para análise de textura.

**Ambiente**: Google Colab com interface interativa

## Método de Extração: GLCM

### O que é GLCM?

A **Matriz de Co-ocorrência de Níveis de Cinza** (GLCM) é uma técnica clássica que analisa a **textura** das imagens medindo como pares de pixels com determinadas intensidades ocorrem em relações espaciais específicas.

### Como Funciona?

1. **Análise Espacial**: Examina a relação entre pixels vizinhos em diferentes direções
2. **Extração de Propriedades**: Calcula medidas estatísticas da textura
3. **Vetor de Características**: Gera um vetor numérico representando a textura da imagem

| [Figura de github.com/ailich/GLCMTextures](https://github.com/ailich/GLCMTextures) |
| ----- |
| ![Descrição da imagem](https://github.com/user-attachments/assets/63a300ac-245d-4b69-93cd-03198c2d0574) |


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

## Classificadores

| Classificador | Descrição |
|---------------|-----------|
| **Random Forest** | Ensemble de árvores de decisão |
| **SVM** | Support Vector Machine com kernel linear |
| **KNN** | K-Nearest Neighbors |
| **MLP** | Multi-Layer Perceptron (rede neural) |

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
9. **Aba TREINAMENTO**: Selecione classificador e "GLCM", clique em "Treinar"
10. **Aba CLASSIFICAÇÃO**: Selecione classificador e "GLCM", clique em "Classificar"

---

## Métricas de Avaliação

Para cada experimento, o sistema gera:

- **Matriz de Confusão**: Visualização de acertos e erros
- **Precision**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos identificados
- **F1-Score**: Média harmônica entre precision e recall
- **Acurácia**: Percentual de predições corretas

## Resultados Obtidos

### Dataset: COVID-19 (GLCM)

---

### Dataset: Fracture (GLCM)

---

### Dataset: OCR (GLCM)

---
