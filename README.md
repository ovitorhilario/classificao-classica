# Classificação de Imagens com GLCM + KNN

Este projeto implementa um sistema completo de **classificação de imagens** utilizando **extração de características baseada em textura** (GLCM - Gray Level Co-occurrence Matrix) seguida de **aprendizado de máquina supervisionado**. 

O sistema foi desenvolvido para operar no **Google Colab** com uma interface interativa que permite testar diferentes combinações de datasets e classificadores. Embora o projeto suporte múltiplos classificadores (Random Forest, SVM, MLP e KNN), os resultados apresentados neste README foram obtidos utilizando exclusivamente **GLCM + KNN**.

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Método GLCM](#método-glcm)
- [Tecnologias](#tecnologias)
- [Datasets](#datasets)
- [Classificadores](#classificadores)
- [Como Usar](#como-usar)
- [Resultados](#resultados)

---

## Sobre o Projeto

Este projeto foi desenvolvido para demonstrar a aplicação de técnicas clássicas de **Visão Computacional** e **Aprendizado de Máquina** na classificação de imagens. O pipeline completo inclui:

1. **Pré-processamento** das imagens
2. **Extração de características** usando GLCM
3. **Redução dimensional** com PCA
4. **Treinamento** de classificadores
5. **Avaliação** com métricas detalhadas

---

## Método GLCM

### O que é GLCM?

A **Matriz de Co-ocorrência de Níveis de Cinza** (GLCM) é uma técnica clássica que analisa a **textura** das imagens medindo como pares de pixels com determinadas intensidades ocorrem em relações espaciais específicas.

### Como Funciona?

O GLCM analisa a textura através de um processo em três etapas:

1. **Construção da Matriz**: Calcula a frequência com que pares de pixels com determinadas intensidades aparecem em uma direção específica
2. **Extração de Propriedades**: Computa medidas estatísticas que descrevem padrões de textura
3. **Geração do Vetor**: Cria um vetor de características numéricas que representa a textura da imagem

<div align="center">

| **Processo de Análise Espacial** | **Matriz de Co-ocorrência Resultante** |
| :---: | :---: |
| ![GLCM Process](https://github.com/user-attachments/assets/63a300ac-245d-4b69-93cd-03198c2d0574) | ![GLCM Matrix](https://github.com/user-attachments/assets/dd99db8a-6c94-4de6-a36b-618c5ef24a0f) |
| *Fonte: [GLCMTextures](https://github.com/ailich/GLCMTextures)* | *Fonte: [Scikit-image](https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html)* |

</div>



### Propriedades Extraídas

O GLCM calcula seis propriedades fundamentais que caracterizam diferentes aspectos da textura:

| Propriedade | O que Mede | Interpretação |
|-------------|------------|---------------|
| **Contrast** | Variação local de intensidade | Alto = muitas diferenças entre regiões claras e escuras |
| **Dissimilarity** | Diferença entre pares de pixels | Similar ao contraste, mas com peso linear |
| **Homogeneity** | Uniformidade da textura | Alto = textura suave e homogênea |
| **Energy** | Uniformidade da distribuição | Alto = poucos pares dominantes (textura ordenada) |
| **Correlation** | Dependência linear | Mede quão correlacionados estão os pixels vizinhos |
| **ASM** | Angular Second Moment | Raiz quadrada da energia, mede ordem |

### Parâmetros de Configuração

A implementação utiliza os seguintes parâmetros para otimizar desempenho e qualidade:

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Distâncias** | 1 pixel | Analisa vizinhança imediata |
| **Ângulos** | 0°, 45°, 90°, 135° | Captura textura em todas as direções |
| **Níveis de Cinza** | 32 | Quantização de 256 → 32 para eficiência computacional |
| **Redução PCA** | 50 componentes | Mantém informação relevante reduzindo dimensionalidade |
| **Tamanho de Imagem** | 256×256 | Padronização para processamento uniforme |

---

## Tecnologias

O projeto utiliza as seguintes bibliotecas e ferramentas:

| Tecnologia | Versão | Finalidade |
|------------|--------|------------|
| **Python** | 3.x | Linguagem base |
| **OpenCV** | Latest | Processamento e manipulação de imagens |
| **Scikit-image** | Latest | Implementação do algoritmo GLCM |
| **Scikit-learn** | Latest | Classificadores ML e métricas de avaliação |
| **NumPy** | Latest | Operações matriciais e vetoriais |
| **Matplotlib** | Latest | Visualização de resultados |
| **IPyWidgets** | Latest | Interface interativa no Colab |

---

## Datasets

O projeto foi testado com três datasets públicos do Kaggle, cada um representando um domínio diferente de classificação:

### 1. COVID-19 X-Ray Dataset
- **Descrição**: Imagens de raio-X de pulmões
- **Classes**: COVID (infectado) vs Normal (saudável)
- **Aplicação**: Diagnóstico médico assistido
- **Fonte**: [Kaggle - COVID-19 X-Rays](https://www.kaggle.com/datasets/tarandeep97/covid19-normal-posteroanteriorpa-xrays)

### 2. Fracture Detection Dataset
- **Descrição**: Raio-X de ossos
- **Classes**: Fractured (fraturado) vs Not Fractured (normal)
- **Aplicação**: Detecção automática de fraturas
- **Fonte**: [Kaggle - Fracture Detection](https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images/data)

### 3. OCR Digits Dataset
- **Descrição**: Imagens de dígitos manuscritos
- **Classes**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Aplicação**: Reconhecimento óptico de caracteres
- **Fonte**: [Kaggle - Standard OCR](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset/data)

---

## Classificadores

O sistema oferece quatro algoritmos de aprendizado de máquina, permitindo comparações de desempenho:

| Classificador | Tipo | Características | Status nos Resultados |
|---------------|------|-----------------|----------------------|
| **KNN** | Instance-based | Classificação por proximidade | ✅ **Utilizado** |
| **Random Forest** | Ensemble | Múltiplas árvores de decisão | ⚪ Disponível |
| **SVM** | Kernel-based | Hiperplano de separação (kernel linear) | ⚪ Disponível |
| **MLP** | Neural Network | Rede neural multicamadas | ⚪ Disponível |

> **Nota**: Este README apresenta resultados obtidos exclusivamente com **GLCM + KNN**. Os demais classificadores estão implementados e podem ser testados através da interface.

---

## Como Usar

### Pré-requisitos
- Conta Google (para acessar Google Drive e Colab)
- Datasets baixados do Kaggle

### Passo a Passo

#### 1. Preparação do Ambiente
```bash
# Clone o repositório
git clone https://github.com/ovitorhilario/classificao-classica.git
```

#### 2. Configuração no Google Drive
1. Faça upload da pasta do projeto para seu Google Drive
2. Organize os datasets na estrutura esperada:
```
classificao-classica/
├── datasets/
│   ├── covid19/
│   ├── fracture/
│   └── ocr/
├── modulos/
└── janela_principal.ipynb
```

#### 3. Execução no Google Colab
1. Abra o arquivo `janela_principal.ipynb` no Google Colab
2. **Configure os caminhos** no início do notebook:
```python
# Ajuste estas variáveis para apontar para sua pasta no Drive
caminho_modulos = '/content/drive/MyDrive/Colab Notebooks/classificao-classica/modulos'
caminho_base = '/content/drive/MyDrive/Colab Notebooks/classificao-classica'
```

#### 4. Workflow de Classificação

Execute as etapas na interface interativa:

| Aba | Ação | Descrição |
|-----|------|-----------|
| **DATASET** | Selecionar dataset | Escolha entre COVID-19, Fracture ou OCR |
| **EXT. CARACTERÍSTICAS** | Extrair GLCM | Processa imagens e salva características |
| **TREINAMENTO** | Treinar modelo | Selecione KNN e clique em "Treinar" |
| **CLASSIFICAÇÃO** | Avaliar modelo | Selecione KNN e clique em "Classificar" |

#### 5. Visualização dos Resultados
Os resultados são exibidos automaticamente e incluem:
- Matriz de confusão
- Relatório de classificação (Precision, Recall, F1-Score)
- Métricas consolidadas

---

## Resultados

Todos os resultados apresentados foram obtidos usando a combinação **GLCM + KNN** para extração de características e classificação.

### Métricas de Avaliação

O sistema gera automaticamente as seguintes métricas:

| Métrica | Descrição |
|---------|-----------|
| **Accuracy** | Percentual total de predições corretas |
| **Precision** | Proporção de predições positivas que estão corretas |
| **Recall** | Proporção de casos positivos que foram identificados |
| **F1-Score** | Média harmônica entre precision e recall |
| **Confusion Matrix** | Visualização detalhada de acertos e erros por classe |

---

### Dataset: COVID-19 (GLCM + KNN)

**Desempenho**: Sistema demonstra capacidade de distinguir entre casos de COVID-19 e exames normais através de análise de textura em imagens de raio-X pulmonar.

<div align="center">

| **Relatório de Classificação** | **Matriz de Confusão** |
| :---: | :---: |
| ![COVID Classification Report](https://github.com/user-attachments/assets/597959f4-6e5b-45f0-9600-160794eb9e89) | ![COVID Confusion Matrix](https://github.com/user-attachments/assets/5e4a83c3-c1bb-452a-b963-1148b334a46f) |

</div>

---

### Dataset: Fracture (GLCM + KNN)

**Desempenho**: O classificador identifica padrões de textura que diferenciam ossos fraturados de ossos saudáveis em imagens radiográficas.

<div align="center">

| **Relatório de Classificação** | **Matriz de Confusão** |
| :---: | :---: |
| ![Fracture Classification Report](https://github.com/user-attachments/assets/8df0227e-57a9-4c4c-99de-8ebc02fbe92a) | ![Fracture Confusion Matrix](https://github.com/user-attachments/assets/a9cf6d2e-fed2-462b-8a20-fd225d129733) |

</div>

---

### Dataset: OCR (GLCM + KNN)

**Desempenho**: Reconhecimento de dígitos manuscritos (0-9) baseado em características de textura, demonstrando aplicabilidade em reconhecimento óptico de caracteres.

<div align="center">

| **Relatório de Classificação** | **Matriz de Confusão** |
| :---: | :---: |
| ![OCR Classification Report](https://github.com/user-attachments/assets/9d90e74b-39bb-480d-82ec-d63a52ef9b81) | ![OCR Confusion Matrix](https://github.com/user-attachments/assets/45f40594-1def-4db3-b7dc-ebbaef5a85dd) |

</div>