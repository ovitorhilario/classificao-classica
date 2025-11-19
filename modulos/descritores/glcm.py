# Este módulo permite a extração de características do tipo GLCM
# GLCM (Gray Level Co-occurrence Matrix) - Matriz de Co-ocorrência de Níveis de Cinza
# Esta técnica analisa a relação espacial entre pixels, calculando propriedades de textura

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm

def extrair_glcm(imagens, distancias=[1], angulos=[0, np.pi/4, np.pi/2, 3*np.pi/4], niveis=256, propriedades=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']):
    """
    Extrai características GLCM (Gray Level Co-occurrence Matrix) de uma lista de imagens.
    
    O GLCM analisa a textura da imagem medindo como pares de pixels com determinadas 
    intensidades ocorrem em uma relação espacial específica.
    
    Parâmetros:
    - imagens: Lista de imagens em formato de array numpy.
    - distancias: Lista de distâncias entre pixels para calcular a co-ocorrência. Default é [1].
    - angulos: Lista de ângulos (em radianos) para calcular a co-ocorrência. 
               Default é [0, π/4, π/2, 3π/4] (0°, 45°, 90°, 135°).
    - niveis: Número de níveis de cinza. Default é 256.
    - propriedades: Lista de propriedades GLCM a extrair. 
                    Opções: 'contrast', 'dissimilarity', 'homogeneity', 'energy', 
                            'correlation', 'ASM' (Angular Second Moment).
    
    Retorno:
    - Array numpy contendo as características GLCM de cada imagem.
    
    Propriedades GLCM:
    - Contrast: Mede a variação local de intensidade (diferença entre valores altos e baixos)
    - Dissimilarity: Similar ao contraste, mas com peso linear
    - Homogeneity: Mede a uniformidade da textura (valores altos = textura uniforme)
    - Energy: Mede a uniformidade da distribuição de pares de pixels
    - Correlation: Mede a dependência linear entre níveis de cinza
    - ASM: Angular Second Moment - raiz quadrada da energia
    """
    
    # Redimensionamento das imagens para 256x256 para padronização e eficiência
    imagens_redimensionadas = [cv2.resize(imagem, (256, 256)) for imagem in imagens]
    
    # Calcula o total de imagens do conjunto
    total_imagens = len(imagens_redimensionadas)
    
    # Inicializa uma lista para armazenar as características GLCM de cada imagem
    lista_caracteristicas_glcm = []
    
    with tqdm(total=total_imagens, desc="Extraindo características GLCM") as pbar:
        # Itera sobre cada imagem na lista de imagens
        for imagem in imagens_redimensionadas:
            # Verifica se a imagem é colorida (tem mais de 2 dimensões)
            if len(imagem.shape) > 2:
                # Converte a imagem de RGB para escala de cinza
                imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
            
            # Reduz o número de níveis de cinza para 32 para melhor desempenho
            # Isso quantiza a imagem de 256 níveis para 32 níveis
            niveis_reduzidos = 32
            imagem_quantizada = (imagem / (256 / niveis_reduzidos)).astype(np.uint8)
            
            # Calcula a matriz GLCM
            # graycomatrix calcula como os pares de pixels ocorrem em diferentes direções
            glcm = graycomatrix(
                imagem_quantizada,
                distances=distancias,
                angles=angulos,
                levels=niveis_reduzidos,
                symmetric=True,
                normed=True
            )
            
            # Extrai as propriedades GLCM
            caracteristicas = []
            for propriedade in propriedades:
                # graycoprops calcula propriedades específicas da matriz GLCM
                propriedade_valores = graycoprops(glcm, propriedade)
                # Achata os valores e adiciona à lista de características
                caracteristicas.extend(propriedade_valores.flatten())
            
            # Adiciona as características extraídas à lista
            lista_caracteristicas_glcm.append(np.array(caracteristicas))
            pbar.update(1)  # Atualiza a barra de progresso
    
    # Converte a lista de características em um array numpy
    caracteristicas_array = np.array(lista_caracteristicas_glcm)
    
    # REDUÇÃO DA DIMENSIONALIDADE COM PCA
    # n_components_pca: Número de componentes principais que serão mantidos
    n_samples, n_features = caracteristicas_array.shape
    n_components_pca = min(50, n_samples, n_features)  # Limita a 50 ou ao número de amostras/features
    
    if n_components_pca > 1:
        print(f'Aplicando PCA para reduzir de {n_features} para {n_components_pca} dimensões...')
        # Cria o objeto PCA com o número especificado de componentes
        pca = PCA(n_components=n_components_pca)
        # Aplica a transformação PCA nas características GLCM
        caracteristicas_reduzidas = pca.fit_transform(caracteristicas_array)
        print(f'Variância explicada pelo PCA: {sum(pca.explained_variance_ratio_)*100:.2f}%')
        return caracteristicas_reduzidas
    else:
        print('PCA não aplicado devido ao número insuficiente de componentes.')
        return caracteristicas_array
