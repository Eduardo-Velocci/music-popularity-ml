# music-popularity-ml
Classificação e agrupamento de músicas baseado em atributos acústicos — Random Forest, XGBoost e K-Means

> Projeto Final — Disciplina: Técnicas Avançadas de Machine Learning  
> Aluno: Eduardo de Almeida Velocci

---

## 📌 Sobre o Projeto

Este projeto aplica técnicas de Machine Learning para analisar músicas com base em seus atributos acústicos, buscando responder três questões centrais:

1. Quais atributos acústicos são mais relevantes para a popularidade?
2. É possível prever se uma música será popular com base nesses atributos?
3. Existem grupos de músicas com características similares?

---

## 📊 Dataset

- **3.370 músicas** com 22 variáveis
- Atributos acústicos: `energy`, `danceability`, `loudness`, `acousticness`, `instrumentalness`, `speechiness`, `valence`, `tempo`, `liveness`, `key`, `mode`, `time_signature`, `duration_ms`
- Variável alvo criada: `popular` (1 se `track_popularity > 70`, caso contrário 0)
- Distribuição das classes: ~75% não popular / ~25% popular (desbalanceado)

---

## 🔬 Metodologia

### 1. Análise Exploratória (EDA)
- Estatísticas descritivas e análise de valores ausentes
- Boxplots para detecção de outliers
- Padronização via `StandardScaler` (z-score)

### 2. Classificação Binária de Popularidade
- Divisão treino/teste: 80/20 com estratificação
- Modelos treinados:
  - **Random Forest** — 1000 árvores, profundidade máxima 10
  - **XGBoost** — 1000 estimadores, learning rate 0.01
  - Versões com **GridSearchCV** para otimização de hiperparâmetros
  - Versões **balanceadas** com `class_weight` para lidar com desbalanceamento
- Métricas: Acurácia, F1-Score e Gini (2×AUC − 1)

### 3. Clusterização com K-Means
- Pré-processamento: padronização + PCA com 6 componentes (~85-90% da variância)
- Escolha do número de clusters: Método do Cotovelo + Silhouette Score
- Resultado: **k = 3 clusters** com perfis acústicos bem definidos
- Visualização via projeção nos 2 primeiros componentes do PCA

---

## 📈 Resultados

### Classificação

| Modelo | Acurácia | F1-Score | Gini |
|---|---|---|---|
| Random Forest (Full) | 0.7522 | 0.0234 | 0.5141 |
| XGBoost (Full) | 0.7448 | 0.1810 | 0.4971 |
| Random Forest (Balanceado) | 0.7018 | 0.4401 | 0.4481 |
| **XGBoost (Balanceado)** | **0.7187** | **0.4444** | **0.4836** |

> Os modelos balanceados apresentaram melhora significativa no recall da classe "Popular" (47%), demonstrando a importância do tratamento do desbalanceamento.

### Features mais importantes (XGBoost Balanceado)
1. `instrumentalness`
2. `loudness`
3. `acousticness`
4. `speechiness`
5. `energy`

### Clusterização

| Cluster | Perfil |
|---|---|
| 0 | Músicas acústicas e instrumentais — baixa energia e loudness |
| 1 | Músicas energéticas e dançantes — alta danceability e loudness |
| 2 | Perfil intermediário — combina características dos dois extremos |

---

## 🛠️ Tecnologias Utilizadas

- Python 3
- pandas, numpy
- scikit-learn (RandomForest, KMeans, PCA, StandardScaler, GridSearchCV)
- XGBoost
- matplotlib, seaborn

---

## ▶️ Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/music-popularity-ml.git
cd music-popularity-ml
```

2. Instale as dependências:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Abra o notebook:
```bash
jupyter notebook notebooks/Avaliação_Final_EduardoVelocci.ipynb
```

---

## ⚠️ Limitações

- A definição binária de popularidade (> 70) é uma simplificação do fenômeno
- Fatores externos como marketing, reconhecimento do artista e contexto cultural não foram considerados
- Dataset limitado a 3.370 músicas

## 🔮 Melhorias Futuras

- Incorporação de dados textuais (letras das músicas)
- Uso de modelos de deep learning
- Análise temporal de tendências musicais
- Expansão do dataset com dados da API do Spotify
