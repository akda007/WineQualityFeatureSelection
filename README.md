# Otimização da Qualidade do Vinho com Branch and Bound

Este projeto aplica o algoritmo de otimização combinatória **Branch and Bound (B\&B)** para resolver um problema de **Seleção de Atributos (Feature Selection)**.

O objetivo é encontrar o **número mínimo** de atributos químicos do dataset "Wine Quality" que são necessários para prever a qualidade de um vinho com um nível de precisão (R² score) pré-definido.

## 1\. O Problema

Dado um conjunto de 11 atributos (features) físico-químicos de um vinho (ex: `alcohol`, `pH`, `sulphates`), qual é o menor conjunto de atributos que precisamos para treinar um modelo de Regressão Linear que atinja uma meta de performance?

  * **Objetivo:** Minimizar o número de features (ex: `['alcohol', 'sulphates']`).
  * **Restrição (Meta):** O R² Score do modelo deve ser $\ge$ um valor X (ex: `0.30`).

## 2\. A Abordagem: Branch and Bound

Este problema é NP-difícil (complexo), pois testar todas as $2^{11}$ combinações (2.048) é viável, mas testar 30 features ($2^{30} \approx 1$ bilião) seria impossível. Usamos o B\&B para encontrar a solução ótima sem ter que explorar todo o espaço de busca.

  * **Ramo (Branch):** O algoritmo explora uma árvore de decisão. Em cada nível, ele decide se deve "INCLUIR" ou "NÃO INCLUIR" um atributo específico.
  * **Limite e Poda (Bound & Prune):** Esta é a otimização principal. Um ramo da árvore é "podado" (ignorado) se:
    1.  **Poda por Limite (Bound):** O número de features no ramo atual já é maior ou igual à melhor solução que encontrámos até agora.
    2.  **Poda por Solução (Feasibility):** O ramo atinge a meta de R², mas não é melhor que a solução atual (ex: atinge a meta com 4 features quando já temos uma solução com 2).

### Validação

O resultado ótimo do B\&B é comparado com uma **Heurística Gulosa (Greedy Forward Selection)**. Isto serve para demonstrar o valor do B\&B, mostrando um caso onde a heurística (rápida mas "míope") pode falhar em encontrar a verdadeira solução ótima.

## 3\. Dataset

Usamos o dataset "Wine Quality" (proveniente do UCI, disponível no Kaggle), que foi pré-processado e limpo, resultando no ficheiro `dataset_cleaning/wine_clean.csv`.

## 4\. Requisitos

  * Python 3.x
  * pandas
  * scikit-learn
  * streamlit (para o dashboard de visualização)

Para instalar as dependências:

```bash
pip install -r requirements.txt
```

## 5\. Como Executar

O projeto é dividido em duas partes: o "motor" (solver B\&B) e o "painel" (dashboard).

### Parte 1: Executar o Solver B\&B

Primeiro, execute o script principal para resolver o problema de otimização. Isto irá gerar os ficheiros JSON necessários para o dashboard.

1.  (Opcional) Abra `bnb_feature_selection.py` e ajuste a constante `MINIMUM_R2_SCORE` para definir a sua meta.
2.  Execute o script no seu terminal:
    ```bash
    python bnb_feature_selection.py
    ```

### Parte 2: Visualizar o Dashboard

Assim que o solver terminar, execute a aplicação Streamlit para ver os resultados.

1.  Certifique-se que os ficheiros `export_*.json` estão no mesmo diretório.
2.  Execute o Streamlit (assumindo que o seu ficheiro se chama `dashboard.py`):
    ```bash
    streamlit run dashboard.py
    ```

## 6\. Ficheiros de Saída (Exportação)

Ao executar `bnb_feature_selection.py`, são gerados 3 ficheiros JSON para alimentar o dashboard:

  * `export_bnb_tree.json`: Um log detalhado de cada nó visitado, podado ou explorado. Usado para construir a visualização da Árvore de Busca.
  * `export_bnb_summary.json`: Métricas de alto nível: tempo total, nós visitados, a solução ótima final e um histórico de todas as soluções viáveis encontradas.
  * `export_heuristic_comparison.json`: Dados para o gráfico de validação, comparando o resultado (Score vs. N.º de Features) do B\&B contra a Heurística Gulosa.

(Para detalhes sobre a estrutura exata de cada JSON, consulte o ficheiro [`README_export.md`](./feature_selection/README_export.md)).
