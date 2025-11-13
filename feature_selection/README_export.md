# Documentação de Exportação de Dados para Dashboard (Streamlit)

Este documento descreve os 3 ficheiros JSON gerados pelo script `bnb_feature_selection.py`. O objetivo destes ficheiros é fornecer todos os dados necessários para o dashboard Streamlit visualizar o processo e os resultados do algoritmo Branch and Bound (B\&B).

## Ficheiros Gerados

Ao executar `python bnb_feature_selection.py`, os seguintes 3 ficheiros serão criados (ou sobrescritos) no mesmo diretório:

1.  `export_bnb_tree.json`
2.  `export_bnb_summary.json`
3.  `export_heuristic_comparison.json`

-----

### 1\. `export_bnb_tree.json`

Este é o ficheiro mais detalhado. Contém um log de **cada nó** que o algoritmo B\&B visitou, permitindo a reconstrução visual da árvore de busca.

  * **Conteúdo:** Uma lista (array) de objetos JSON, onde cada objeto é um nó.
  * **Como usar no Streamlit:** Ideal para usar com `graphviz` ou `plotly.graph_objects` (ex: Sankey ou Sunburst) para desenhar a árvore. A relação `id` -\> `parent_id` define a estrutura da árvore.

**Estrutura de cada Objeto de Nó:**

```json
{
  "id": 1,
  "parent_id": 0,
  "decision": "NÃO fixed acidity",
  "features": [],
  "feature_count": 0,
  "score": null,
  "status": "PODADO_BOUND"
}
```

**Notas sobre a Estrutura:**

  * `score: null`: O `null` (convertido do `None` do Python) aparece em vez de `-Infinity`. Representa um nó que não foi avaliado (ex: podado por bound) ou que teve um score inválido (ex: 0 features).
  * `parent_id: -1`: É usado para o nó "RAIZ" (o início de tudo).

**Valores Possíveis para `status`:**

  * `"EXPLORADO"`: Nó expandido. Teve filhos (ramos "INCLUIR" e "NÃO INCLUIR").
  * `"PODADO_BOUND"`: Nó podado porque `feature_count` era $\ge$ à melhor solução já encontrada.
  * `"SOLUCAO_OTIMA_ATUAL"`: Nó que representa uma solução válida e que é a **melhor** encontrada até agora. (É uma folha).
  * `"PODADO_SOLUCAO_PIOR"`: Nó que representa uma solução válida, mas que **não é melhor** que a atual (tem $\ge$ features). (É uma folha).
  * `"FOLHA_INVALIDA"`: Nó que chegou ao fim da árvore (testou todas as features) e **não atingiu** a meta de R2. (É uma folha).

-----

### 2\. `export_bnb_summary.json`

Este ficheiro contém o resultado final e as métricas de execução de alto nível do B\&B.

  * **Conteúdo:** Um único objeto JSON com os resultados sumariados.
  * **Como usar no Streamlit:** Perfeito para os cartões de KPI (Key Performance Indicators) e para mostrar a solução final.

**Estrutura do Objeto:**

```json
{
  "final_solution": {
    "features": ["volatile acidity", "alcohol"],
    "feature_count": 2,
    "r2_score": 0.3345
  },
  "execution_metrics": {
    "total_time_seconds": 5.21,
    "nodes_visited": 79,
    "solutions_found_count": 2,
    "r2_goal": 0.30
  },
  "solutions_timeline": [
    {
      "features": ["total sulfur dioxide", "pH", "sulphates", "alcohol"],
      "feature_count": 4,
      "score": 0.3066
    },
    {
      "features": ["volatile acidity", "alcohol"],
      "feature_count": 2,
      "score": 0.3345
    }
  ]
}
```

-----

### 3\. `export_heuristic_comparison.json`

Este ficheiro contém os dados necessários para o gráfico de validação, comparando o resultado ótimo do B\&B com uma heurística gulosa (Greedy Forward Selection).

  * **Conteúdo:** Um objeto JSON com os dados do B\&B e os passos da heurística.
  * **Como usar no Streamlit:** Para criar um gráfico de linhas (`plotly` ou `altair`) que mostre "Score R2 vs. N.º de Features" para as duas abordagens.

**Estrutura do Objeto:**

```json
{
  "bnb_optimal": {
    "features": ["volatile acidity", "alcohol"],
    "feature_count": 2,
    "r2_score": 0.3345
  },
  "greedy_heuristic_steps": [
    {
      "feature_count": 1,
      "r2_score": 0.2850,
      "features": ["alcohol"]
    },
    {
      "feature_count": 2,
      "r2_score": 0.3345,
      "features": ["alcohol", "volatile acidity"]
    },
    {
      "feature_count": 3,
      "r2_score": 0.3421,
      "features": ["alcohol", "volatile acidity", "sulphates"]
    }
  ]
}
```