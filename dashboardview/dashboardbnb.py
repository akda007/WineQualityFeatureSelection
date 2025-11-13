import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

FILE_PATH_WINE = "WineQT.csv"
FILE_PATH_SUMMARY = "../feature_selection/export_bnb_summary.json"
FILE_PATH_TREE = "../feature_selection/export_bnb_tree.json"
FILE_PATH_HEURISTIC = "../feature_selection/export_heuristic_comparison.json"

st.set_page_config(
    page_title="Projeto Branch and Bound - Wine Quality",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data 
def load_data(path):
    try:
        df = pd.read_csv(path)
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
        return df
    except FileNotFoundError:
        st.error(f"Erro: O arquivo CSV ('{path}') não foi encontrado. Verifique o caminho.")
        return pd.DataFrame()

@st.cache_data
def load_json_data(path):
    if not os.path.exists(path):
        st.warning(f"O arquivo JSON '{path}' não foi encontrado. Verifique o caminho e a existência do arquivo.")
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        st.warning(f"Erro ao decodificar JSON do arquivo '{path}'. Verifique o formato.")
        return None
    except Exception as e:
        st.warning(f"Ocorreu um erro ao carregar o arquivo '{path}': {e}")
        return None

df_wine = load_data(FILE_PATH_WINE)
bnb_summary = load_json_data(FILE_PATH_SUMMARY)
bnb_tree = load_json_data(FILE_PATH_TREE)
bnb_heuristic_comp = load_json_data(FILE_PATH_HEURISTIC)


if df_wine.empty:
    st.stop() 


def generate_plotly_tree_viz(tree_data, summary_data):
    if not tree_data or not summary_data:
        return go.Figure()

    # Cores de Status Padrão
    status_colors = {
        "EXPLORADO": '#E0F7FA', # Azul Claro
        "SOLUÇÃO": '#C8E6C9',   # Verde Claro (Cor padrão, mas será sobrescrito pela lógica abaixo)
        "PODADO_BOUND": '#FFAB91', # Laranja Claro
        "PODADO_VIABILIDADE": '#FFCDD2', # Vermelho Claro
        "RAIZ": '#BBDEFB', # Azul Pálido
        "lightgray": '#DDDDDD'
    }

    # Cores Específicas para Soluções Viáveis na Timeline
    COLOR_FINAL_SOLUTION = '#4CAF50' # Verde Brilhante
    COLOR_INTERMEDIATE_SOLUTION = '#4DD0E1' # Ciano/Aqua

    # --- Pre-processing para Soluções Viáveis ---
    viable_solutions = set()
    solutions_timeline = summary_data.get('solutions_timeline', [])
    for solution in solutions_timeline:
        features_set = frozenset(solution.get('features', []))
        viable_solutions.add(features_set)

    final_solution = summary_data.get('final_solution', {})
    final_features_set = frozenset(final_solution.get('features', []))
    final_score = final_solution.get('r2_score')
    final_feature_count = final_solution.get('feature_count')
    
    # Remove a solução final da lista de soluções viáveis intermediárias
    if final_features_set in viable_solutions:
        viable_solutions.remove(final_features_set)
    
    # --- Estrutura da Visualização ---
    node_coords = {}
    X_coords = []
    Y_coords = []
    node_info = []
    node_colors = []
    
    max_depth = max(node['feature_count'] for node in tree_data) if tree_data else 0
    
    for i, node in enumerate(tree_data):
        node_id = str(node['id'])
        status = node['status']
        
        is_final_solution_node = False
        is_intermediate_solution_node = False
        
        node_features_set = frozenset(node['features'])
        
        # 1. Checa Solução Ótima Final
        if (node['feature_count'] == final_feature_count and 
            abs((node['score'] or -2) - (final_score or -1)) < 1e-6 and # Verifica score com tolerância
            node_features_set == final_features_set):
            is_final_solution_node = True
            
        # 2. Checa Soluções Viáveis Intermediárias
        if not is_final_solution_node and node_features_set in viable_solutions:
            is_intermediate_solution_node = True
            
        
        score_text = f"R²: {node['score']:.4f}" if node.get('score') is not None and node['score'] > -1 else "R²: N/A"
        
        x = i * 5
        y = max_depth - node['feature_count'] 

        node_coords[node_id] = (x, y)
        X_coords.append(x)
        Y_coords.append(y)
        
        # Aplica cores
        if is_final_solution_node:
            node_colors.append(COLOR_FINAL_SOLUTION) 
        elif is_intermediate_solution_node:
            node_colors.append(COLOR_INTERMEDIATE_SOLUTION)
        else:
            node_colors.append(status_colors.get(status, 'lightgray'))

        hover_text = (
            f"<b>ID: {node_id}</b><br>"
            f"Decisão: {node['decision']}<br>"
            f"{score_text}<br>"
            f"Status: {status}<br>"
            f"Features: {', '.join(node['features'])}"
        )
        
        if is_final_solution_node:
            hover_text += "<br><b>[ÓTIMO GLOBAL]</b>"
        elif is_intermediate_solution_node:
            hover_text += "<br><b>[SOLUÇÃO VIÁVEL NA TIMELINE]</b>"
            
        node_info.append(hover_text)

    edge_x = []
    edge_y = []

    for node in tree_data:
        if node['parent_id'] != -1:
            parent_id = str(node['parent_id'])
            node_id = str(node['id'])
            
            if node_id in node_coords and parent_id in node_coords:
                x0, y0 = node_coords[parent_id]
                x1, y1 = node_coords[node_id]

                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#666'),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter(
        x=X_coords, y=Y_coords,
        mode='markers',
        hoverinfo='text',
        text=node_info,
        marker=dict(
            size=14,
            color=node_colors,
            line_width=2,
            line_color='#333',
        )
    ))

    fig.update_layout(
        title='Visualização Interativa da Árvore Branch and Bound',
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
        ),
        width=1800,
        height=800,
        margin=dict(t=50, b=50, l=50, r=50),
    )
    
    y_labels = {max_depth - i: f"Profundidade {i} (Nº Features={i})" for i in range(max_depth + 1)}
    fig.update_yaxes(tickvals=list(y_labels.keys()), ticktext=list(y_labels.values()), title="Profundidade (Nº de Features)")
    fig.update_xaxes(showticklabels=False, title="Dispersão Horizontal (Use o Scroll e Zoom!)")


    return fig


st.sidebar.title("Menu do Projeto")

pages = ["1. EDA e Base de Dados", "2. Execução do Branch and Bound", "3. Resultados e Validação"]
page = st.sidebar.radio("Seções", pages)


if page == "1. EDA e Base de Dados":
    st.header("1. Análise Exploratória de Dados (EDA) e Base")
    st.markdown("Visualização das características físico-químicas do **Wine Quality Dataset** e seus padrões iniciais, que fundamentam a modelagem.")
    st.write("---")

    st.subheader("1.1 Estrutura e Estatísticas Básicas")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total de Observações (Linhas)", df_wine.shape[0])
        st.metric("Total de Variáveis (Colunas)", df_wine.shape[1])
        st.caption("Visão inicial do Dataset (sem coluna 'Id'):")
        st.dataframe(df_wine.head(5))

    with col2:
        st.write("Estatísticas Descritivas (Média, Mediana, Desvio Padrão):")
        stats_df = df_wine.describe().T[['mean', '50%', 'std']].rename(columns={'50%': 'Mediana', 'mean': 'Média', 'std': 'Desvio Padrão'})
        st.dataframe(stats_df, use_container_width=True) 

    st.write("---")

    st.subheader("1.2 Distribuição das Variáveis e Assimetria")
    
    all_cols = df_wine.columns.tolist()
    
    var_selecionada = st.selectbox(
        "Selecione uma variável para ver sua Distribuição (Histograma):",
        all_cols,
        index=all_cols.index('quality') if 'quality' in all_cols else 0
    )

    fig_hist = px.histogram(
        df_wine, 
        x=var_selecionada, 
        title=f"Distribuição de **{var_selecionada}**",
        marginal="box", 
        color_discrete_sequence=['#9e0142'] 
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.info(
        "**Assimetria:** Observamos assimetria acentuada em variáveis como `chlorides`, `total sulfur dioxide` e `residual sugar`. Optou-se por não transformar os dados para facilitar a interpretação na otimização combinatória."
    )
    st.write("---")

    st.subheader("1.3 Padrões de Correlação com a Qualidade")
    st.markdown("A `quality` (nota 0 a 10) é a variável alvo (Target). Buscamos um subconjunto de _features_ que maximize o score de regressão (R²).")

    col3, col4 = st.columns(2)
    
    with col3:
        fig_scatter1 = px.scatter(
            df_wine,
            x="alcohol", 
            y="quality",
            trendline="ols", 
            title="Relação: Teor Alcoólico vs. Qualidade",
            color_discrete_sequence=['#5e4fa2']
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
        st.caption("Vinhos com maior teor alcoólico tendem a ter maior qualidade.")

    
    with col4:
        fig_scatter2 = px.scatter(
            df_wine,
            x="volatile acidity", 
            y="quality",
            trendline="ols",
            title="Relação: Acidez Volátil vs. Qualidade",
            color_discrete_sequence=['#d53e4f']
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
        st.caption("Relação negativa: O excesso de acidez volátil está associado a menor qualidade.")

    
    st.write("##### Correlação Forte entre Variáveis Preditivas:")
    fig_corr = px.scatter(
        df_wine,
        x="residual sugar",
        y="density",
        title="Correlação entre Residual Sugar e Density",
        trendline="ols",
        color_discrete_sequence=['#fee08b']
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Forte correlação observada entre `residual sugar` e `density`.")

elif page == "2. Execução do Branch and Bound":
    st.header("2. Execução do Branch and Bound (B&B)")
    st.markdown("O B&B foi executado para encontrar o subconjunto de *features* que maximiza o score $R^2$ em um modelo de Regressão Linear, dentro de uma restrição de *budget* (número máximo de features).")
    st.write("---")

    if bnb_summary and bnb_tree:
        st.subheader("2.1 Métricas do Algoritmo")
        metrics = bnb_summary.get('execution_metrics', {})
        final_solution = bnb_summary.get('final_solution', {})

        col_time, col_nodes, col_solutions, col_goal = st.columns(4)
        
        with col_time:
            st.metric("Tempo Total de Execução", f"{metrics.get('total_time_seconds', 0):.2f} s")
        with col_nodes:
            st.metric("Nós da Árvore Visitados", f"{metrics.get('nodes_visited', 0)}")
        with col_solutions:
            st.metric("Soluções Viáveis Encontradas", f"{metrics.get('solutions_found_count', 0)}")
        with col_goal:
            st.metric("Meta Mínima de R² (Goal)", f"{metrics.get('r2_goal', 0):.2f}")

        st.info(
            f"**Solução Ótima Final:** O algoritmo Branch and Bound encontrou o máximo R² de **{final_solution.get('r2_score', 0):.4f}** com **{final_solution.get('feature_count', 0)}** features. Este é o **ótimo global** do problema de otimização combinatória, pois maximiza o $R^2$ **respeitando a restrição de budget** (Nº de features) e atingindo a meta mínima de $R^2$ (0.30)."
        )
        st.write("---")


        st.subheader("2.2 Linha do Tempo das Melhores Soluções")
        st.markdown("Todas as soluções que superaram o `best_bound` (melhor cota) até o momento são registradas:")

        solutions = bnb_summary.get('solutions_timeline', [])
        
        solutions_df = pd.DataFrame(solutions)
        if not solutions_df.empty:
            solutions_df['Features'] = solutions_df['features'].apply(lambda x: ', '.join(x))
            solutions_df = solutions_df.rename(columns={
                'feature_count': 'Nº de Features',
                'score': 'Score R²'
            })
            st.dataframe(solutions_df[['Nº de Features', 'Score R²', 'Features']], hide_index=True, use_container_width=True)
        else:
            st.warning("Nenhuma solução viável foi registrada na linha do tempo.")
        
        st.write("---")

        st.subheader("2.3 Visualização Estrutural Interativa da Árvore de Busca")
        st.markdown(
            "Use o mouse para **dar zoom, arrastar (pan)** e inspecionar os detalhes de cada nó (status, R² e decisão) **passando o mouse sobre eles (hover)**."
        )
        
        col_legend1, col_legend2, col_legend3, col_legend4, col_legend5 = st.columns(5)
        col_legend1.markdown(f"Cor: <span style='background-color: #4CAF50; padding: 2px; border-radius: 3px;'>&nbsp;&nbsp;</span> **Ótimo Global**", unsafe_allow_html=True)
        col_legend2.markdown(f"Cor: <span style='background-color: #4DD0E1; padding: 2px; border-radius: 3px;'>&nbsp;&nbsp;</span> **Solução Viável**", unsafe_allow_html=True)
        col_legend3.markdown(f"Cor: <span style='background-color: #FFAB91; padding: 2px; border-radius: 3px;'>&nbsp;&nbsp;</span> **Podado por Cota**", unsafe_allow_html=True)
        col_legend4.markdown(f"Cor: <span style='background-color: #FFCDD2; padding: 2px; border-radius: 3px;'>&nbsp;&nbsp;</span> **Podado por Inviabilidade**", unsafe_allow_html=True)
        col_legend5.markdown(f"Cor: <span style='background-color: #E0F7FA; padding: 2px; border-radius: 3px;'>&nbsp;&nbsp;</span> **Explorado/Raiz**", unsafe_allow_html=True)
        st.write("\n")

        fig = generate_plotly_tree_viz(bnb_tree, bnb_summary)
        st.plotly_chart(fig, use_container_width=False)
        
    else:
        st.warning("Dados de resumo ou árvore do B&B não puderam ser carregados. Verifique se os caminhos dos arquivos JSON estão corretos (deve ser: '../feature_selection/nome_do_arquivo.json').")


elif page == "3. Resultados e Validação":
    st.header("3. Resultados e Validação do Modelo")
    st.markdown("Compara-se o ótimo global encontrado pelo Branch and Bound com a performance de uma heurística gulosa (Greedy Forward Selection), que é mais rápida, mas não garante a otimalidade global.")
    st.write("---")

    if bnb_summary and bnb_heuristic_comp:
        
        st.subheader("3.1 Solução Ótima Global (Branch and Bound)")
        final_solution = bnb_summary.get('final_solution', {})
        
        col_features, col_score = st.columns(2)

        with col_features:
            features = final_solution.get('features', [])
            st.metric("Nº de Features", final_solution.get('feature_count', 0))
            st.markdown(f"**Features Ótimas:**")
            st.code(', '.join(features), language='text')

        with col_score:
            r2_score = final_solution.get('r2_score', 0)
            feature_count = final_solution.get('feature_count', 0)
            st.metric("Score R² Ótimo", f"{r2_score:.4f}")
            st.metric("Taxa de Redução de Features", f"{(10 - feature_count) / 10 * 100:.1f}%")

        st.info("O Branch and Bound encontrou o **melhor trade-off** entre desempenho e simplicidade. O R² de 0.3344 é o máximo que se pode obter **respeitando a restrição de budget** (Nº de Features) e **superando a meta mínima** (0.30).")
        
        st.write("---")


        st.subheader("3.2 Comparação de Desempenho (R² vs. Nº de Features)")
        
        # NOTA EXPLICATIVA ADICIONADA AQUI PARA CLARIFICAR A DIFERENÇA
        st.info("É importante notar que o Branch and Bound busca o *ótimo global* respeitando uma **restrição de budget** (número máximo de features e meta mínima de R²). A Heurística Gulosa é executada em todos os 10 passos para mostrar sua performance sem essa restrição, por isso pode atingir um R² superior ao usar mais features.")
        
        greedy_steps = bnb_heuristic_comp.get('greedy_heuristic_steps', [])
        
        df_greedy = pd.DataFrame(greedy_steps)
        if not df_greedy.empty:
            df_greedy['Método'] = 'Heurística Gulosa (Greedy)'
            
            bnb_optimal_data = bnb_heuristic_comp.get('bnb_optimal', {})
            df_bnb_optimal = pd.DataFrame([{
                'feature_count': bnb_optimal_data.get('feature_count', 0),
                'r2_score': bnb_optimal_data.get('r2_score', 0),
                'features': bnb_optimal_data.get('features', []),
                'Método': 'Branch and Bound (Ótimo Global)'
            }])
            
            # Garante que o ponto do B&B (2 features) seja incluído na comparação, mesmo que a heurística atinja R² mais alto depois.
            df_comparison = pd.concat([df_greedy, df_bnb_optimal], ignore_index=True).drop_duplicates(subset=['feature_count', 'r2_score', 'Método'])

            fig_comparison = px.line(
                df_comparison,
                x='feature_count',
                y='r2_score',
                color='Método',
                markers=True,
                title='Comparação: Score R² por Nº de Features (B&B vs. Heurística)',
                color_discrete_map={
                    'Branch and Bound (Ótimo Global)': '#4285F4',
                    'Heurística Gulosa (Greedy)': '#FBBC04'
                }
            )

            fig_comparison.update_layout(
                xaxis_title='Número de Features no Modelo',
                yaxis_title='Score R² (Qualidade do Modelo)',
                legend_title='Método',
                hovermode='x unified'
            )
            
            fig_comparison.update_xaxes(dtick=1)
            
            st.plotly_chart(fig_comparison, use_container_width=True)

            st.caption("A Heurística Gulosa (Forward Selection) adiciona a feature que mais melhora o R² em cada passo, sem reavaliar as features anteriores.")
            
            st.write("---")
            st.subheader("Análise Detalhada dos Passos da Heurística")
            
            df_greedy_display = df_greedy.rename(columns={
                'feature_count': 'Nº Features',
                'r2_score': 'R² Score'
            })
            df_greedy_display['Features Incluídas'] = df_greedy_display['features'].apply(lambda x: ', '.join(x))
            
            st.dataframe(df_greedy_display[['Nº Features', 'R² Score', 'Features Incluídas']], hide_index=True, use_container_width=True)
            
        else:
            st.warning("Dados de comparação da heurística não puderam ser carregados.")

    else:
        st.warning("Dados de resumo ou comparação da heurística não puderam ser carregados. Verifique se os caminhos dos arquivos JSON estão corretos (deve ser: '../feature_selection/nome_do_arquivo.json').")