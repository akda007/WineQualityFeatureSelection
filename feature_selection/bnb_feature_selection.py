import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
import time
import json

# --- Configuração do Problema ---
# "meta" (restrição)
MINIMUM_R2_SCORE = 0.30

# --- Variáveis Globais para Rastreamento ---
best_solution_features = []
best_solution_feature_count = float('inf')
nodes_visited = 0

# --- Estruturas de Dados para Exportação ---
# Irá armazenar o log de cada nó para construir a árvore
tree_data_log = []
# Irá armazenar cada solução viável encontrada
solutions_found_log = []
# Contador para IDs de nós únicos
node_id_counter = 0

# --- Carregar e Preparar os Dados ---
try:
    # Ajuste o caminho conforme necessário
    df = pd.read_csv("../dataset_cleaning/wine_clean.csv")
except FileNotFoundError:
    print(f"Erro: Ficheiro './dataset_cleaning/wine_clean.csv' não encontrado.")
    sys.exit()
except Exception as e:
    print(f"Erro ao ler os dados: {e}")
    sys.exit()

TARGET_VARIABLE = 'quality'
ALL_FEATURES = [col for col in df.columns if col not in ['Id', TARGET_VARIABLE]]

try:
    X_all = df[ALL_FEATURES]
    y_all = df[TARGET_VARIABLE]
except KeyError:
    print("Erro: Colunas não encontradas. Verifique o CSV.")
    sys.exit()

# --- Função de Avaliação ("Custo" de um Nó) ---
def train_and_evaluate(features_to_use: list) -> float:
    """
    Treina um modelo e retorna o seu score R2.
    Incrementa o contador global de nós visitados.
    """
    global nodes_visited
    nodes_visited += 1
    
    # Se a lista de features estiver vazia, retorna um score péssimo
    if not features_to_use:
        return -float('inf')
    
    try:
        X = X_all[features_to_use]
        y = y_all
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        score = r2_score(y, predictions)
        return score
    
    except Exception as e:
        print(f"Erro durante o treino com features {features_to_use}: {e}")
        return -float('inf')

# --- O Algoritmo Branch and Bound (Modificado para Log) ---
def solve_feature_selection_bnb(
    current_feature_index: int,
    current_features_list: list,
    parent_id: int,
    decision_text: str
):
    """
    Função B&B recursiva que agora regista cada nó visitado
    para a exportação da árvore.
    """
    global best_solution_feature_count, best_solution_features
    global node_id_counter, tree_data_log, solutions_found_log

    # --- Setup do Nó Atual ---
    node_id = node_id_counter
    node_id_counter += 1
    
    node_log = {
        "id": node_id,
        "parent_id": parent_id,
        "decision": decision_text,
        "features": list(current_features_list),
        "feature_count": len(current_features_list),
        "score": -1.0, # Score será atualizado abaixo
        "status": "" # Status será definido antes de retornar
    }

    # --- 1. LÓGICA DE PODA (PRUNING) B&B ---
    # Poda por Limite (Bound): Se já temos mais features que a nossa melhor solução, podar.
    if len(current_features_list) >= best_solution_feature_count:
        node_log["status"] = "PODADO_BOUND"
        tree_data_log.append(node_log)
        return

    # --- 2. AVALIAR O NÓ ATUAL ---
    model_score = train_and_evaluate(current_features_list)
    # Converter -inf para None (que se tornará 'null' no JSON)
    node_log["score"] = model_score if model_score != -float('inf') else None

    # --- 3. VERIFICAR A RESTRIÇÃO ("META") ---
    if model_score >= MINIMUM_R2_SCORE:
        # Encontrámos uma solução viável
        solution_data = {
            "features": list(current_features_list),
            "feature_count": len(current_features_list),
            "score": model_score
        }
        solutions_found_log.append(solution_data)

        # É uma *nova melhor* solução?
        if len(current_features_list) < best_solution_feature_count:
            print("*" * 40)
            print(f"*** NOVA MELHOR SOLUÇÃO ENCONTRADA! ***")
            print(f"*** N.º de Features: {len(current_features_list)}")
            print(f"*** Features: {current_features_list}")
            print(f"*** Score R2: {model_score:.4f}")
            print("*" * 40)
            
            best_solution_feature_count = len(current_features_list)
            best_solution_features = list(current_features_list)
            
            node_log["status"] = "SOLUCAO_OTIMA_ATUAL"
            
        else:
            # É uma solução viável, mas não é melhor que a que já temos (tem mais features)
            node_log["status"] = "PODADO_SOLUCAO_PIOR"

        tree_data_log.append(node_log)
        # Paramos de explorar este ramo (não queremos adicionar mais features)
        return

    # --- 4. CONDIÇÃO DE PARAGEM (FIM DA ÁRVORE) ---
    # Chegámos ao fim (folha) sem atingir a meta
    if current_feature_index >= len(ALL_FEATURES):
        node_log["status"] = "FOLHA_INVALIDA"
        tree_data_log.append(node_log)
        return

    # --- 5. LÓGICA DE RAMIFICAÇÃO (BRANCHING) ---
    # Se chegámos aqui, o nó será expandido
    node_log["status"] = "EXPLORADO"
    tree_data_log.append(node_log)
    
    next_feature = ALL_FEATURES[current_feature_index]
    next_index = current_feature_index + 1

    # Ramo 1: "NÃO INCLUIR" a próxima feature
    solve_feature_selection_bnb(
        current_feature_index=next_index,
        current_features_list=current_features_list,
        parent_id=node_id,
        decision_text=f"NÃO {next_feature}"
    )

    # Ramo 2: "INCLUIR" a próxima feature
    new_features_list = current_features_list + [next_feature]
    solve_feature_selection_bnb(
        current_feature_index=next_index,
        current_features_list=new_features_list,
        parent_id=node_id,
        decision_text=f"INCLUIR {next_feature}"
    )

# --- Heurística Gulosa (Greedy) para Comparação ---
def run_greedy_heuristic():
    """
    Executa uma heurística gulosa (forward selection) para
    comparar com o resultado ótimo do B&B.
    """
    print("\nA executar a Heurística Gulosa (Greedy) para comparação...")
    greedy_steps_log = []
    selected_features = []
    best_score_so_far = -float('inf')
    
    # Itera para adicionar uma feature de cada vez
    for i in range(len(ALL_FEATURES)):
        best_new_feature = None
        best_score_this_step = -float('inf')
        
        # Testa todas as features restantes
        for feature in ALL_FEATURES:
            if feature not in selected_features:
                current_test_features = selected_features + [feature]
                score = train_and_evaluate(current_test_features)
                
                if score > best_score_this_step:
                    best_score_this_step = score
                    best_new_feature = feature
        
        # Se o score melhorou, adiciona a feature
        if best_score_this_step > best_score_so_far:
            best_score_so_far = best_score_this_step
            selected_features.append(best_new_feature)
            
            step_data = {
                "feature_count": len(selected_features),
                "r2_score": best_score_so_far if best_score_so_far != -float('inf') else None,
                "features": list(selected_features)
            }
            greedy_steps_log.append(step_data)
            print(f"  Greedy Step {len(selected_features)}: Score {best_score_so_far:.4f} com {selected_features}")
        else:
            # O score não melhorou, paramos
            break
            
    print("Heurística Gulosa completa.")
    return greedy_steps_log

# --- Função Principal de Execução e Exportação ---
def main():
    global nodes_visited # Resetar o contador para não contar a heurística
    
    # 1. Executar Heurística
    nodes_visited = 0 # Não contar visitas da heurística no B&B
    greedy_results = run_greedy_heuristic()
    
    # 2. Executar B&B
    print("\n" + "-" * 40)
    print("A iniciar o Branch and Bound para Feature Selection...")
    print(f"Objetivo: Minimizar features")
    print(f"Restrição (Meta): R2 Score >= {MINIMUM_R2_SCORE}")
    print(f"Total de features para testar: {len(ALL_FEATURES)}")
    print("-" * 40)

    nodes_visited = 0 # Resetar contador para o B&B
    start_time = time.time()
    
    # Iniciar a busca a partir da raiz
    solve_feature_selection_bnb(
        current_feature_index=0,
        current_features_list=[],
        parent_id=-1, # -1 indica que é a raiz
        decision_text="RAIZ"
    )
    
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 40)
    print("B&B COMPLETO.")
    print(f"Tempo Total: {total_time:.2f} segundos")
    print(f"Total de modelos treinados (nós visitados): {nodes_visited}")
    print(f"Total de soluções viáveis encontradas: {len(solutions_found_log)}")
    
    final_solution = {}
    if best_solution_feature_count != float('inf'):
        print(f"MELHOR SOLUÇÃO (ÓTIMA):")
        print(f"  Features: {best_solution_features}")
        print(f"  N.º de Features: {best_solution_feature_count}")
        
        # Encontrar o score da melhor solução (está no log)
        final_score = 0
        for sol in solutions_found_log:
             if sol["features"] == best_solution_features:
                final_score = sol["score"]
                break
        
        final_solution = {
            "features": best_solution_features,
            "feature_count": best_solution_feature_count,
            "r2_score": final_score
        }
    else:
        print("Nenhuma solução encontrada que atinja a meta de R2.")
        print(f"Tente baixar o valor de 'MINIMUM_R2_SCORE' (atualmente {MINIMUM_R2_SCORE}).")

    # 3. Exportar Ficheiros JSON
    print("\nA exportar ficheiros JSON para o dashboard...")

    # Ficheiro 1: A Árvore de Busca Completa
    try:
        with open('export_bnb_tree.json', 'w', encoding='utf-8') as f:
            json.dump(tree_data_log, f, indent=2, ensure_ascii=False)
        print("  - 'export_bnb_tree.json' (LOG DA ÁRVORE) ... OK")
    except Exception as e:
        print(f"  - ERRO ao exportar 'export_bnb_tree.json': {e}")

    # Ficheiro 2: Sumário da Execução do B&B
    summary_data = {
        "final_solution": final_solution,
        "execution_metrics": {
            "total_time_seconds": total_time,
            "nodes_visited": nodes_visited,
            "solutions_found_count": len(solutions_found_log),
            "r2_goal": MINIMUM_R2_SCORE
        },
        "solutions_timeline": solutions_found_log # Histórico de soluções encontradas
    }
    try:
        with open('export_bnb_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print("  - 'export_bnb_summary.json' (SUMÁRIO B&B) ... OK")
    except Exception as e:
        print(f"  - ERRO ao exportar 'export_bnb_summary.json': {e}")
        
    # Ficheiro 3: Comparação com Heurística
    heuristic_data = {
        "bnb_optimal": final_solution,
        "greedy_heuristic_steps": greedy_results
    }
    try:
        with open('export_heuristic_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(heuristic_data, f, indent=2, ensure_ascii=False)
        print("  - 'export_heuristic_comparison.json' (COMPARAÇÃO HEURÍSTICA) ... OK")
    except Exception as e:
        print(f"  - ERRO ao exportar 'export_heuristic_comparison.json': {e}")

if __name__ == "__main__":
    main()