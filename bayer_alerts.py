import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def gerar_dados_incendio(n_records=1000):
    """
    Gera dados sintéticos para treinar a Rede Bayesiana.
    Inspirado na secção 2 do notebook (generate_patient_data).
    """
    data = []
    for _ in range(n_records):
        temp = np.random.normal(25, 8)
        hum = np.random.normal(60, 20) - (temp * 0.5)
        wind = np.random.normal(20, 15)
        
        score = 0
        if temp > 30: score += 2
        if temp > 38: score += 3
        if hum < 40: score += 1
        if hum < 20: score += 3
        if wind > 40: score += 1
        if wind > 60: score += 2
        
        if score > 5:
            risco = 'Alto'
        elif score > 2:
            risco = 'Medio'
        else:
            risco = 'Baixo'
            
        if np.random.rand() < 0.1:
            risco = np.random.choice(['Baixo', 'Medio', 'Alto'])
            
        data.append({
            'temp': np.clip(temp, 0, 50),
            'hum': np.clip(hum, 10, 100),
            'wind': np.clip(wind, 0, 100),
            'RiscoIncendio': risco
        })
        
    df = pd.DataFrame(data)
    print(f"✓ {len(df)} registos sintéticos gerados para a Rede Bayesiana.")
    return df

def discretizar_dados(df):
    """
    Discretiza os dados contínuos em categorias.
    Inspirado na secção 5.1 do notebook (discretizar_dados).
    """
    df_discreto = df.copy()
    
    # Discretiza Temperatura (Calor)
    df_discreto['Calor'] = pd.cut(
        df['temp'],
        bins=[0, 30, 38, 51],
        labels=['Normal', 'Alto', 'Extremo'],
        right=False
    )
    
    # Discretiza Humidade
    df_discreto['Humidade'] = pd.cut(
        df['hum'],
        bins=[0, 30, 60, 101],
        labels=['Seco', 'Normal', 'Humido'],
        right=False
    )
    
    # Discretiza Vento
    df_discreto['Vento'] = pd.cut(
        df['wind'],
        bins=[0, 30, 60, 101],
        labels=['Fraco', 'Moderado', 'Forte'],
        right=False
    )
    
    print("✓ Dados discretizados em categorias.")
    return df_discreto[['Calor', 'Humidade', 'Vento', 'RiscoIncendio']]

if __name__ == "__main__":
    
    df_continuo = gerar_dados_incendio(n_records=2000)
    df_treino = discretizar_dados(df_continuo)
    df_treino = df_treino.dropna()

    modelo_bn = DiscreteBayesianNetwork([
        ('Calor', 'RiscoIncendio'),
        ('Humidade', 'RiscoIncendio'),
        ('Vento', 'RiscoIncendio')
    ])
    
    print(f"\nEstrutura da Rede Bayesiana definida com {len(modelo_bn.nodes())} nós.")
    print(f"Nós: {modelo_bn.nodes()}")
    print(f"Arestas: {modelo_bn.edges()}")

    print("\nA treinar modelo (a estimar CPDs)...")
    modelo_bn.fit(df_treino, estimator=MaximumLikelihoodEstimator)
    print("✓ Modelo treinado.")

    print("\n--- CPDs (Tabelas de Probabilidade Condicional) ---")
    print(modelo_bn.get_cpds('Calor'))
    print("-" * 30)
    print(modelo_bn.get_cpds('Humidade'))
    print("-" * 30)
    print(modelo_bn.get_cpds('Vento'))
    print("-" * 30)
    print(f"CPD 'RiscoIncendio' (depende de {len(modelo_bn.get_cpds('RiscoIncendio').variables)-1} variáveis)")
    
    inferencia = VariableElimination(modelo_bn)
    
    print("\n--- EXEMPLOS DE INFERÊNCIA ---")
    
    print("\n[1] P(Risco | Calor=Extremo, Humidade=Seco)")
    resultado_1 = inferencia.query(
        variables=['RiscoIncendio'],
        evidence={'Calor': 'Extremo', 'Humidade': 'Seco'}
    )
    print(resultado_1)

    print("\n[2] P(Risco | Calor=Extremo, Humidade=Seco, Vento=Forte)")
    resultado_2 = inferencia.query(
        variables=['RiscoIncendio'],
        evidence={'Calor': 'Extremo', 'Humidade': 'Seco', 'Vento': 'Forte'}
    )
    print(resultado_2)

    print("\n[3] P(Calor | RiscoIncendio=Alto)")
    resultado_3 = inferencia.query(
        variables=['Calor'],
        evidence={'RiscoIncendio': 'Alto'}
    )
    print(resultado_3)
    
    print("\n[4] P(Humidade | RiscoIncendio=Alto)")
    resultado_4 = inferencia.query(
        variables=['Humidade'],
        evidence={'RiscoIncendio': 'Alto'}
    )
    print(resultado_4)