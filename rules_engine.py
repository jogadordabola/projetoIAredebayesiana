import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class MotorInferencia:
    """
    Motor de inferência baseado em regras para detecção de risco de incêndio.
    Inspirado na classe SistemaAlarmes do notebook de UCI.
    """

    def __init__(self, ficheiro_regras):
        """
        Carrega a base de conhecimentos (regras) a partir de um ficheiro JSON.
        """
        print(f"A carregar regras de '{ficheiro_regras}'...")
        try:
            with open(ficheiro_regras, 'r', encoding='utf-8') as f:
                self.regras = json.load(f)
            self.regras.sort(key=lambda r: r['prioridade'])
            print(f"✓ {len(self.regras)} regras carregadas e ordenadas por prioridade.")
        except FileNotFoundError:
            print(f"ERRO: Ficheiro de regras '{ficheiro_regras}' não encontrado.")
            self.regras = []
        except json.JSONDecodeError:
            print(f"ERRO: Ficheiro de regras '{ficheiro_regras}' não é um JSON válido.")
            self.regras = []

    def _verificar_condicao(self, alerta_valor, operador, regra_valor):
        """
        Verifica uma única condição da regra.
        """
        if operador == '>':
            return alerta_valor > regra_valor
        if operador == '<':
            return alerta_valor < regra_valor
        if operador == '==':
            return alerta_valor == regra_valor
        if operador == '!=':
            return alerta_valor != regra_valor
        if operador == '>=':
            return alerta_valor >= regra_valor
        if operador == '<=':
            return alerta_valor <= regra_valor
        return False

    def avaliar_alerta(self, alerta_row):
        """
        Avalia um único alerta (linha do DataFrame) contra a base de regras.
        Retorna o resultado da primeira regra que der "match".
        """
        for regra in self.regras:
            condicoes_cumpridas = True
            for condicao in regra['condicoes']:
                variavel = condicao['variavel']
                if variavel not in alerta_row:
                    condicoes_cumpridas = False
                    break

                alerta_valor = alerta_row[variavel]
                operador = condicao['operador']
                regra_valor = condicao['valor']

                if not self._verificar_condicao(alerta_valor, operador, regra_valor):
                    condicoes_cumpridas = False
                    break

            if condicoes_cumpridas:
                return regra['resultado']['risco'], regra['resultado']['acao'], regra['id']

        return 'NORMAL', 'Monitorização de rotina.', 'SEM_REGRA'

    def processar_dataset(self, ficheiro_csv):
        """
        Lê um dataset (alerts.csv) e aplica o motor de inferência a cada linha.
        Inspirado na secção 4.2 do notebook.
        """
        if not self.regras:
            print("ERRO: Não há regras carregadas. A processar...")
            return None

        try:
            df = pd.read_csv(ficheiro_csv)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"\nA processar '{ficheiro_csv}' com {len(df)} registos...")
        except FileNotFoundError:
            print(f"ERRO: Ficheiro '{ficheiro_csv}' não encontrado.")
            return None
        except Exception as e:
            print(f"ERRO ao ler o CSV: {e}")
            return None

        resultados = df.apply(
            lambda row: self.avaliar_alerta(row),
            axis=1,
            result_type='expand'
        )
        
        resultados.columns = ['risco', 'acao_recomendada', 'regra_ativada']
        
        df_resultado = pd.concat([df, resultados], axis=1)
        
        print("✓ Processamento concluído.")
        return df_resultado

def gerar_dados_simulados(filename='alerts.csv', n_records=100):
    """
    Gera dados sintéticos de alertas de incêndio.
    Inspirado na secção 2 do notebook.
    """
    print(f"A gerar dados simulados para '{filename}'...")
    start_date = datetime(2024, 7, 1)
    data = []
    event_types = ['nenhum', 'fogueira_descontrolada', 'raio_seco', 'nenhum', 'nenhum']
    
    for i in range(n_records):
        timestamp = start_date + timedelta(hours=i*3)
        zona = np.random.choice(['Serra da Estrela', 'Monchique', 'Peneda-Gerês', 'Sintra'])
        
        temp = np.random.normal(30, 8)
        hum = np.random.normal(40, 15)
        wind = np.random.normal(30, 15)
        event_type = np.random.choice(event_types, p=[0.8, 0.05, 0.05, 0.05, 0.05])
        
        if i % 20 == 0:
            temp = 42
            hum = 18
            
        if i % 15 == 0:
            event_type = 'raio_seco'

        data.append({
            'timestamp': timestamp,
            'zone': zona,
            'temp': round(np.clip(temp, 15, 50), 1),
            'hum': round(np.clip(hum, 10, 90), 1),
            'wind': round(np.clip(wind, 0, 80), 1),
            'event_type': event_type
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✓ Ficheiro '{filename}' gerado com {len(df)} registos.")
    return filename

if __name__ == "__main__":
    
    try:
        with open('regras.json', 'x', encoding='utf-8') as f:
            json_content = """
            [
              {"id": "REGRA_CRITICA_01", "prioridade": 1, "descricao": "Temp > 40 E Hum < 20", "condicoes": [{"variavel": "temp", "operador": ">", "valor": 40}, {"variavel": "hum", "operador": "<", "valor": 20}], "resultado": {"risco": "CRÍTICO", "acao": "Mobilização imediata."}},
              {"id": "REGRA_ALTA_01", "prioridade": 2, "descricao": "Evento 'raio_seco'", "condicoes": [{"variavel": "event_type", "operador": "==", "valor": "raio_seco"}], "resultado": {"risco": "ALTO", "acao": "Enviar vigilância."}},
              {"id": "REGRA_ALTA_02", "prioridade": 2, "descricao": "Temp > 38 E Hum < 30", "condicoes": [{"variavel": "temp", "operador": ">", "valor": 38}, {"variavel": "hum", "operador": "<", "valor": 30}], "resultado": {"risco": "ALTO", "acao": "Reforçar vigilância."}},
              {"id": "REGRA_MEDIA_01", "prioridade": 3, "descricao": "Temp > 35", "condicoes": [{"variavel": "temp", "operador": ">", "valor": 35}], "resultado": {"risco": "MÉDIO", "acao": "Aviso à população."}},
              {"id": "REGRA_BAIXA_01", "prioridade": 4, "descricao": "Vento > 40", "condicoes": [{"variavel": "wind", "operador": ">", "valor": 40}], "resultado": {"risco": "BAIXO", "acao": "Monitorizar."}}
            ]
            """
            f.write(json_content)
            print("(!) Ficheiro 'regras.json' de exemplo criado.")
    except FileExistsError:
        pass
    
    ficheiro_dados = gerar_dados_simulados('alerts.csv', n_records=100)
    
    motor = MotorInferencia(ficheiro_regras='regras.json')
    
    df_final = motor.processar_dataset(ficheiro_dados)
    
    if df_final is not None:
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("\n--- AMOSTRA DE TODOS OS RESULTADOS (Top 10) ---")
        print(df_final[['timestamp', 'zone', 'temp', 'hum', 'wind', 'risco', 'acao_recomendada']].head(10))
        
        print("\n--- ESTATÍSTICAS GERAIS DE RISCO ---")
        print(df_final['risco'].value_counts())

        print("\n" + "="*50)
        print("     ⚠️  RELATÓRIO DE AÇÕES RECOMENDADAS ⚠️")
        print("="*50)
        
        casos_com_acao = df_final[df_final['risco'] != 'NORMAL'].copy()
        
        if casos_com_acao.empty:
            print("\nNenhum alerta de risco (Baixo, Médio, Alto ou Crítico) foi ativado.")
        else:
            print(f"\nDetetados {len(casos_com_acao)} alertas que requerem ação:")
            
            casos_com_acao['risco_num'] = casos_com_acao['risco'].map({'CRÍTICO': 1, 'ALTO': 2, 'MÉDIO': 3, 'BAIXO': 4})
            casos_com_acao = casos_com_acao.sort_values(by=['risco_num', 'timestamp'])
            
            for index, row in casos_com_acao.iterrows():
                print("\n" + "-"*30)
                print(f"  Data/Hora: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                print(f"  Zona:      {row['zone']}")
                print(f"  RISCO:     {row['risco']} (Regra: {row['regra_ativada']})")
                print(f"  Condições: Temp={row['temp']}C, Hum={row['hum']}%, Vento={row['wind']}km/h, Evento='{row['event_type']}'")
                print(f"  AÇÃO:      {row['acao_recomendada']}")
        
        print("\n" + "="*50)