"""AED TikTok Shop — requer dados reais do questionário A2.
Execução: pip install pandas matplotlib; python analise_exploratoria_tiktok_shop.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO = Path('dados/respostas_tiktok_shop.csv')
SAIDA = Path('resultados')
COLUNAS = ['faixa_etaria','tempo_tiktok','frequencia_uso','viu_produto',
           'acessou_shop','comprou_shop','numero_compras','descoberta_sem_busca',
           'compra_nao_planejada','influencia_criador','influencia_desconto',
           'intencao_nova_compra']
BINARIAS = ['viu_produto','acessou_shop','comprou_shop',
            'descoberta_sem_busca','compra_nao_planejada']
ESCALAS = ['influencia_criador','influencia_desconto','intencao_nova_compra']

def executar():
    if not ARQUIVO.exists():
        raise FileNotFoundError(f'Insira a base real em {ARQUIVO}; não há dados simulados.')
    SAIDA.mkdir(exist_ok=True)
    df = pd.read_csv(ARQUIVO)
    faltantes = sorted(set(COLUNAS) - set(df.columns))
    if faltantes:
        raise ValueError(f'Colunas ausentes: {faltantes}')
    df = df[COLUNAS].copy()
    for c in BINARIAS:
        df[c] = df[c].astype('string').str.strip().str.lower().replace({
            'sim':'1','não':'0','nao':'0','true':'1','false':'0'})
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ESCALAS + ['numero_compras']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    print('Dimensão:', df.shape, '| Duplicatas completas:',df.duplicated().sum())
    df.isna().sum().rename('ausentes').to_csv(SAIDA/'ausentes.csv')
    pd.DataFrame({'tipo':df.dtypes.astype(str),'ausentes':df.isna().sum(),
                  'percentual_ausente':df.isna().mean().mul(100).round(2)}).to_csv(SAIDA/'qualidade.csv')
    alertas = {}
    for c in BINARIAS:
        alertas[c] = int((df[c].notna() & ~df[c].isin([0,1])).sum())
    for c in ESCALAS:
        alertas[c] = int((df[c].notna() & ~df[c].isin([1,2,3,4,5])).sum())
    alertas['numero_compras'] = int((df.numero_compras.notna() & ((df.numero_compras<0) | (df.numero_compras%1!=0))).sum())
    alertas['comprou_sem_compras'] = int(((df.comprou_shop==1)&(df.numero_compras==0)).sum())
    pd.Series(alertas,name='registros_suspeitos').to_csv(SAIDA/'alertas_validacao.csv')
    df[ESCALAS+['numero_compras']].describe().to_csv(SAIDA/'estatisticas.csv')
    for c in ['faixa_etaria','frequencia_uso']+BINARIAS:
        df[c].value_counts(dropna=False).rename('n').to_csv(SAIDA/f'frequencia_{c}.csv')
    pares = [('frequencia_uso','viu_produto'),('frequencia_uso','comprou_shop'),
             ('descoberta_sem_busca','compra_nao_planejada'),('faixa_etaria','comprou_shop')]
    for a,b in pares:
        pd.crosstab(df[a],df[b],normalize='index').mul(100).round(2).to_csv(SAIDA/f'cruzamento_{a}_{b}.csv')
    for c in ['frequencia_uso','viu_produto','comprou_shop','influencia_criador','influencia_desconto']:
        fig,ax=plt.subplots(figsize=(8,4))
        df[c].value_counts(dropna=False).sort_index().plot.bar(ax=ax)
        ax.set_title(f'Distribuição: {c}');ax.set_xlabel(c);ax.set_ylabel('Respostas')
        fig.tight_layout();fig.savefig(SAIDA/f'grafico_{c}.png',dpi=160);plt.close(fig)
    print('Arquivos gerados em',SAIDA.resolve())

if __name__=='__main__':
    executar()
