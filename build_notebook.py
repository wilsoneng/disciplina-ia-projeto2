"""Gerador programático do notebook corrigido (v3) para o Projeto 2 (R6 Air Quality).

Uso:
    python build_notebook.py              -> gera P2_AirQuality_v3.ipynb (RODAR_BUSCAS=True)
    python build_notebook.py --modo-false -> gera P2_AirQuality_v3_modoFalse.ipynb (RODAR_BUSCAS=False)
    P2_SMOKE=1 python build_notebook.py   -> gera P2_AirQuality_v3_smoke.ipynb (teste rápido)
"""

import argparse
import os
import nbformat as nbf


def montar_celulas(rodar_buscas: bool = True, smoke: bool = False):
    celulas = []

    def md(texto: str):
        celulas.append(nbf.v4.new_markdown_cell(texto.strip() + "\n"))

    def code(texto: str):
        celulas.append(nbf.v4.new_code_cell(texto.strip() + "\n"))

    # ------------------------------------------------------------------
    # CABEÇALHO
    # ------------------------------------------------------------------
    md(
        """
# Projeto 2 · Comparação de Modelos Supervisionados · R6 Air Quality (UCI 360)

**Disciplina:** Inteligência Artificial · POLI/UPE · Prof. Emmanuel Andrade  
**Tarefa:** Regressão horária de `CO(GT)` (mg/m³) a partir de sensores de baixo custo e variáveis meteorológicas  
**Dataset:** R6 Air Quality, UCI Machine Learning Repository (ID 360), DOI `10.24432/C59K5F`, licença CC BY 4.0  
**Referência original:** De Vito, S. et al. (2008). *Sensors and Actuators B: Chemical*, 129(2), 750–757.

## Como executar este notebook
- `RODAR_BUSCAS = True`: executa do zero a validação cruzada, a busca de hiperparâmetros, a comparação de reduções de dimensionalidade, a avaliação **única** no conjunto de teste e grava todos os CSVs em `resultados/` e figuras em `figuras/`.
- `RODAR_BUSCAS = False`: **não** executa buscas exaustivas e **não** lê o conjunto de teste; carrega os CSVs pré-computados (de `URL_BASE` pública ou, com aviso, da pasta local `resultados/`), reajusta apenas o modelo campeão no conjunto de treino e gera todas as tabelas e figuras em poucos minutos.

## Regras de protocolo seguidas
1. **Sem vazamento de dados:** a divisão treino/teste (80/20, `random_state=42`) ocorre após o filtro estrutural de alvo ausente (`CO(GT) = -200`) e **antes** da análise exploratória (Seção 3). Todas as estatísticas, correlações e decisões usam apenas o treino.
2. **Sem vazamento de cenário:** outros analisadores de referência (`C6H6(GT)`, `NOx(GT)`, `NO2(GT)`) são excluídos, pois não estariam disponíveis em campo sem estação certificada; `NMHC(GT)` (~88% ausente no treino) é descartada.
3. **Pipeline completo:** engenharia de atributos (`AtributosAirQuality`), imputação pela mediana (`SimpleImputer`), padronização (`StandardScaler`) e redução (`PCA` / `SelectKBest`) residem dentro de `sklearn.pipeline.Pipeline` e são ajustados somente no treino de cada dobra.
4. **Escolha exclusivamente por CV:** o modelo entregue é escolhido apenas por métricas de validação cruzada (nunca pelo teste). O teste é lido uma única vez na Seção 9, exatamente nas 10 configurações da Tabela T1.
5. **Semente fixa (`42`)** na divisão, validação cruzada, buscas aleatórias e estimadores.
"""
    )

    # ------------------------------------------------------------------
    # 1. CONFIGURAÇÃO
    # ------------------------------------------------------------------
    md(
        """
## 1. Configuração

Instalação opcional de `ucimlrepo` (para Google Colab), importação de bibliotecas, impressão das versões do ambiente, chave `RODAR_BUSCAS`, definição da paleta fixa por modelo e formatação global de figuras (fonte mínima 16 pt).
"""
    )

    code(
        """
try:
    import ucimlrepo
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"])
    import ucimlrepo
"""
    )

    pasta_res = "resultados_smoke" if smoke else "resultados"
    pasta_fig = "figuras_smoke" if smoke else "figuras"

    code(
        f"""
import os
import sys
import ast
import time
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.stats import loguniform, randint

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, RandomizedSearchCV, validation_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from ucimlrepo import fetch_ucirepo

# ElasticNet com alpha pequeno pode emitir ConvergenceWarning mesmo com max_iter=20000;
# filtramos avisos repetitivos de convergência/bibliotecas para manter a saída legível.
warnings.filterwarnings("ignore")

RODAR_BUSCAS = {str(rodar_buscas)}
SEED = 42
np.random.seed(SEED)

PASTA_RESULTADOS = "{pasta_res}"
PASTA_FIGURAS = "{pasta_fig}"
os.makedirs(PASTA_RESULTADOS, exist_ok=True)
os.makedirs(PASTA_FIGURAS, exist_ok=True)

# Para entrega pública com RODAR_BUSCAS = False, hospede os CSVs de '{pasta_res}/'
# (ex.: GitHub raw) e substitua URL_BASE pela URL pública terminando em '/'.
URL_BASE = "https://raw.githubusercontent.com/wilsoneng/disciplina-ia-projeto2/main/resultados/"

ARQUIVOS_CSV = [
    "resultados_configuracoes.csv",
    "melhores_hiperparametros.csv",
    "cv_grade_completa.csv",
    "predicoes_teste.csv",
    "curva_validacao.csv",
    "cv_reducao.csv",
    "efeito_atributos.csv",
    "analise_temporal_cv.csv",
]


def caminho_csv(nome):
    return os.path.join(PASTA_RESULTADOS, nome)


def ler_csv(nome):
    if URL_BASE.startswith("COLE_AQUI"):
        local = caminho_csv(nome)
        if not os.path.exists(local):
            raise FileNotFoundError(
                f"Arquivo '{{local}}' não encontrado e URL_BASE ainda é placeholder. "
                "Execute com RODAR_BUSCAS = True ou preencha URL_BASE com o link público."
            )
        print(
            f"[AVISO] URL_BASE não preenchida; lendo '{{local}}' localmente. "
            "Para entrega ao professor, hospede os CSVs de 'resultados/' e preencha URL_BASE."
        )
        return pd.read_csv(local)
    url = URL_BASE.rstrip("/") + "/" + nome
    return pd.read_csv(url)


# Formatação global obrigatória (Seção 10 do enunciado: fonte mínima 16 pt)
plt.rcParams.update({{
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.5,
    "savefig.dpi": 200,
}})
sns.set_palette("colorblind")


def salvar(nome):
    plt.savefig(os.path.join(PASTA_FIGURAS, nome + ".png"), bbox_inches="tight", dpi=200)
    plt.show()


MODELOS = ["ElasticNet", "RandomForest", "SVR"]
COR = {{
    "Baseline": "black",
    "ElasticNet": "tab:blue",
    "RandomForest": "tab:orange",
    "SVR": "tab:green",
}}
ALVO = "CO(GT)"
"""
    )

    code(
        """
# Versões registradas do ambiente de execução
versoes = {
    "Python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "ucimlrepo": getattr(ucimlrepo, "__version__", "instalado (sem __version__)"),
}
for k, v in versoes.items():
    print(f"{k:<14}: {v}")
"""
    )

    # ------------------------------------------------------------------
    # 2. CARGA DOS DADOS
    # ------------------------------------------------------------------
    md(
        """
## 2. Carga dos dados

Carga direta da fonte via `ucimlrepo` (`id=360`). Nesta seção são feitas apenas **checagens estruturais** sobre o dataset completo (formato, tipos, dicionário de dados, duplicatas, contagem do código sentinela `-200`, faixas físicas e consistência temporal). Nenhuma distribuição, correlação ou decisão de modelagem é calculada aqui.
"""
    )

    code(
        """
data = fetch_ucirepo(id=360)
df = data.data.features.copy()
if data.data.targets is not None:
    for c in data.data.targets.columns:
        if c not in df.columns:
            df[c] = data.data.targets[c]

print(f"Formato do dataset bruto: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("\\n--- Primeiras 10 linhas ---")
print(df.head(10).to_string())

meta = data.metadata
def _meta(k):
    return meta.get(k) if isinstance(meta, dict) else getattr(meta, k, None)

print("\\n--- Metadados UCI ---")
for k in ["dataset_doi", "license", "citation"]:
    print(f"{k}: {_meta(k)}")

# Garantir coerção numérica caso alguma coluna venha como string com vírgula decimal
for c in df.columns:
    if c in ("Date", "Time"):
        continue
    if df[c].dtype == object:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

print("\\n--- Tipos das colunas ---")
print(df.dtypes.to_string())

dicionario = pd.DataFrame([
    ["Date", "Data da medição", "m/d/aaaa", "Identificador temporal", "Usada para datetime/dia_semana; removida dos preditores"],
    ["Time", "Hora da medição", "hh:mm:ss", "Identificador temporal", "Usada para datetime/hora; removida dos preditores"],
    ["CO(GT)", "Concentração de CO", "mg/m³", "Analisador de referência", "ALVO"],
    ["PT08.S1(CO)", "Resposta do sensor de óxido metálico (CO)", "adimensional", "Sensor baixo custo", "Preditor"],
    ["NMHC(GT)", "Hidrocarbonetos não metânicos", "µg/m³", "Analisador de referência", "Descartada se ausência > 70% no treino"],
    ["C6H6(GT)", "Benzeno", "µg/m³", "Analisador de referência", "Excluída (vazamento de cenário)"],
    ["PT08.S2(NMHC)", "Resposta do sensor (NMHC)", "adimensional", "Sensor baixo custo", "Preditor"],
    ["NOx(GT)", "Óxidos de nitrogênio", "ppb", "Analisador de referência", "Excluída (vazamento de cenário)"],
    ["PT08.S3(NOx)", "Resposta do sensor (NOx)", "adimensional", "Sensor baixo custo", "Preditor"],
    ["NO2(GT)", "Dióxido de nitrogênio", "µg/m³", "Analisador de referência", "Excluída (vazamento de cenário)"],
    ["PT08.S4(NO2)", "Resposta do sensor (NO2)", "adimensional", "Sensor baixo custo", "Preditor"],
    ["PT08.S5(O3)", "Resposta do sensor (O3)", "adimensional", "Sensor baixo custo", "Preditor"],
    ["T", "Temperatura", "°C", "Meteorológica", "Preditor"],
    ["RH", "Umidade relativa", "%", "Meteorológica", "Preditor"],
    ["AH", "Umidade absoluta", "g/m³", "Meteorológica", "Preditor"],
], columns=["coluna", "significado", "unidade", "origem", "papel"])
print("\\n--- Dicionário de dados ---")
print(dicionario.to_string(index=False))
"""
    )

    code(
        """
# Checagens estruturais: duplicatas, código sentinela -200, faixas físicas e consistência temporal
print("Duplicatas de linhas (dataset completo):", int(df.duplicated().sum()))
print("Duplicatas sem Date/Time:", int(df.drop(columns=["Date", "Time"]).duplicated().sum()))

num_cols = list(df.select_dtypes("number").columns)
cnt200 = (df[num_cols] == -200).sum()
print("\\n--- Contagem do código de ausência (-200) por coluna (bruto) ---")
print(pd.DataFrame({"n_-200": cnt200, "%": (cnt200 / len(df) * 100).round(2)}).to_string())

# Faixas físicas fixas a priori (apenas para checagem de plausibilidade)
FAIXAS = {
    "CO(GT)": (0, 15),
    "PT08.S1(CO)": (0, 5000),
    "NMHC(GT)": (0, 2000),
    "C6H6(GT)": (0, 50),
    "PT08.S2(NMHC)": (0, 5000),
    "NOx(GT)": (0, 2000),
    "PT08.S3(NOx)": (0, 5000),
    "NO2(GT)": (0, 400),
    "PT08.S4(NO2)": (0, 5000),
    "PT08.S5(O3)": (0, 5000),
    "T": (-20, 50),
    "RH": (0, 100),
    "AH": (0, 3),
}

linhas_pl = []
for c in num_cols:
    s = df[c][df[c] != -200]
    lo, hi = FAIXAS[c]
    fora = int(((s < lo) | (s > hi)).sum())
    linhas_pl.append([c, f"[{lo}, {hi}]", int(s.notna().sum()), fora, round(100 * fora / max(len(s), 1), 2)])

fecha = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
horas = pd.to_timedelta(df["Time"].astype(str).str.replace(".", ":", regex=False), errors="coerce")
linhas_pl.append(["Date", "m/d/aaaa", len(df), int(fecha.isna().sum()), 0.0])
linhas_pl.append(["Time", "hh:mm:ss", len(df), int(horas.isna().sum()), 0.0])
print("\\n--- Checagem de plausibilidade física e formato ---")
print(pd.DataFrame(linhas_pl, columns=["coluna", "faixa_esperada", "n_validos", "n_fora", "%"]).to_string(index=False))

df["datetime"] = fecha + horas
assert df["datetime"].notna().all(), "Falha ao converter Date/Time para datetime."
assert df["datetime"].is_monotonic_increasing, "Ordem temporal crescente violada."
print(f"\\nPeríodo observado: {df['datetime'].min()} -> {df['datetime'].max()}")
print("Ordem temporal crescente: True")
print("Valores negativos fora do código -200:", {c: int(((df[c] < 0) & (df[c] != -200)).sum()) for c in num_cols})
print("Registros sem alvo válido (CO(GT) == -200):", int((df[ALVO] == -200).sum()))
print("Declaração: não há grupos de pacientes/entidades no R6; GroupShuffleSplit/GroupKFold não se aplicam.")
print("Declaração: todas as entradas usadas são numéricas/cíclicas; codificação categórica não se aplica.")
"""
    )

    # ------------------------------------------------------------------
    # 3. ANÁLISE EXPLORATÓRIA (SOMENTE TREINO)
    # ------------------------------------------------------------------
    md(
        """
## 3. Análise exploratória (somente conjunto de treino)

Para evitar qualquer vazamento de informação do conjunto de teste na análise exploratória e nas decisões de tratamento, realizamos primeiro o filtro estrutural de alvo inexistente (`CO(GT) == -200`) e uma **divisão preliminar 80/20 dos índices** (`random_state=42`). Todas as estatísticas, correlações, figuras (`F1`, `F2a`, `F2b`) e os cinco achados abaixo são computados **exclusivamente sobre `idx_tr`**. A Seção 4 formaliza a divisão e o protocolo de validação cruzada; os índices `idx_te` só serão lidos uma única vez na Seção 9.
"""
    )

    smoke_slice = "\nidx_tr = idx_tr[:1500]  # [SMOKE] subamostra para teste rápido\n" if smoke else ""
    code(
        f"""
# Divisão preliminar: filtro de dado inexistente (alvo == -200) e separação 80/20 dos índices
mask_alvo_ausente = df[ALVO] == -200
print(f"Linhas removidas por CO(GT) = -200: {{int(mask_alvo_ausente.sum())}} ({{100 * mask_alvo_ausente.mean():.2f}}%)")
df_f = df.loc[~mask_alvo_ausente].copy()
print("Linhas com alvo válido:", len(df_f))

idx_tr, idx_te = train_test_split(df_f.index, test_size=0.20, random_state=SEED){smoke_slice}
train_eda = df_f.loc[idx_tr].copy()
cols_num_eda = [c for c in train_eda.select_dtypes("number").columns if c != ALVO]
train_eda[cols_num_eda] = train_eda[cols_num_eda].replace(-200, np.nan)

print(f"Treino reservado para EDA: {{len(train_eda)}} amostras | Teste isolado: {{len(idx_te)}} amostras (não lido)")
"""
    )

    code(
        """
num_eda = train_eda.select_dtypes("number")
hora_eda = train_eda["datetime"].dt.hour.rename("hora")
dia_sem_eda = train_eda["datetime"].dt.dayofweek.rename("dia_semana")

# F1: Distribuição do alvo no treino (média e mediana)
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(train_eda[ALVO], bins=45, ax=ax, color="tab:blue", edgecolor="black")
ax.axvline(train_eda[ALVO].mean(), color="black", ls="--", label=f"Média = {train_eda[ALVO].mean():.2f} mg/m³")
ax.axvline(train_eda[ALVO].median(), color="tab:orange", ls=":", label=f"Mediana = {train_eda[ALVO].median():.2f} mg/m³")
ax.set_title("F1 — Distribuição de CO(GT) no conjunto de treino")
ax.set_xlabel("CO(GT) (mg/m³)")
ax.set_ylabel("Frequência (horas)")
ax.legend()
plt.tight_layout()
salvar("F1_distribuicao_alvo")

# Estatísticas descritivas (somente treino)
desc = pd.DataFrame({
    "média": num_eda.mean(),
    "mediana": num_eda.median(),
    "dp": num_eda.std(ddof=1),
    "min": num_eda.min(),
    "max": num_eda.max(),
    "Q1": num_eda.quantile(0.25),
    "Q3": num_eda.quantile(0.75),
    "assimetria": num_eda.skew(),
    "curtose": num_eda.kurt(),
})
desc["IQR"] = desc["Q3"] - desc["Q1"]
print("--- Estatísticas descritivas (treino) ---")
print(desc.round(3).to_string())

# Faltantes no treino (após mapear -200 -> NaN)
falt_tr = (train_eda[num_eda.columns].isna().mean() * 100).round(2).sort_values(ascending=False)
print("\\n--- Ausências no treino (%) ---")
print(falt_tr.to_string())

taxa_nmhc = float(train_eda["NMHC(GT)"].isna().mean())
DROP_NMHC = taxa_nmhc > 0.70
zeros_co = int((train_eda[ALVO] == 0).sum())
MAPE_OK = zeros_co == 0
print(f"\\nAusência de NMHC(GT) no treino: {100 * taxa_nmhc:.2f}% -> remover (>70%): {DROP_NMHC}")
print(f"Zeros em CO(GT) no treino: {zeros_co} -> MAPE reportado: {MAPE_OK}")

# Relação dos preditores com o alvo (Pearson e Spearman)
pear = num_eda.corr(method="pearson")[ALVO].drop(ALVO)
spear = num_eda.corr(method="spearman")[ALVO].drop(ALVO)
rel = pd.DataFrame({"pearson": pear, "spearman": spear})
rel = rel.reindex(rel["spearman"].abs().sort_values(ascending=False).index)
print("\\n--- Correlação com CO(GT) no treino ---")
print(rel.round(3).to_string())
"""
    )

    code(
        """
# F2a: Mapa de correlação de Spearman entre atributos (somente treino)
cols_f2a = [c for c in num_eda.columns if not (DROP_NMHC and c == "NMHC(GT)")]
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    train_eda[cols_f2a].corr(method="spearman"),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 16},
    cbar_kws={"label": "Correlação de Spearman"},
    ax=ax,
)
ax.set_title("F2a — Correlação de Spearman entre atributos (treino)")
plt.tight_layout()
salvar("F2a_correlacao")

# F2b: Padrão horário e de fim de semana do alvo CO(GT) no treino
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(x=hora_eda, y=train_eda[ALVO], ax=ax, color="tab:blue")
ax.set_title("F2b — Distribuição de CO(GT) por hora do dia (treino)")
ax.set_xlabel("Hora do dia (0h a 23h)")
ax.set_ylabel("CO(GT) (mg/m³)")
plt.tight_layout()
salvar("F2b_co_por_hora")

media_hora = train_eda.groupby(hora_eda)[ALVO].mean()
media_util = float(train_eda.loc[dia_sem_eda < 5, ALVO].mean())
media_fds = float(train_eda.loc[dia_sem_eda >= 5, ALVO].mean())
serie_ord = train_eda.sort_values("datetime")[ALVO].reset_index(drop=True)
ac1 = float(serie_ord.autocorr(1))
print(f"Média CO(GT) dias úteis: {media_util:.3f} mg/m³ | fim de semana: {media_fds:.3f} mg/m³")
print(f"Autocorrelação lag-1 aproximada no treino ordenado: {ac1:.3f}")

# Identificação de pares redundantes (|Spearman| > 0,8)
def grupo_de(c):
    if c.startswith("PT08"):
        return "sensor"
    if c in ("CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"):
        return "analisador"
    if c in ("T", "RH", "AH"):
        return "meteorológica"
    return "outro"

preds_eda = [c for c in cols_f2a if c != ALVO]
cm_eda = train_eda[preds_eda].corr(method="spearman")
linhas_red = []
for i, a in enumerate(preds_eda):
    for b in preds_eda[i + 1:]:
        r = cm_eda.loc[a, b]
        if abs(r) > 0.8:
            fora = "analisador" in (grupo_de(a), grupo_de(b))
            ca, cb = abs(spear[a]), abs(spear[b])
            cand = "fora do cenário" if fora else (a if ca < cb else b)
            linhas_red.append({
                "atributo_a": a,
                "atributo_b": b,
                "grupo": f"{grupo_de(a)} x {grupo_de(b)}",
                "rho_spearman": round(r, 3),
                "|rho| alvo (a)": round(ca, 3),
                "|rho| alvo (b)": round(cb, 3),
                "candidato_descarte": cand,
            })
PARES_RED = pd.DataFrame(linhas_red)
print("\\n--- Pares redundantes (|Spearman| > 0,8) ---")
print(PARES_RED.to_string(index=False) if len(PARES_RED) else "Nenhum par com |rho| > 0,8")
CANDIDATOS_DESCARTE = sorted({c for c in PARES_RED["candidato_descarte"] if c != "fora do cenário"}) if len(PARES_RED) else []
print("Candidatos a descarte no cenário principal:", CANDIDATOS_DESCARTE)
"""
    )

    code(
        """
# Cinco achados da EDA (somente treino) ligados diretamente às decisões de tratamento
ass_co = float(train_eda[ALVO].skew())
curt_co = float(train_eda[ALVO].kurt())
q1c, q3c = train_eda[ALVO].quantile(0.25), train_eda[ALVO].quantile(0.75)
cand_co = int(((train_eda[ALVO] < q1c - 3 * (q3c - q1c)) | (train_eda[ALVO] > q3c + 3 * (q3c - q1c))).sum())
hmin, hmax = int(media_hora.idxmin()), int(media_hora.idxmax())
sens_na = float(train_eda["PT08.S1(CO)"].isna().mean() * 100)
lista_cand = ", ".join(CANDIDATOS_DESCARTE) if CANDIDATOS_DESCARTE else "nenhum"

achados = [
    f"1. Assimetria e extremos do alvo: CO(GT) tem assimetria {ass_co:.2f} e curtose {curt_co:.2f}, com {cand_co} "
    f"pontos acima de Q3+3*IQR ({100*cand_co/len(train_eda):.2f}%), todos dentro de [0, 15] mg/m³. "
    f"-> Decisão: manter sem transformação logarítmica no alvo para preservar o erro na unidade original.",
    f"2. Padrão de ausências: NMHC(GT) tem {100*taxa_nmhc:.2f}% de falhas; sensores PT08 e clima têm {sens_na:.2f}% "
    f"de falhas simultâneas. -> Decisão: descartar NMHC(GT) (>70%) e imputar os demais pela mediana dentro do Pipeline.",
    f"3. Sazonalidade e sensibilidade térmica: média de CO(GT) varia de {media_hora.min():.2f} ({hmin}h) a "
    f"{media_hora.max():.2f} ({hmax}h) e cai no fim de semana ({media_fds:.2f} vs {media_util:.2f} mg/m³); "
    f"escalas variam de ~0,2 (AH) a ~2.700 (PT08.S4). -> Decisão: criar atributos cíclicos/fim de semana/interação e usar StandardScaler.",
    f"4. Autocorrelação temporal: lag-1 aproximado = {ac1:.2f}. -> Decisão: seguir o protocolo KFold(5, shuffle=True, seed=42), "
    f"documentando o otimismo e quantificando a diferença em validação temporal complementar ao final.",
    f"5. Colinearidade e vazamento de cenário: C6H6/NOx/NO2 são do mesmo analisador do alvo e sensores têm |rho|>0,8 "
    f"({lista_cand}). -> Decisão: excluir analisadores de referência e avaliar PCA e SelectKBest na Seção 8.",
]
for a in achados:
    print(a)
"""
    )

    # ------------------------------------------------------------------
    # 4. DIVISÃO DOS DADOS
    # ------------------------------------------------------------------
    md(
        """
## 4. Divisão dos dados

Materializamos apenas os dados de treino (`train_df`) a partir de `idx_tr` e definimos o esquema de validação cruzada de 5 partes (`KFold(n_splits=5, shuffle=True, random_state=42)`). O conjunto de teste (`idx_te`) permanece reservado e **não** é materializado aqui.

**Observações do protocolo:**
- Não existem grupos de pacientes/equipamentos distintos neste conjunto (`GroupShuffleSplit`/`GroupKFold` não se aplicam).
- Por se tratar de série horária com autocorrelação (~0,79), a divisão aleatória é uma limitação conhecida que será quantificada na análise complementar (somente sobre o treino).
"""
    )

    code(
        """
train_df = df_f.loc[idx_tr].copy()
train_datetime = train_df["datetime"].copy()  # guardado apenas para a análise temporal complementar no treino
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"Treino materializado: {len(train_df)} amostras (80%)")
print(f"Teste reservado (apenas índices): {len(idx_te)} amostras (20%, lido somente na Seção 9)")
"""
    )

    # ------------------------------------------------------------------
    # 5. TRATAMENTO E ENGENHARIA DE ATRIBUTOS
    # ------------------------------------------------------------------
    md(
        """
## 5. Tratamento e engenharia de atributos

1. **Vazamento e descartes estruturais:** remoção de `Date`, `Time` e `datetime` dos preditores; descarte de `NMHC(GT)` (>70% ausente no treino); exclusão de `C6H6(GT)`, `NOx(GT)` e `NO2(GT)` (outros analisadores certificados simultâneos, indisponíveis sem a estação de referência).
2. **Faltantes e valores extremos:** conversão de `-200` para `NaN` nos preditores; verificação por critério `3×IQR` e faixas físicas fixas *a priori* no treino, definindo as regras de tratamento determinísticas (`aplicar_tratamento`).
3. **Engenharia de atributos dentro do `Pipeline`:** `AtributosAirQuality(BaseEstimator, TransformerMixin)` recebe os 8 atributos físicos base (`5 sensores PT08 + T + RH + AH`) e duas colunas auxiliares temporais (`hora`, `dia_semana`), gerando **5 atributos novos** (`hora_sin`, `hora_cos`, `fim_de_semana`, `razao_s1_tempK`, `temp_x_umid`) e removendo `hora` e `dia_semana` brutas (**8 base -> 13 atributos**).
4. **Escala e imputação:** `SimpleImputer(strategy="median")` + `StandardScaler()` dentro do `Pipeline`.
5. **Efeito dos novos atributos:** medido por validação cruzada de 5 partes nos **três modelos** com parâmetros padrão (`efeito_atributos.csv`).
"""
    )

    code(
        """
# Conversão estrutural de -200 -> NaN nos preditores numéricos do treino
colunas_num_pred = [c for c in train_df.select_dtypes("number").columns if c != ALVO]
train_df[colunas_num_pred] = train_df[colunas_num_pred].replace(-200, np.nan)

# Tabela de outliers (3xIQR) e faixas físicas fixas (avaliadas exclusivamente no treino)
n_tr = len(train_df)
linhas_out, DECISAO, COLS_REMOVER = [], {}, []
for c, (lo, hi) in FAIXAS.items():
    if c not in train_df.columns or (DROP_NMHC and c == "NMHC(GT)"):
        continue
    s = train_df[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    cand = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
    neg = int((s < 0).sum()) if lo >= 0 else 0
    sup = int((s > hi).sum())
    inf = int((s < lo).sum()) if lo >= 0 else 0
    if neg > 0 and neg / n_tr < 0.001:
        dec = "remover linhas com negativo (<0,1%)"
        COLS_REMOVER.append(c)
        afet = neg
    elif neg > 0:
        dec = "manter e discutir (negativos >=0,1%)"
        afet = neg
    elif sup > 0 or inf > 0:
        dec = "limitar à faixa física"
        afet = sup + inf
    else:
        dec = "manter (3xIQR dentro da faixa física)" if cand > 0 else "manter"
        afet = cand
    DECISAO[c] = dec
    linhas_out.append([c, cand, round(100 * cand / n_tr, 2), neg, round(100 * neg / n_tr, 3), sup + inf, round(100 * afet / n_tr, 2), dec])

TAB_OUT = pd.DataFrame(
    linhas_out,
    columns=["coluna", "cand_3xIQR", "%_3xIQR", "negativos", "%_neg", "fora_faixa", "%_afetado", "decisao"],
)
print("--- Tratamento de valores extremos (somente treino) ---")
print(TAB_OUT.to_string(index=False))


def preparar_colunas_base(d):
    d = d.copy()
    dt = d["datetime"]
    d["hora"] = dt.dt.hour + dt.dt.minute / 60.0
    d["dia_semana"] = dt.dt.dayofweek
    if DROP_NMHC and "NMHC(GT)" in d.columns:
        d = d.drop(columns=["NMHC(GT)"])
    if COLS_REMOVER:
        d = d[~(d[COLS_REMOVER] < 0).any(axis=1)]
    for c, dec in DECISAO.items():
        if dec.startswith("limitar") and c in d.columns:
            lo, hi = FAIXAS[c]
            d[c] = d[c].clip(lo, hi)
    return d.drop(columns=["Date", "Time", "datetime", "C6H6(GT)", "NOx(GT)", "NO2(GT)"], errors="ignore")


train_prep = preparar_colunas_base(train_df)
COLS_BASE = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
COLS_TEMPO_AUX = ["hora", "dia_semana"]
COLS_NOVOS = ["hora_sin", "hora_cos", "fim_de_semana", "razao_s1_tempK", "temp_x_umid"]
COLS_ATR_ATIVOS = COLS_BASE + COLS_NOVOS
COLS_ENTRADA = COLS_BASE + COLS_TEMPO_AUX

X_tr = train_prep[COLS_ENTRADA].copy()
y_tr = train_prep[ALVO].copy()
train_datetime = train_datetime.loc[X_tr.index]

print(f"\\nTreino após tratamento: {X_tr.shape[0]} amostras | entradas base: {len(COLS_BASE)} (+2 auxiliares temporais)")
print(f"Atributos finais com engenharia ativada: {len(COLS_ATR_ATIVOS)} -> {COLS_ATR_ATIVOS}")
"""
    )

    code(
        """
class AtributosAirQuality(BaseEstimator, TransformerMixin):
    \"\"\"Cria 5 atributos de domínio/calendário e remove colunas temporais brutas.\"\"\"

    def __init__(self, ativar=True):
        self.ativar = ativar

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X_out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=COLS_ENTRADA)
        if self.ativar:
            X_out["hora_sin"] = np.sin(2.0 * np.pi * X_out["hora"] / 24.0)
            X_out["hora_cos"] = np.cos(2.0 * np.pi * X_out["hora"] / 24.0)
            X_out["fim_de_semana"] = (X_out["dia_semana"] >= 5).astype(float)
            X_out["razao_s1_tempK"] = X_out["PT08.S1(CO)"] / (X_out["T"] + 273.15)
            X_out["temp_x_umid"] = X_out["T"] * X_out["RH"]
        cols = list(COLS_BASE) + (list(COLS_NOVOS) if self.ativar else [])
        return X_out[cols]

    def get_feature_names_out(self, input_features=None):
        cols = list(COLS_BASE) + (list(COLS_NOVOS) if self.ativar else [])
        return np.asarray(cols, dtype=object)


def pipe(estimador, reducao=None, ativar_fe=True):
    passos = [
        ("atr", AtributosAirQuality(ativar=ativar_fe)),
        ("imp", SimpleImputer(strategy="median")),
        ("esc", StandardScaler()),
    ]
    if reducao is not None:
        passos.append(reducao)
    passos.append(("modelo", estimador))
    return Pipeline(passos)


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100.0)


METRICAS = ["rmse", "mae", "r2"] + (["mape"] if MAPE_OK else [])


def avaliar_cv(estimador, X, y, cv_obj=None):
    cv_usar = kf if cv_obj is None else cv_obj
    reg = {k: [] for k in METRICAS}
    t_fit, t_pred = [], []
    for a, b in cv_usar.split(X):
        mdl = clone(estimador)
        t0 = time.perf_counter()
        mdl.fit(X.iloc[a], y.iloc[a])
        t_fit.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        p = mdl.predict(X.iloc[b])
        t_pred.append(time.perf_counter() - t0)
        yv = y.iloc[b].values
        reg["rmse"].append(float(np.sqrt(np.mean((yv - p) ** 2))))
        reg["mae"].append(float(mean_absolute_error(yv, p)))
        reg["r2"].append(float(r2_score(yv, p)))
        if MAPE_OK:
            reg["mape"].append(mape(yv, p))
    linha = {}
    for k in METRICAS:
        linha[f"{k}_cv_media"] = float(np.mean(reg[k]))
        linha[f"{k}_cv_desvio"] = float(np.std(reg[k], ddof=1))
    linha["tempo_treino_cv_s"] = float(np.mean(t_fit))
    linha["tempo_previsao_cv_s"] = float(np.mean(t_pred))
    return linha


def estimador_base(nome, hp=None):
    hp = hp or {}
    if nome == "ElasticNet":
        return ElasticNet(random_state=SEED, max_iter=20000, **hp)
    if nome == "RandomForest":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **hp)
    if nome == "SVR":
        return SVR(**hp)
    raise ValueError(f"Modelo desconhecido: {nome}")
"""
    )

    code(
        """
# Medição do efeito dos 5 atributos novos (CV 5-folds no treino, nos 3 modelos padrão)
if RODAR_BUSCAS:
    linhas_fe = []
    for nome in MODELOS:
        for ativar, rot in [(False, "sem_atributos_novos (8)"), (True, "com_atributos_novos (13)")]:
            r = avaliar_cv(pipe(estimador_base(nome), ativar_fe=ativar), X_tr, y_tr)
            r.update({"modelo": nome, "cenario": rot, "n_atributos": 13 if ativar else 8})
            linhas_fe.append(r)
    df_efeito_fe = pd.DataFrame(linhas_fe)
    df_efeito_fe.to_csv(caminho_csv("efeito_atributos.csv"), index=False)
else:
    df_efeito_fe = ler_csv("efeito_atributos.csv")

print("--- Efeito dos 5 atributos novos (CV 5-folds no treino, parâmetros padrão) ---")
vis_fe = pd.DataFrame({
    "modelo": df_efeito_fe["modelo"],
    "cenario": df_efeito_fe["cenario"],
    "RMSE CV": df_efeito_fe.apply(lambda r: f"{r['rmse_cv_media']:.4f} ± {r['rmse_cv_desvio']:.4f}", axis=1),
    "MAE CV": df_efeito_fe.apply(lambda r: f"{r['mae_cv_media']:.4f} ± {r['mae_cv_desvio']:.4f}", axis=1),
    "R² CV": df_efeito_fe.apply(lambda r: f"{r['r2_cv_media']:.4f} ± {r['r2_cv_desvio']:.4f}", axis=1),
})
print(vis_fe.to_string(index=False))
"""
    )

    # ------------------------------------------------------------------
    # 6. BASELINE E MODELOS PADRÃO
    # ------------------------------------------------------------------
    md(
        """
## 6. Baseline e modelos padrão (antes do ajuste)

Comparação inicial de quatro configurações com parâmetros padrão do scikit-learn (`DummyRegressor(strategy="mean")`, `ElasticNet`, `RandomForestRegressor` e `SVR`), avaliadas por validação cruzada de 5 partes exclusivamente no treino. O mesmo estimador linear (`ElasticNet`) é mantido antes e depois do ajuste na Seção 7.
"""
    )

    code(
        """
if RODAR_BUSCAS:
    antes_linhas = []
    for nome in ["Baseline"] + MODELOS:
        est = DummyRegressor(strategy="mean") if nome == "Baseline" else estimador_base(nome)
        r = avaliar_cv(pipe(est), X_tr, y_tr)
        r.update({
            "modelo": nome,
            "versao": "baseline" if nome == "Baseline" else "padrao",
            "configuracao": "nenhum",
            "n_atributos": len(COLS_ATR_ATIVOS),
        })
        antes_linhas.append(r)
    antes = pd.DataFrame(antes_linhas)
else:
    _res_prev = ler_csv("resultados_configuracoes.csv")
    antes = _res_prev[_res_prev["versao"].isin(["baseline", "padrao"])].copy()

print("--- Baseline e modelos padrão (CV 5-folds no treino) ---")
print(pd.DataFrame(
    {k.upper(): antes.apply(lambda r, k=k: f"{r[k + '_cv_media']:.3f} ± {r[k + '_cv_desvio']:.3f}", axis=1).values for k in METRICAS},
    index=antes["modelo"].values,
).to_string())
"""
    )

    # ------------------------------------------------------------------
    # 7. AJUSTE DE HIPERPARÂMETROS
    # ------------------------------------------------------------------
    md(
        """
## 7. Ajuste de hiperparâmetros

Busca de hiperparâmetros sobre o `Pipeline` completo por meio de função única `buscar(...)`, otimizando `neg_root_mean_squared_error` em validação cruzada de 5 partes no treino:
- **ElasticNet (`GridSearchCV`):** grade `alpha` em `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10]` × `l1_ratio` em `[0.1, 0.5, 0.9, 1.0]` (24 combinações). Caso `alpha` atinja a borda da grade, há 1 rodada de extensão automática (`×10` ou `/10`). `l1_ratio = 1.0` é teto matemático de domínio (Lasso) e é apenas registrado.
- **Random Forest (`RandomizedSearchCV`, `n_iter=20`, `random_state=42`):** `n_estimators ~ randint(100, 401)`, `max_features ∈ {'sqrt', 0.3, 0.5, 0.7, 1.0}`, `min_samples_leaf ~ randint(1, 21)`.
- **SVR (`RandomizedSearchCV`, `n_iter=30`, `random_state=42`):** `C ~ loguniform(0.1, 1000)`, `gamma ~ loguniform(1e-3, 1)`, `epsilon ~ loguniform(0.01, 1)`, `kernel ∈ {'rbf', 'linear'}`. Justificativa: uma grade cartesiana densa no SVR custa dezenas de minutos (como observado na análise anterior), enquanto a busca aleatória em escala logarítmica cobre faixas contínuas amplas com orçamento computacional controlado.

**Regra de bordas:** para buscas aleatórias, um hiperparâmetro contínuo/inteiro é marcado como "na borda" quando o melhor valor cai nos 10% extremos da faixa (em escala logarítmica para `C`, `gamma`, `epsilon`, e escala linear para `n_estimators` e `min_samples_leaf`, respeitando limites naturais de domínio como `min_samples_leaf >= 1` e `max_features <= 1.0`). Se RF ou SVR caírem na borda na rodada 1, executa-se **uma rodada extra** (`n_iter=15`) com a faixa deslocada (`×10` ou `/10` no lado da borda), mantendo o melhor resultado entre as duas rodadas.
"""
    )

    if smoke:
        espacos_str = """
ESPACOS = {
    "ElasticNet": {
        "tipo": "grid",
        "n_iter": None,
        "params": {
            "modelo__alpha": [0.001, 0.01, 0.1],
            "modelo__l1_ratio": [0.5, 1.0],
        },
        "desc": "alpha in [0.001, 0.01, 0.1] | l1_ratio in [0.5, 1.0]",
    },
    "RandomForest": {
        "tipo": "random",
        "n_iter": 2,
        "params": {
            "modelo__n_estimators": randint(50, 101),
            "modelo__max_features": ["sqrt", 0.5],
            "modelo__min_samples_leaf": randint(1, 11),
        },
        "desc": "n_estimators~randint(50,101) | max_features in ['sqrt',0.5] | min_samples_leaf~randint(1,11)",
    },
    "SVR": {
        "tipo": "random",
        "n_iter": 2,
        "params": {
            "modelo__kernel": ["rbf", "linear"],
            "modelo__C": loguniform(0.1, 100),
            "modelo__gamma": loguniform(1e-3, 1),
            "modelo__epsilon": loguniform(0.01, 0.5),
        },
        "desc": "kernel in ['rbf','linear'] | C~loguniform(0.1,100) | gamma~loguniform(1e-3,1) | epsilon~loguniform(0.01,0.5)",
    },
}
N_ITER_R2 = 2
"""
    else:
        espacos_str = """
ESPACOS = {
    "ElasticNet": {
        "tipo": "grid",
        "n_iter": None,
        "params": {
            "modelo__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            "modelo__l1_ratio": [0.1, 0.5, 0.9, 1.0],
        },
        "desc": "alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10] | l1_ratio in [0.1, 0.5, 0.9, 1.0]",
    },
    "RandomForest": {
        "tipo": "random",
        "n_iter": 20,
        "params": {
            "modelo__n_estimators": randint(100, 401),
            "modelo__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
            "modelo__min_samples_leaf": randint(1, 21),
        },
        "desc": "n_estimators~randint(100,401) | max_features in ['sqrt',0.3,0.5,0.7,1.0] | min_samples_leaf~randint(1,21)",
    },
    "SVR": {
        "tipo": "random",
        "n_iter": 30,
        "params": {
            "modelo__kernel": ["rbf", "linear"],
            "modelo__C": loguniform(0.1, 1000.0),
            "modelo__gamma": loguniform(1e-3, 1.0),
            "modelo__epsilon": loguniform(0.01, 1.0),
        },
        "desc": "kernel in ['rbf','linear'] | C~loguniform(0.1,1000) | gamma~loguniform(1e-3,1) | epsilon~loguniform(0.01,1)",
    },
}
N_ITER_R2 = 15
"""

    code(
        espacos_str
        + """
# Curva de variância do PCA no treino (só para determinar K_PCA_95 e K_PCA a priori)
Xs_tr = pipe(DummyRegressor())[:-1].fit_transform(X_tr, y_tr)
pca_full = PCA(random_state=SEED).fit(Xs_tr)
var_exp = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_exp)
K_PCA_95 = int(np.argmax(var_acum >= 0.95) + 1)
K_PCA = int(np.argmax(var_acum >= 0.90) + 1)
if K_PCA == K_PCA_95:
    K_PCA = int(np.argmax(var_acum >= 0.85) + 1)
print(f"PCA treino (13 attrs): 95% -> {K_PCA_95} comp. ({100*var_acum[K_PCA_95-1]:.2f}%) | k menor -> {K_PCA} comp. ({100*var_acum[K_PCA-1]:.2f}%)")


def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def red_step(cfg):
    if cfg == "nenhum":
        return None
    if cfg == "pca95":
        return ("pca", PCA(n_components=0.95, random_state=SEED))
    if cfg == "pca_k":
        return ("pca", PCA(n_components=K_PCA, random_state=SEED))
    return ("selk", SelectKBest(f_regression, k=int(cfg.split("_")[1])))


def buscar(nome, espaco, tipo="grid", config_red="nenhum", hp_fixos=None, n_iter=20):
    est = estimador_base(nome, hp_fixos or {})
    if nome == "RandomForest":
        est.set_params(n_jobs=1)  # evita oversubscrição quando a busca usa n_jobs=-1
    base = pipe(est, red_step(config_red))
    if tipo == "grid":
        s = GridSearchCV(
            base, espaco, cv=kf, scoring="neg_root_mean_squared_error",
            return_train_score=True, n_jobs=-1, refit=True
        )
    elif tipo == "random":
        s = RandomizedSearchCV(
            base, espaco, n_iter=n_iter, cv=kf, scoring="neg_root_mean_squared_error",
            return_train_score=True, n_jobs=-1, refit=True, random_state=SEED
        )
    else:
        raise ValueError("tipo deve ser 'grid' ou 'random'")
    t0 = time.perf_counter()
    s.fit(X_tr, y_tr)
    s.tempo_busca_s = float(time.perf_counter() - t0)
    return s


# Regras de borda e extensão (ElasticNet em grade; RF e SVR por 10% extremos em distribuição)
BOUNDS_RAND = {
    "RandomForest": {
        "modelo__n_estimators": ("int", 100, 400),
        "modelo__min_samples_leaf": ("int_floor1", 1, 20),
    },
    "SVR": {
        "modelo__C": ("log", 0.1, 1000.0),
        "modelo__gamma": ("log", 1e-3, 1.0),
        "modelo__epsilon": ("log", 0.01, 1.0),
    },
}


def detectar_bordas_grid(gs, grade):
    bordas = {}
    for k, vals in grade.items():
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        b = gs.best_params_[k]
        if len(nums) >= 2 and isinstance(b, (int, float)) and not isinstance(b, bool):
            if b == min(nums):
                bordas[k] = "inf"
            elif b == max(nums):
                if k == "modelo__l1_ratio" and b == 1.0:
                    continue  # 1.0 é limite matemático de domínio (Lasso)
                bordas[k] = "sup"
    return bordas


def estender_grade_enet(grade, bordas, melhores):
    nova = {k: list(v) for k, v in grade.items()}
    if bordas.get("modelo__alpha") == "inf":
        b = melhores["modelo__alpha"]
        nova["modelo__alpha"] += [b / 10.0, b / 100.0]
    elif bordas.get("modelo__alpha") == "sup":
        b = melhores["modelo__alpha"]
        nova["modelo__alpha"] += [b * 10.0, b * 100.0]
    if bordas.get("modelo__l1_ratio") == "inf":
        nova["modelo__l1_ratio"] += [0.01, 0.05]
    nova["modelo__alpha"] = sorted(set(nova["modelo__alpha"]))
    nova["modelo__l1_ratio"] = sorted(set(nova["modelo__l1_ratio"]))
    return nova


def detectar_bordas_rand(nome, gs, bounds=None):
    bordas = {}
    b_dict = (bounds if bounds is not None else BOUNDS_RAND).get(nome, {})
    for k, (escala, lo, hi) in b_dict.items():
        val = float(gs.best_params_[k])
        if escala == "log":
            l_lo, l_hi, l_v = np.log10(lo), np.log10(hi), np.log10(val)
            tol = 0.10 * (l_hi - l_lo)
            if l_v <= l_lo + tol:
                bordas[k] = "inf"
            elif l_v >= l_hi - tol:
                bordas[k] = "sup"
        elif escala == "int":
            tol = 0.10 * (hi - lo)
            if val <= lo + tol:
                bordas[k] = "inf"
            elif val >= hi - tol:
                bordas[k] = "sup"
        elif escala == "int_floor1":
            tol = 0.10 * (hi - lo)
            # min_samples_leaf=1 é limite físico mínimo; só estende no lado superior
            if val >= hi - tol:
                bordas[k] = "sup"
    return bordas


def deslocar_espaco_rand(nome, espaco_base, bordas):
    novo = dict(espaco_base)
    novos_bounds = dict(BOUNDS_RAND.get(nome, {}))
    desc_extra = []
    for k, lado in bordas.items():
        escala, lo, hi = BOUNDS_RAND[nome][k]
        if escala == "log":
            if lado == "sup":
                n_lo, n_hi = lo * 10.0, hi * 10.0
            else:
                n_lo, n_hi = lo / 10.0, hi / 10.0
            # Limitar epsilon <= 2.0 e C <= 10000 para estabilidade numérica do SVR
            if k == "modelo__epsilon":
                n_hi = min(n_hi, 2.0)
            if k == "modelo__C":
                n_hi = min(n_hi, 10000.0)
            novo[k] = loguniform(n_lo, n_hi)
            novos_bounds[k] = (escala, n_lo, n_hi)
            desc_extra.append(f"{k.replace('modelo__','')}~loguniform({n_lo:.4g},{n_hi:.4g})")
        elif escala == "int":
            if lado == "sup":
                n_lo, n_hi = max(100, int(hi * 0.8)), int(hi * 2.5)
            else:
                n_lo, n_hi = max(10, int(lo // 10)), int(hi // 2)
            novo[k] = randint(n_lo, n_hi + 1)
            novos_bounds[k] = (escala, n_lo, n_hi)
            desc_extra.append(f"{k.replace('modelo__','')}~randint({n_lo},{n_hi+1})")
        elif escala == "int_floor1" and lado == "sup":
            n_lo, n_hi = hi, hi * 3
            novo[k] = randint(n_lo, n_hi + 1)
            novos_bounds[k] = (escala, n_lo, n_hi)
            desc_extra.append(f"{k.replace('modelo__','')}~randint({n_lo},{n_hi+1})")
    return novo, desc_extra, {nome: novos_bounds}


def registrar_grade(nome, gs, rodada, tipo_reg="estimador"):
    cvr = pd.DataFrame(gs.cv_results_)
    out = pd.DataFrame(index=range(len(cvr)))
    out["modelo"] = nome
    out["tipo"] = tipo_reg
    out["rodada"] = rodada
    for c in [c for c in cvr.columns if c.startswith("param_")]:
        out[c.replace("param_modelo__", "hp_").replace("param_selk__", "hp_")] = cvr[c].apply(to_py).astype(str).values
    out["rmse_treino_medio"] = -cvr["mean_train_score"].values
    out["rmse_cv_media"] = -cvr["mean_test_score"].values
    out["rmse_cv_desvio"] = cvr["std_test_score"].values
    return out
"""
    )

    code(
        """
HP = {}
ajus_cv = {}
if RODAR_BUSCAS:
    hiper_linhas, grade_partes = [], []
    for nome, cfg_e in ESPACOS.items():
        tipo_b = cfg_e["tipo"]
        esp1 = cfg_e["params"]
        gs1 = buscar(nome, esp1, tipo_b, "nenhum", n_iter=cfg_e["n_iter"] or 20)
        grade_partes.append(registrar_grade(nome, gs1, 1))

        bordas1 = detectar_bordas_grid(gs1, esp1) if tipo_b == "grid" else detectar_bordas_rand(nome, gs1)
        gs_fin = gs1
        rodada_fin = 1
        tempo_tot = gs1.tempo_busca_s
        n_comb = len(gs1.cv_results_["params"])
        desc_fin = cfg_e["desc"]
        bordas_fin = bordas1

        # Rodada 2 caso algum parâmetro elegível tenha ficado na borda
        if bordas1:
            if tipo_b == "grid":
                esp2 = estender_grade_enet(esp1, bordas1, gs1.best_params_)
                if esp2 != esp1:
                    gs2 = buscar(nome, esp2, "grid", "nenhum")
                    grade_partes.append(registrar_grade(nome, gs2, 2))
                    tempo_tot += gs2.tempo_busca_s
                    n_comb += len(gs2.cv_results_["params"])
                    rodada_fin = 2
                    desc_fin = f"alpha in {esp2['modelo__alpha']} | l1_ratio in {esp2['modelo__l1_ratio']}"
                    if gs2.best_score_ >= gs1.best_score_:
                        gs_fin = gs2
                        bordas_fin = detectar_bordas_grid(gs2, esp2)
            else:
                esp2, d2, bounds2 = deslocar_espaco_rand(nome, esp1, bordas1)
                gs2 = buscar(nome, esp2, "random", "nenhum", n_iter=N_ITER_R2)
                grade_partes.append(registrar_grade(nome, gs2, 2))
                tempo_tot += gs2.tempo_busca_s
                n_comb += len(gs2.cv_results_["params"])
                rodada_fin = 2
                desc_fin = cfg_e["desc"] + " + R2(" + "; ".join(d2) + ")"
                if gs2.best_score_ > gs1.best_score_:
                    gs_fin = gs2
                    bordas_fin = detectar_bordas_rand(nome, gs2, bounds=bounds2)

        hp = to_py({k.replace("modelo__", ""): v for k, v in gs_fin.best_params_.items()})
        # Arredondar floats contínuos apenas na impressão; manter precisão completa em HP
        HP[nome] = hp
        ajus_cv[nome] = avaliar_cv(pipe(estimador_base(nome, hp)), X_tr, y_tr)
        hiper_linhas.append({
            "modelo": nome,
            "tipo_busca": f"{tipo_b} ({'exaustiva' if tipo_b=='grid' else 'aleatoria'})",
            "rodada_final": rodada_fin,
            "espaco_busca": desc_fin,
            "n_combinacoes": int(n_comb),
            "tempo_busca_s": round(tempo_tot, 2),
            "hiperparametros": str(hp),
            "parametros_na_borda_r1": str(list(bordas1.keys())),
            "parametros_na_borda_final": str(list(bordas_fin.keys())),
            "selectk_k_escolhido": "",
            "atributos_descartados_selectk": "",
            "candidatos_redundancia_descartados": "",
        })
        print(
            f"{nome}: rodada {rodada_fin} | melhor={hp} | RMSE CV={ajus_cv[nome]['rmse_cv_media']:.4f} "
            f"± {ajus_cv[nome]['rmse_cv_desvio']:.4f} | comb={n_comb} | t={tempo_tot:.1f}s | borda_r1={list(bordas1.keys())}"
        )

    hiper = pd.DataFrame(hiper_linhas)
    grade_cv = pd.concat(grade_partes, ignore_index=True)

    # Identifica o hiperparâmetro de maior amplitude no CV e calcula validation_curve para F5
    amps = []
    for nome in MODELOS:
        g = grade_cv[(grade_cv.modelo == nome) & (grade_cv.tipo == "estimador")]
        for c in [col for col in g.columns if col.startswith("hp_")]:
            s_num = pd.to_numeric(g[c], errors="coerce")
            if s_num.notna().all() and s_num.nunique() >= 2:
                # Para buscas contínuas, agrupa em quantis ou usa valores exatos quando discretos
                if g[c].nunique() <= 12:
                    mm = g.groupby(c)["rmse_cv_media"].mean()
                else:
                    bins = pd.qcut(s_num, q=min(6, s_num.nunique()), duplicates="drop")
                    mm = g.groupby(bins, observed=True)["rmse_cv_media"].mean()
                amps.append((nome, c, float(mm.max() - mm.min())))
    amp_df = pd.DataFrame(amps, columns=["modelo", "hiperparametro", "amplitude"]).sort_values("amplitude", ascending=False)
    print("\\n--- Amplitude do RMSE CV por hiperparâmetro ---")
    print(amp_df.to_string(index=False))

    mod_f5 = amp_df.iloc[0]["modelo"]
    hp_f5 = amp_df.iloc[0]["hiperparametro"].replace("hp_", "")
    faixas_vc = {
        ("ElasticNet", "alpha"): np.logspace(-6, 2, 9),
        ("ElasticNet", "l1_ratio"): np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        ("SVR", "C"): np.array([0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]),
        ("SVR", "gamma"): np.logspace(-4, 0, 5),
        ("SVR", "epsilon"): np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0]),
        ("RandomForest", "n_estimators"): np.array([25, 50, 100, 200, 300, 400]),
        ("RandomForest", "min_samples_leaf"): np.array([1, 2, 5, 10, 15, 20]),
    }
    p_range = faixas_vc[(mod_f5, hp_f5)]
    hp_fix_vc = {k: v for k, v in HP[mod_f5].items() if k != hp_f5}
    tr_sc, val_sc = validation_curve(
        pipe(estimador_base(mod_f5, hp_fix_vc)),
        X_tr, y_tr,
        param_name=f"modelo__{hp_f5}",
        param_range=p_range,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    df_vc = pd.DataFrame({
        "modelo": mod_f5,
        "hiperparametro": hp_f5,
        "valor": [float(x) for x in p_range],
        "rmse_treino_media": (-tr_sc).mean(axis=1),
        "rmse_treino_desvio": (-tr_sc).std(axis=1, ddof=1),
        "rmse_val_media": (-val_sc).mean(axis=1),
        "rmse_val_desvio": (-val_sc).std(axis=1, ddof=1),
    })
    df_vc.to_csv(caminho_csv("curva_validacao.csv"), index=False)
else:
    hiper = ler_csv("melhores_hiperparametros.csv")
    grade_cv = ler_csv("cv_grade_completa.csv")
    df_vc = ler_csv("curva_validacao.csv")
    HP = {r.modelo: ast.literal_eval(r.hiperparametros) for r in hiper.itertuples()}

print("\\n--- O ajuste valeu o tempo? (CV 5-folds no treino) ---")
for nome in MODELOS:
    pad_m = float(antes.loc[antes.modelo == nome, "rmse_cv_media"].iloc[0])
    pad_s = float(antes.loc[antes.modelo == nome, "rmse_cv_desvio"].iloc[0])
    if RODAR_BUSCAS:
        aj_m, aj_s = ajus_cv[nome]["rmse_cv_media"], ajus_cv[nome]["rmse_cv_desvio"]
    else:
        _r = ler_csv("resultados_configuracoes.csv")
        _row = _r[(_r.modelo == nome) & (_r.versao == "ajustado")].iloc[0]
        aj_m, aj_s = float(_row["rmse_cv_media"]), float(_row["rmse_cv_desvio"])
    t_b = float(hiper.loc[hiper.modelo == nome, "tempo_busca_s"].iloc[0])
    print(f"{nome:<13}: {pad_m:.4f} ± {pad_s:.4f} -> {aj_m:.4f} ± {aj_s:.4f} (Δ = {pad_m - aj_m:+.4f}) | tempo busca = {t_b:.1f}s")
"""
    )

    # ------------------------------------------------------------------
    # 8. REDUÇÃO DE DIMENSIONALIDADE
    # ------------------------------------------------------------------
    md(
        """
## 8. Redução de dimensionalidade (PCA e SelectKBest)

Com os hiperparâmetros ajustados da Seção 7 fixos, avaliamos cada modelo em quatro configurações por validação cruzada de 5 partes no treino:
1. **Sem redução** (`nenhum`, 13 atributos);
2. **`PCA(n_components=0.95)`** (`pca95`, retendo >= 95% da variância);
3. **`PCA(n_components=K_PCA)`** (`pca_k`, configuração menor escolhida pela curva de variância explicada, 90% ou 85%);
4. **Segunda técnica (`SelectKBest(f_regression)`)**, com `k ∈ {4, 6, 8, 10}` escolhido via `GridSearchCV` sobre o `Pipeline` completo.

**Regra para as linhas 8–10 da Tabela T1:** como o PCA é a técnica obrigatória de redução no enunciado (e o `SelectKBest` é a segunda técnica complementar), para cada modelo a configuração entregue nas linhas 8–10 da T1 é a de **menor RMSE de CV entre as duas configurações de PCA** (`pca95` ou `pca_k`). O `SelectKBest` é avaliado em CV na tabela complementar e na grade do `SelectKBest`.
"""
    )

    code(
        """
KS_SELECT = [4, 6, 8, 10]


def pipeline_final(nome, cfg):
    return pipe(estimador_base(nome, HP[nome]), red_step(cfg))


def n_atributos_cfg(cfg):
    if cfg == "nenhum":
        return len(COLS_ATR_ATIVOS)
    if cfg == "pca95":
        return K_PCA_95
    if cfg == "pca_k":
        return K_PCA
    return int(cfg.split("_")[1])


if RODAR_BUSCAS:
    filas_red, grade_red_partes, cfg_k, DESCARTADOS = [], [], {}, {}
    for nome in MODELOS:
        # 1. Sem redução (reutiliza ajus_cv) e as 2 configurações de PCA
        r_sem = dict(ajus_cv[nome])
        r_sem.update({"modelo": nome, "configuracao": "nenhum", "n_atributos": len(COLS_ATR_ATIVOS)})
        filas_red.append(r_sem)

        for cfg in ["pca95", "pca_k"]:
            r = avaliar_cv(pipeline_final(nome, cfg), X_tr, y_tr)
            r.update({"modelo": nome, "configuracao": cfg, "n_atributos": n_atributos_cfg(cfg)})
            filas_red.append(r)

        # 2. SelectKBest: k em {4, 6, 8, 10} via GridSearchCV sobre o Pipeline
        gs_k = buscar(nome, {"selk__k": KS_SELECT}, "grid", "selectk_8", hp_fixos=HP[nome])
        grade_red_partes.append(registrar_grade(nome, gs_k, 1, tipo_reg="reducao"))
        k_best = int(gs_k.best_params_["selk__k"])
        cfg_k[nome] = k_best

        r_sk = avaliar_cv(pipeline_final(nome, f"selectk_{k_best}"), X_tr, y_tr)
        r_sk.update({"modelo": nome, "configuracao": f"selectk_{k_best}", "n_atributos": k_best})
        filas_red.append(r_sk)

        # Registro dos atributos descartados pelo SelectKBest (ajuste no treino inteiro apenas para relatar)
        prep_sk = pipe(DummyRegressor())[:-1]
        X_sk = prep_sk.fit_transform(X_tr, y_tr)
        sel = SelectKBest(f_regression, k=k_best).fit(X_sk, y_tr)
        DESCARTADOS[nome] = [c for c, keep in zip(COLS_ATR_ATIVOS, sel.get_support()) if not keep]
        print(f"{nome}: SelectKBest escolhe k={k_best} e descarta {DESCARTADOS[nome]}")

    cv_red = pd.DataFrame(filas_red)
    cv_red.to_csv(caminho_csv("cv_reducao.csv"), index=False)

    grade_cv = pd.concat([grade_cv, pd.concat(grade_red_partes, ignore_index=True)], ignore_index=True)
    hiper["selectk_k_escolhido"] = hiper["modelo"].map(cfg_k)
    hiper["atributos_descartados_selectk"] = hiper["modelo"].map(lambda m: str(DESCARTADOS[m]))
    hiper["candidatos_redundancia_descartados"] = hiper["modelo"].map(
        lambda m: str([c for c in DESCARTADOS[m] if c in CANDIDATOS_DESCARTE])
    )
    grade_cv.to_csv(caminho_csv("cv_grade_completa.csv"), index=False)
    hiper.to_csv(caminho_csv("melhores_hiperparametros.csv"), index=False)
else:
    cv_red = ler_csv("cv_reducao.csv")
    DESCARTADOS = {r.modelo: ast.literal_eval(r.atributos_descartados_selectk) for r in hiper.itertuples()}

# Escolha da melhor configuração de PCA por modelo (para as linhas 8-10 da T1; SelectKBest fica na tabela complementar)
MELHOR_REDUCAO = {}
for nome in MODELOS:
    sub_pca = cv_red[(cv_red.modelo == nome) & (cv_red.configuracao.isin(["pca95", "pca_k"]))]
    MELHOR_REDUCAO[nome] = str(sub_pca.loc[sub_pca["rmse_cv_media"].idxmin(), "configuracao"])

print("\\n--- Comparação CV (5-folds no treino): com e sem redução ---")
print(
    cv_red[["modelo", "configuracao", "n_atributos", "rmse_cv_media", "rmse_cv_desvio", "r2_cv_media", "tempo_treino_cv_s"]]
    .round(4)
    .to_string(index=False)
)
print("\\nMelhor redução PCA escolhida por CV (para linhas 8-10 da T1):", MELHOR_REDUCAO)
print("Comparação com candidatos redundantes da Seção 3:")
for nome in MODELOS:
    d_sk = DESCARTADOS.get(nome, [])
    print(f"  {nome}: descartados SelectKBest = {d_sk} | coincidentes com |rho|>0,8 = {[c for c in d_sk if c in CANDIDATOS_DESCARTE]}")
"""
    )

    # ------------------------------------------------------------------
    # 9. AVALIAÇÃO NO TESTE
    # ------------------------------------------------------------------
    md(
        """
## 9. Avaliação final no teste (leitura única) e seleção do modelo

- Quando `RODAR_BUSCAS = True`, o conjunto de teste (`idx_te`) é lido **uma única vez** e avaliado **exclusivamente** nas 10 configurações obrigatórias da Tabela T1 (1 baseline, 3 padrão, 3 ajustadas sem redução e 3 ajustadas com a melhor redução por CV).
- **Seleção do modelo a entregar:** conduzida **unicamente por métricas de validação cruzada** (colunas `*_teste` são explicitamente removidas e protegidas por `assert`). Entre as 6 configurações ajustadas (linhas 5–10), escolhe-se a de menor RMSE de CV; caso a diferença para a segunda colocada seja menor que o desvio padrão da melhor (empate técnico), desempata-se por **menor número de atributos**, depois por **maior interpretabilidade** (`ElasticNet > RandomForest > SVR`) e, por fim, por **menor tempo de treino em CV**.
- Quando `RODAR_BUSCAS = False`, o teste **não** é lido: os resultados são carregados de `resultados_configuracoes.csv` e `predicoes_teste.csv`, e apenas o modelo campeão é treinado no conjunto de treino para comprovar reprodutibilidade rápida.
"""
    )

    code(
        """
COLS_RES = [
    "linha_T1", "modelo", "versao", "configuracao", "hiperparametros", "n_atributos",
    "rmse_cv_media", "rmse_cv_desvio", "mae_cv_media", "mae_cv_desvio",
    "r2_cv_media", "r2_cv_desvio", "mape_cv_media", "mape_cv_desvio",
    "tempo_treino_cv_s", "tempo_previsao_cv_s",
    "rmse_teste", "mae_teste", "r2_teste", "mape_teste", "tempo_treino", "tempo_previsao",
]


def avaliar_teste_unico(p, X, y, Xt, yt):
    t0 = time.perf_counter()
    mdl = clone(p).fit(X, y)
    t_tr = float(time.perf_counter() - t0)
    t0 = time.perf_counter()
    pr = mdl.predict(Xt)
    t_pr = float(time.perf_counter() - t0)
    ytv = yt.values
    n_efetivo = (
        int(mdl.named_steps["pca"].n_components_)
        if "pca" in mdl.named_steps
        else (int(mdl.named_steps["selk"].k) if "selk" in mdl.named_steps else len(COLS_ATR_ATIVOS))
    )
    met = {
        "rmse_teste": float(np.sqrt(np.mean((ytv - pr) ** 2))),
        "mae_teste": float(mean_absolute_error(ytv, pr)),
        "r2_teste": float(r2_score(ytv, pr)),
        "mape_teste": mape(ytv, pr) if MAPE_OK else np.nan,
        "tempo_treino": t_tr,
        "tempo_previsao": t_pr,
        "n_atributos": n_efetivo,
    }
    return met, pr


def metricas_cv_dict(d):
    keys = [k for k in COLS_RES if k.endswith(("_cv_media", "_cv_desvio")) or (k.startswith("tempo_") and k.endswith("_cv_s"))]
    return {k: d[k] for k in keys if k in d}


if RODAR_BUSCAS:
    # ===== ÚNICA leitura do conjunto de teste =====
    test_df = df_f.loc[idx_te].copy()
    test_df[colunas_num_pred] = test_df[colunas_num_pred].replace(-200, np.nan)
    test_prep = preparar_colunas_base(test_df)
    X_te = test_prep[COLS_ENTRADA].copy()
    y_te = test_prep[ALVO].copy()
    print(f"Teste preparado (leitura única): {len(X_te)} amostras")

    filas_t1, preds_lista = [], []

    def registrar_linha(linha_t1, nome, versao, cfg, est_pipe, cv_dict, hp_txt):
        met, pr = avaliar_teste_unico(est_pipe, X_tr, y_tr, X_te, y_te)
        filas_t1.append({
            "linha_T1": linha_t1,
            "modelo": nome,
            "versao": versao,
            "configuracao": cfg,
            "hiperparametros": hp_txt,
            **cv_dict,
            **met,
        })
        preds_lista.append(pd.DataFrame({
            "linha_T1": linha_t1,
            "modelo": nome,
            "versao": versao,
            "configuracao": cfg,
            "valor_real": y_te.values,
            "valor_previsto": pr,
        }))

    # Linha 1 (Baseline) e Linhas 2-4 (Padrão)
    for nome in ["Baseline"] + MODELOS:
        row_a = antes[antes.modelo == nome].iloc[0].to_dict()
        if nome == "Baseline":
            p_ = pipe(DummyRegressor(strategy="mean"))
            registrar_linha("1", "Baseline", "baseline", "nenhum", p_, metricas_cv_dict(row_a), "strategy='mean'")
        else:
            p_ = pipe(estimador_base(nome))
            registrar_linha("2-4", nome, "padrao", "nenhum", p_, metricas_cv_dict(row_a), "padrao")

    # Linhas 5-7 (Ajustado sem redução)
    for nome in MODELOS:
        row_s = cv_red[(cv_red.modelo == nome) & (cv_red.configuracao == "nenhum")].iloc[0].to_dict()
        registrar_linha("5-7", nome, "ajustado", "nenhum", pipeline_final(nome, "nenhum"), metricas_cv_dict(row_s), str(HP[nome]))

    # Linhas 8-10 (Ajustado com a melhor redução por CV)
    for nome in MODELOS:
        cfg_r = MELHOR_REDUCAO[nome]
        row_r = cv_red[(cv_red.modelo == nome) & (cv_red.configuracao == cfg_r)].iloc[0].to_dict()
        vers_r = "ajustado_selectk" if cfg_r.startswith("selectk") else "ajustado_pca"
        registrar_linha("8-10", nome, vers_r, cfg_r, pipeline_final(nome, cfg_r), metricas_cv_dict(row_r), str(HP[nome]))

    resultados = pd.DataFrame(filas_t1).reindex(columns=COLS_RES)
    assert len(resultados) == 10, f"T1 deve ter exatamente 10 linhas, obteve {len(resultados)}"
    predicoes = pd.concat(preds_lista, ignore_index=True)

    resultados.to_csv(caminho_csv("resultados_configuracoes.csv"), index=False)
    predicoes.to_csv(caminho_csv("predicoes_teste.csv"), index=False)
    print("CSVs gravados em", PASTA_RESULTADOS)
else:
    resultados = ler_csv("resultados_configuracoes.csv")
    predicoes = ler_csv("predicoes_teste.csv")

# ---------- Seleção do modelo a entregar (SOMENTE por CV, nunca pelo teste) ----------
RANK_INTERP = {"ElasticNet": 1, "RandomForest": 2, "SVR": 3}
cols_sem_teste = [c for c in resultados.columns if not c.endswith("_teste")]
candidatas_cv = resultados.loc[resultados["linha_T1"].isin(["5-7", "8-10"]), cols_sem_teste].copy()
assert not any(c.endswith("_teste") for c in candidatas_cv.columns), "Vazamento: coluna de teste na seleção!"

candidatas_cv = candidatas_cv.sort_values("rmse_cv_media").reset_index(drop=True)
best, second = candidatas_cv.iloc[0], candidatas_cv.iloc[1]
dif = float(second["rmse_cv_media"] - best["rmse_cv_media"])

if dif < float(best["rmse_cv_desvio"]):
    par = candidatas_cv.iloc[:2].copy()
    par["_interp"] = par["modelo"].map(RANK_INTERP)
    par = par.sort_values(["n_atributos", "_interp", "tempo_treino_cv_s"]).reset_index(drop=True)
    escolhido = par.iloc[0]
    outro = par.iloc[1]
    if escolhido["n_atributos"] != outro["n_atributos"]:
        criterio_des = "menor número de atributos"
    elif escolhido["_interp"] != outro["_interp"]:
        criterio_des = "maior interpretabilidade"
    else:
        criterio_des = "menor tempo de treino CV"
    motivo_escolha = (
        f"empate técnico na CV (Δ={dif:.4f} < dp={best['rmse_cv_desvio']:.4f}); "
        f"desempatado por {criterio_des}"
    )
else:
    escolhido = best
    motivo_escolha = f"menor RMSE de CV (Δ={dif:.4f} >= dp={best['rmse_cv_desvio']:.4f})"

CAMPEAO = str(escolhido["modelo"])
CFG_CAMPEAO = str(escolhido["configuracao"])
VERSAO_CAMPEAO = str(escolhido["versao"])

print("\\n--- Seleção do modelo a entregar (baseada exclusivamente na CV) ---")
print(
    candidatas_cv[["linha_T1", "modelo", "configuracao", "n_atributos", "rmse_cv_media", "rmse_cv_desvio", "tempo_treino_cv_s"]]
    .round(4)
    .to_string(index=False)
)
print(f"\\n>>> Modelo escolhido: {CAMPEAO} ({VERSAO_CAMPEAO} | {CFG_CAMPEAO}) — {motivo_escolha}")

if not RODAR_BUSCAS:
    t0 = time.perf_counter()
    _mdl_chk = clone(pipeline_final(CAMPEAO, CFG_CAMPEAO)).fit(X_tr, y_tr)
    print(f"[Modo False] Modelo escolhido ajustado somente no treino em {time.perf_counter() - t0:.2f}s")
"""
    )

    # ------------------------------------------------------------------
    # 10. FIGURAS E TABELAS
    # ------------------------------------------------------------------
    md(
        """
## 10. Figuras e tabelas

Geração de todas as figuras obrigatórias (`F3` a `F9`, além de `F1`, `F2a` e `F2b` já geradas na Seção 3 apenas no treino) e das tabelas `T1`, `T2`, tabela complementar de reduções e grade do `SelectKBest`, utilizando os CSVs salvos e respeitando fonte mínima de 16 pt e cores fixas por modelo.
"""
    )

    code(
        """
# F3: Diagrama do pipeline (blocos ajustados no treino destacados, fonte >= 16 pt)
etapas = [
    ("Dado bruto\\n9.357 × 15", False),
    ("Filtro\\nCO = -200", False),
    ("Divisão 80/20\\n(seed 42)", False),
    ("Eng. Atributos\\n(+5 novas)", True),
    ("Imputação\\n(mediana)", True),
    ("Escala\\n(Standard)", True),
    ("Redução\\n(PCA/SelectK)", True),
    ("Modelo\\nEN / RF / SVR", True),
]
fig, ax = plt.subplots(figsize=(26, 6))
ax.set_xlim(0, len(etapas))
ax.set_ylim(0, 1.25)
ax.axis("off")
for i, (txt, ajust) in enumerate(etapas):
    ax.add_patch(
        FancyBboxPatch(
            (i + 0.06, 0.32), 0.88, 0.58, boxstyle="round,pad=0.02",
            fc="tab:green" if ajust else "lightgray", ec="black", alpha=0.35, lw=2,
        )
    )
    ax.text(i + 0.5, 0.61, txt, ha="center", va="center", fontsize=16)
    if i < len(etapas) - 1:
        ax.annotate("", xy=(i + 1.06, 0.61), xytext=(i + 0.94, 0.61), arrowprops=dict(arrowstyle="->", lw=2.5))
ax.set_title("F3 — Arquitetura do Pipeline (verde = ajustado somente no treino de cada dobra)")
ax.text(0.06, 0.08, "Conjunto de teste (20%) isolado antes da EDA e avaliado uma única vez na Seção 9.", fontsize=16)
plt.tight_layout()
salvar("F3_pipeline")

# F4: Antes e depois do ajuste (RMSE de CV ± desvio) com linha preta tracejada do baseline
pad_f4 = resultados[resultados.versao == "padrao"].set_index("modelo")
aj_f4 = resultados[resultados.linha_T1 == "5-7"].set_index("modelo")
base_rmse = float(resultados.loc[resultados.versao == "baseline", "rmse_cv_media"].iloc[0])

x = np.arange(len(MODELOS))
w = 0.36
fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(
    x - w / 2, pad_f4.loc[MODELOS, "rmse_cv_media"], w,
    yerr=pad_f4.loc[MODELOS, "rmse_cv_desvio"], capsize=6,
    color="lightgray", edgecolor="black", label="Padrão (antes do ajuste)",
)
ax.bar(
    x + w / 2, aj_f4.loc[MODELOS, "rmse_cv_media"], w,
    yerr=aj_f4.loc[MODELOS, "rmse_cv_desvio"], capsize=6,
    color=[COR[m] for m in MODELOS], edgecolor="black", label="Ajustado (sem redução)",
)
ax.axhline(base_rmse, color="black", ls="--", label=f"Baseline Dummy = {base_rmse:.3f}")
ax.set_xticks(x, MODELOS)
ax.set_ylabel("RMSE médio CV (mg/m³)")
ax.set_title("F4 — Antes e depois do ajuste de hiperparâmetros (CV 5-folds)")
ax.set_ylim(0, max(base_rmse * 1.30, 1.85))
ax.legend(loc="upper right")
plt.tight_layout()
salvar("F4_antes_depois")

# F5: Curva de validação (treino e validação com barras de desvio) do hiperparâmetro de maior amplitude
mod_f5 = str(df_vc["modelo"].iloc[0])
hp_f5 = str(df_vc["hiperparametro"].iloc[0])
fig, ax = plt.subplots(figsize=(11, 6))
ax.errorbar(
    df_vc["valor"], df_vc["rmse_treino_media"], yerr=df_vc["rmse_treino_desvio"],
    marker="s", ls="--", capsize=5, color="tab:red", label="Treino (CV 5-folds)",
)
ax.errorbar(
    df_vc["valor"], df_vc["rmse_val_media"], yerr=df_vc["rmse_val_desvio"],
    marker="o", ls="-", capsize=5, color=COR[mod_f5], label="Validação (CV 5-folds)",
)
if (df_vc["valor"] > 0).all() and (df_vc["valor"].max() / max(df_vc["valor"].min(), 1e-12) >= 20):
    ax.set_xscale("log")
ax.set_xlabel(f"{mod_f5}: {hp_f5}")
ax.set_ylabel("RMSE (mg/m³)")
ax.set_title(f"F5 — Sensibilidade de {mod_f5} ao hiperparâmetro {hp_f5}")
ax.grid(True, ls=":", alpha=0.6)
ax.legend()
plt.tight_layout()
salvar("F5_sensibilidade_hiperparametro")
"""
    )

    code(
        """
# F6: Variância explicada acumulada pelo PCA (somente treino) com as duas configurações marcadas
fig, ax = plt.subplots(figsize=(11, 6))
comp_eixo = np.arange(1, len(var_acum) + 1)
ax.plot(comp_eixo, var_acum * 100.0, marker="o", color="tab:blue", label="Variância acumulada (%)")
ax.axhline(95.0, color="black", ls="--", label="Limiar 95%")
ax.axvline(K_PCA_95, color="tab:red", ls=":", label=f"PCA 95%: {K_PCA_95} comp. ({100*var_acum[K_PCA_95-1]:.1f}%)")
ax.axvline(K_PCA, color="tab:green", ls="-.", label=f"PCA k: {K_PCA} comp. ({100*var_acum[K_PCA-1]:.1f}%)")
ax.set_xticks(comp_eixo)
ax.set_xlabel("Número de componentes principais")
ax.set_ylabel("Variância explicada acumulada (%)")
ax.set_title("F6 — Curva de variância explicada pelo PCA (treino)")
ax.grid(True, ls=":", alpha=0.6)
ax.legend(loc="lower right")
plt.tight_layout()
salvar("F6_variancia_pca")

# F7: Projeção nas duas primeiras componentes principais (treino), coloridas pelo alvo CO(GT)
pc2 = pca_full.transform(Xs_tr)[:, :2]
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(pc2[:, 0], pc2[:, 1], c=y_tr.values, cmap="viridis", s=16, alpha=0.65, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("CO(GT) (mg/m³)")
ax.set_xlabel(f"PC1 ({100*var_exp[0]:.1f}% da variância)")
ax.set_ylabel(f"PC2 ({100*var_exp[1]:.1f}% da variância)")
ax.set_title("F7 — Dados de treino nas duas primeiras componentes do PCA")
ax.grid(True, ls=":", alpha=0.5)
plt.tight_layout()
salvar("F7_componentes")

# F8: Comparação no teste — sem redução vs com redução escolhida por CV (RMSE e tempo de treino)
sem_f8 = resultados[resultados.linha_T1 == "5-7"].set_index("modelo")
red_f8 = resultados[resultados.linha_T1 == "8-10"].set_index("modelo")
grupos_f8 = ["Sem redução (13 atrib.)", "Com redução escolhida (CV)"]

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
x_f8 = np.arange(len(grupos_f8))
w_f8 = 0.24
max_r8, max_t8 = 0.0, 0.0
for i, nome in enumerate(MODELOS):
    y_rmse = [float(sem_f8.loc[nome, "rmse_teste"]), float(red_f8.loc[nome, "rmse_teste"])]
    y_time = [float(sem_f8.loc[nome, "tempo_treino"]), float(red_f8.loc[nome, "tempo_treino"])]
    max_r8 = max(max_r8, max(y_rmse))
    max_t8 = max(max_t8, max(y_time))
    axes[0].bar(x_f8 + (i - 1) * w_f8, y_rmse, w_f8, color=COR[nome], edgecolor="black", label=nome)
    axes[1].bar(x_f8 + (i - 1) * w_f8, y_time, w_f8, color=COR[nome], edgecolor="black", label=nome)

axes[0].set_title("F8a — RMSE no conjunto de teste")
axes[0].set_ylabel("RMSE de teste (mg/m³)")
axes[0].set_ylim(0, max_r8 * 1.25)
axes[1].set_title("F8b — Tempo de treino (treino completo)")
axes[1].set_ylabel("Tempo de treino (s)")
axes[1].set_ylim(0, max_t8 * 1.25)
for ax in axes:
    ax.set_xticks(x_f8, grupos_f8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper center", ncol=3)
plt.tight_layout()
salvar("F8_com_sem_reducao")

# F9: Erros do modelo escolhido por CV (previsto vs real e resíduos no teste)
pr_c = predicoes[(predicoes.modelo == CAMPEAO) & (predicoes.configuracao == CFG_CAMPEAO)]
real = pr_c["valor_real"].values
prev = pr_c["valor_previsto"].values
resid = real - prev

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
lim = [0.0, float(max(real.max(), prev.max())) * 1.03]
axes[0].scatter(real, prev, s=18, alpha=0.5, color=COR[CAMPEAO], edgecolors="none")
axes[0].plot(lim, lim, "k--", label="Previsão perfeita (y = x)")
axes[0].set_xlabel("CO(GT) real (mg/m³)")
axes[0].set_ylabel("CO(GT) previsto (mg/m³)")
axes[0].set_title(f"F9a — {CAMPEAO} ({CFG_CAMPEAO}): previsto × real")
axes[0].grid(True, ls=":", alpha=0.5)
axes[0].legend()

axes[1].scatter(prev, resid, s=18, alpha=0.5, color=COR[CAMPEAO], edgecolors="none")
axes[1].axhline(0.0, color="black", ls="--", label="Resíduo zero")
axes[1].set_xlabel("CO(GT) previsto (mg/m³)")
axes[1].set_ylabel("Resíduo = real − previsto (mg/m³)")
axes[1].set_title(f"F9b — Resíduos do modelo escolhido ({CAMPEAO})")
axes[1].grid(True, ls=":", alpha=0.5)
axes[1].legend()
plt.tight_layout()
salvar("F9_erros_modelo_escolhido")

faixas_q = pd.qcut(real, 4)
print("--- RMSE no teste por quartil do alvo (modelo escolhido) ---")
print(pd.Series(resid ** 2).groupby(faixas_q, observed=True).mean().pow(0.5).round(4).to_string())
"""
    )

    code(
        """
# ---------- Tabela T1 (exatamente 10 linhas) ----------
def txt_cv(df_, k):
    return df_.apply(
        lambda r: f"{r[k + '_cv_media']:.3f} ± {r[k + '_cv_desvio']:.3f}" if pd.notna(r[k + "_cv_media"]) else "—",
        axis=1,
    )


T1_vis = pd.DataFrame({
    "linha": resultados["linha_T1"],
    "modelo": resultados["modelo"],
    "versao": resultados["versao"],
    "configuracao": resultados["configuracao"],
    "n_atributos": resultados["n_atributos"],
    "RMSE CV": txt_cv(resultados, "rmse"),
    "MAE CV": txt_cv(resultados, "mae"),
    "R² CV": txt_cv(resultados, "r2"),
    "MAPE CV (%)": txt_cv(resultados, "mape") if MAPE_OK else "—",
    "RMSE teste": resultados["rmse_teste"].round(4),
    "MAE teste": resultados["mae_teste"].round(4),
    "R² teste": resultados["r2_teste"].round(4),
    "MAPE teste (%)": resultados["mape_teste"].round(2) if MAPE_OK else np.nan,
    "t treino (s)": resultados["tempo_treino"].round(4),
    "t previsao (s)": resultados["tempo_previsao"].round(4),
})
T1_vis.to_csv(caminho_csv("T1.csv"), index=False)
print("=== TABELA T1 (10 configurações obrigatórias) ===")
print(T1_vis.to_string(index=False))

# ---------- Tabela T2 (ajuste de hiperparâmetros e bordas) ----------
aj_rows = resultados[resultados.linha_T1 == "5-7"].set_index("modelo")
T2 = hiper.copy()
for k in METRICAS:
    T2[f"{k.upper()} CV (ajustado)"] = T2["modelo"].map(
        lambda m, k=k: f"{aj_rows.loc[m, k + '_cv_media']:.3f} ± {aj_rows.loc[m, k + '_cv_desvio']:.3f}"
    )
T2.to_csv(caminho_csv("T2.csv"), index=False)
print("\\n=== TABELA T2 (Hiperparâmetros, espaço, tempo e bordas) ===")
print(T2.to_string(index=False))

# ---------- Tabela complementar de reduções (somente CV, média por dobra) ----------
tcomp = pd.DataFrame({
    "modelo": cv_red["modelo"],
    "configuracao": cv_red["configuracao"],
    "n_atributos": cv_red["n_atributos"],
    "RMSE CV": txt_cv(cv_red, "rmse"),
    "MAE CV": txt_cv(cv_red, "mae"),
    "R² CV": txt_cv(cv_red, "r2"),
    "MAPE CV (%)": txt_cv(cv_red, "mape") if MAPE_OK else "—",
    "t treino médio dobra (s)": cv_red["tempo_treino_cv_s"].round(4),
    "t prev. médio dobra (s)": cv_red["tempo_previsao_cv_s"].round(5),
})
print("\\n=== TABELA COMPLEMENTAR: Redução de dimensionalidade (CV 5-folds no treino) ===")
print(tcomp.to_string(index=False))

# ---------- Grade do SelectKBest ----------
gk = grade_cv[grade_cv.tipo == "reducao"][["modelo", "hp_k", "rmse_cv_media", "rmse_cv_desvio"]].copy()
gk["hp_k"] = pd.to_numeric(gk["hp_k"], errors="coerce")
gk["RMSE CV"] = gk.apply(lambda r: f"{r['rmse_cv_media']:.4f} ± {r['rmse_cv_desvio']:.4f}", axis=1)
gk = gk.sort_values(["modelo", "hp_k"])[["modelo", "hp_k", "RMSE CV"]].rename(columns={"hp_k": "k"})
print("\\n=== GRADE DO SELECTKBEST (CV 5-folds no treino) ===")
print(gk.to_string(index=False))
"""
    )

    # ------------------------------------------------------------------
    # ANÁLISE COMPLEMENTAR (SOMENTE TREINO)
    # ------------------------------------------------------------------
    md(
        """
## Análise complementar (somente treino): impacto da autocorrelação temporal

Para quantificar o otimismo introduzido pelo `KFold` aleatório em uma série temporal horária, comparamos — **exclusivamente no treino** ordenado cronologicamente por `datetime` — o desempenho do modelo escolhido (`CAMPEAO`, `CFG_CAMPEAO`) entre `KFold(shuffle=True, random_state=42)` e `KFold(shuffle=False)` (blocos temporais contíguos). O conjunto de teste não é tocado.
"""
    )

    code(
        """
if RODAR_BUSCAS:
    ordem_t = train_datetime.sort_values().index
    X_tr_ord = X_tr.loc[ordem_t]
    y_tr_ord = y_tr.loc[ordem_t]

    pipe_camp = pipeline_final(CAMPEAO, CFG_CAMPEAO)
    kf_emb = KFold(n_splits=5, shuffle=True, random_state=SEED)
    kf_seq = KFold(n_splits=5, shuffle=False)

    res_emb = avaliar_cv(pipe_camp, X_tr_ord, y_tr_ord, cv_obj=kf_emb)
    res_seq = avaliar_cv(pipe_camp, X_tr_ord, y_tr_ord, cv_obj=kf_seq)

    df_temp = pd.DataFrame([
        {"esquema": "KFold aleatório (shuffle=True)", "modelo": CAMPEAO, "configuracao": CFG_CAMPEAO, **res_emb},
        {"esquema": "KFold em blocos temporais (shuffle=False)", "modelo": CAMPEAO, "configuracao": CFG_CAMPEAO, **res_seq},
    ])
    df_temp.to_csv(caminho_csv("analise_temporal_cv.csv"), index=False)
else:
    df_temp = ler_csv("analise_temporal_cv.csv")

m_emb = float(df_temp.loc[0, "rmse_cv_media"])
s_emb = float(df_temp.loc[0, "rmse_cv_desvio"])
m_seq = float(df_temp.loc[1, "rmse_cv_media"])
s_seq = float(df_temp.loc[1, "rmse_cv_desvio"])
print(f"Modelo: {CAMPEAO} ({CFG_CAMPEAO})")
print(f"RMSE KFold aleatório          : {m_emb:.4f} ± {s_emb:.4f} mg/m³")
print(f"RMSE KFold blocos temporais   : {m_seq:.4f} ± {s_seq:.4f} mg/m³")
print(f"Diferença (otimismo aleatório): {m_seq - m_emb:+.4f} mg/m³ ({100 * (m_seq - m_emb) / m_emb:+.1f}%)")
"""
    )

    nb = nbf.v4.new_notebook()
    nb["cells"] = celulas
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    return nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo-true", action="store_true", help="Gera P2_AirQuality_v3.ipynb com RODAR_BUSCAS = True.")
    parser.add_argument("--modo-false", action="store_true", help="Gera cópia com RODAR_BUSCAS = False para teste.")
    args = parser.parse_args()

    smoke = os.environ.get("P2_SMOKE") == "1"
    if smoke:
        nome_arq = "P2_AirQuality_v3_smoke.ipynb"
        rodar = True
    elif args.modo_false:
        nome_arq = "P2_AirQuality_v3_modoFalse.ipynb"
        rodar = False
    else:
        nome_arq = "P2_AirQuality_v3.ipynb"
        rodar = bool(args.modo_true)

    nb = montar_celulas(rodar_buscas=rodar, smoke=smoke)
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome_arq)
    with open(destino, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook gerado: {destino} ({len(nb['cells'])} células)")


if __name__ == "__main__":
    main()
