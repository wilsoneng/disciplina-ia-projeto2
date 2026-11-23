# Plano — Terceiro Notebook Corrigido (Projeto 2 · R6 Air Quality)

## Objetivo
Criar um terceiro notebook (apenas código, sem relatório) para o **Projeto 2 — Comparação de Modelos Supervisionados com Dados Reais (Dataset R6: Air Quality, UCI ID 360, regressão de `CO(GT)`)**, reunindo e corrigindo as falhas identificadas nos dois notebooks anteriores (`amos/Projeto2A3.ipynb` e `projeto_final_Wilson_Amos (1).ipynb`) frente às instruções do enunciado (`amos/Projeto 2 - IA - 2026.2_1.pdf`).

Base estrutural: `/home/wilson/disciplina_ia/projeto_final/amos/Projeto2A3.ipynb` (Colab A, sem vazamento), incorporando as melhorias do Colab B e corrigindo os problemas de ambos.

## Estrutura da pasta `/home/wilson/disciplina_ia/projeto_final/v3/`
- `PLANO_notebook_corrigido.md`: este plano + checklist do enunciado + seção "Estado atual / próximos passos" (documento de retomada para outra IA ou humano).
- `build_notebook.py`: script Python que gera `P2_AirQuality_v3.ipynb` via `nbformat` (células escritas como strings). Toda edição futura deve ser feita aqui e o script reexecutado (nunca editar o JSON do `.ipynb` diretamente).
- `P2_AirQuality_v3.ipynb`: notebook final gerado e executado com todas as saídas salvas.
- `resultados/`: CSVs gerados pelo notebook (`resultados_configuracoes.csv`, `melhores_hiperparametros.csv`, `cv_grade_completa.csv`, `predicoes_teste.csv`, `curva_validacao.csv`, `cv_reducao.csv`, `efeito_atributos.csv`, `T1.csv`, `T2.csv`).
- `figuras/`: figuras geradas em PNG (200 dpi, fonte mínima 16 pt).

---

## Correções consolidadas (A + B)
1. **Divisão antes da EDA (corrige B):** `train_test_split(..., test_size=0.2, random_state=42)` ocorre no início da Seção 3, antes de qualquer estatística, gráfico ou decisão de remoção/tratamento. Apenas checagens estruturais (shape, dtypes, contagem de `-200`, duplicatas, parse de datas) usam o dataset completo na Seção 2.
2. **Escolha do modelo exclusivamente por CV (corrige falha crítica de B):** nunca usa métricas de teste (`*_teste`) na decisão; usa o critério formal pré-declarado sobre a validação cruzada 5-fold (menor RMSE CV; empate técnico se diferença < desvio padrão da melhor → menos atributos → maior interpretabilidade → menor tempo de treino), com `assert` explícito garantindo que colunas `*_teste` não entram na seleção.
3. **Linhas 8–10 da T1 com redução real (corrige falha de A):** a redução escolhida por modelo para as linhas 8–10 da T1 é a de menor RMSE de CV entre as configurações **com redução** (`pca95`, `pca_k`, `selectk_best`), em vez de copiar "sem redução".
4. **Avaliação única e estrita no teste (corrige falha de A):** na Seção 9, o conjunto de teste é avaliado **exatamente nas 10 configurações da T1** (1 baseline, 3 padrão, 3 ajustadas sem redução, 3 ajustadas com redução escolhida).
5. **Mesmo estimador linear antes e depois (corrige falha de B):** `ElasticNet(random_state=42, max_iter=20000)` na Seção 6 (padrão) e na Seção 7 (ajustado).
6. **Reprodutibilidade real com `RODAR_BUSCAS = False` (corrige A e B):** elimina a célula `LIMPAR_TUDO` de A e elimina os dicionários hardcoded de B. Cria `ler_csv(nome)` que lê da `URL_BASE` (quando preenchida) ou de `resultados/<nome>` localmente com aviso explícito, permitindo reproduzir tabelas e figuras em segundos/poucos minutos sem tocar no teste.
7. **Engenharia de atributos dentro do Pipeline (combina o melhor de B e A):** transformador customizado `AtributosAirQuality(BaseEstimator, TransformerMixin)` dentro do `Pipeline`, gerando `hora_sin`, `hora_cos`, `fim_de_semana`, `razao_s1_tempK` e `temp_x_umid` (13 atributos finais a partir dos 8 base), medindo o efeito por CV 5-fold nos **três** modelos padrão.
8. **Ajuste de hiperparâmetros com tratamento de bordas (corrige B e amplia A):** função única `buscar()` sobre o `Pipeline` completo; `GridSearchCV` para ElasticNet (com extensão de borda), `RandomizedSearchCV` para Random Forest (`n_iter=20`) e SVR (`n_iter=25`, `loguniform`), com detecção de borda (10% extremos, em escala log para parâmetros contínuos de SVR) e 1 rodada extra se RF ou SVR caírem na borda.
9. **Figuras 100% conforme Seção 10 do enunciado:** `plt.rcParams` com fonte >= 16 pt em **todos** os textos/anotações (corrige B), paleta consistente por modelo, baseline preto tracejado, barras de erro com `ddof=1`, `F5` com `validation_curve` (treino e validação com barras de erro) salva em CSV, `F8` comparando sem vs com redução escolhida (RMSE teste e tempo de treino).
10. **Compatibilidade Colab e versões antigas:** instalação automática opcional de `ucimlrepo` no topo, célula explícita de versões das bibliotecas, RMSE calculado via NumPy (sem depender de `root_mean_squared_error` de versões recentes do scikit-learn).

---

## Estrutura do Notebook (Seções exigidas no item 12.2 do enunciado)
- Cabeçalho Markdown (título, fonte/DOI/licença, instruções de execução, regras de protocolo)
- **1. Configuração** (dependências, imports, `RODAR_BUSCAS`, `SEED=42`, versões, `rcParams`, `COR`, `URL_BASE`/`ler_csv`)
- **2. Carga dos dados** (`fetch_ucirepo(360)`, exibição inicial, dicionário com unidades, `-200`, duplicatas, faixas físicas, parse temporal com asserts, declarações de ausência de grupos e categóricas)
- **3. Análise exploratória** (divisão preliminar 80/20 dos índices após filtro de `CO(GT) == -200` + EDA **somente no treino**: F1, descritivas, faltantes, Spearman/Pearson, F2a, F2b, autocorrelação lag 1, pares redundantes `|rho|>0,8`, 5 achados ligados a decisões)
- **4. Divisão dos dados** (materialização apenas de `X_tr, y_tr`, `KFold(5, shuffle=True, random_state=42)`, discussão de grupos e autocorrelação)
- **5. Tratamento e engenharia de atributos** (exclusão de `Date`/`Time`/`NMHC(GT)`/`C6H6(GT)`/`NOx(GT)`/`NO2(GT)`, `-200 -> NaN`, tabela de outliers 3×IQR e faixas físicas, `AtributosAirQuality`, `pipe()`, efeito dos 5 atributos novos nos 3 modelos padrão -> `efeito_atributos.csv`)
- **6. Baseline e modelos padrão** (`DummyRegressor`, `ElasticNet`, `RandomForestRegressor`, `SVR` com laço manual, `perf_counter`, `ddof=1`, MAPE se `MAPE_OK`)
- **7. Ajuste de hiperparâmetros** (função única `buscar()`, grades/distribuições, detecção/extensão de bordas, `melhores_hiperparametros.csv`, `cv_grade_completa.csv`, `curva_validacao.csv`, discussão custo × ganho)
- **8. Redução de dimensionalidade** (curva de variância do PCA no treino, `PCA(n_components=0.95)` e `PCA(n_components=K_menor)`, `SelectKBest(f_regression)` com `k in {4, 6, 8, 10}`, avaliação CV completa -> `cv_reducao.csv`, seleção da melhor redução por modelo)
- **9. Avaliação no teste** (bloco `if RODAR_BUSCAS:` com leitura única do teste para exatamente as 10 configurações da T1 -> `resultados_configuracoes.csv` e `predicoes_teste.csv`; escolha do modelo a entregar exclusivamente por CV com `assert`; modo `RODAR_BUSCAS=False` refazendo ajuste apenas do modelo escolhido no treino)
- **10. Figuras e tabelas** (F3 a F9 e tabelas T1, T2, complementar de reduções e grade SelectKBest; exportação `T1.csv` e `T2.csv`)
- **Análise complementar (somente treino)** (comparação de CV KFold embaralhada vs KFold sequencial sem embaralhamento no treino ordenado por `datetime` para quantificar o otimismo da divisão aleatória)

---

## Checklist do Enunciado (Projeto 2 — Seção 13)
- [ ] Dataset confirmado com o professor até 18/11/2026
- [ ] Checkpoint enviado em 26/11/2026
- [x] Análise exploratória com achados ligados às decisões de tratamento *(no código)*
- [x] Pelo menos dois atributos novos, com o efeito medido *(5 atributos, medido nos 3 modelos)*
- [x] Baseline, modelo linear e mais dois modelos de famílias diferentes *(Dummy, ElasticNet, RF, SVR)*
- [x] Busca de hiperparâmetros nos três modelos, sobre o pipeline inteiro
- [x] PCA em duas configurações e comparação com e sem redução *(+ SelectKBest)*
- [x] Teste usado uma única vez, na tabela T1 *(10 linhas exatas)*
- [x] Todas as figuras obrigatórias, com fonte de pelo menos 16 pt
- [ ] Relatório no template REPA, com a fonte do dataset e o link do Colab *(fora do escopo deste código)*
- [ ] Slides: 9 de conteúdo, mais capa e final, testados em tela cheia *(fora do escopo deste código)*
- [ ] Ensaio cronometrado da dupla abaixo de 15 minutos
- [ ] Colab testado em janela anônima, com "Executar tudo" sem erro
- [ ] Arquivos no padrão de nome, enviados até 03/12/2026 às 23h59

---

## Estado atual / próximos passos
- **Status:** Concluído e validado ponta a ponta após as 5 correções finais.
- **Execução completa (`RODAR_BUSCAS = True`):** [P2_AirQuality_v3.ipynb](P2_AirQuality_v3.ipynb) executado sequencialmente (células 1–22 sem erros), gerando 10 figuras (`F1`, `F2a`, `F2b`, `F3`–`F9`) em `figuras/` e 10 CSVs em `resultados/`.
  - Tempos de busca (`GridSearchCV`/`RandomizedSearchCV`): ElasticNet = `23,99 s` (72 comb., 2 rodadas, `l1_ratio` estendido até `0,01`), RandomForest = `233,15 s` (35 comb., 2 rodadas, sem borda ao final), SVR = `466,11 s` (30 comb., 1 rodada, incluindo `kernel ∈ {'rbf', 'linear'}`).
  - Efeito dos 5 atributos novos (CV 5-folds, parâmetros padrão): melhorou os 3 modelos (`ElasticNet`: `0,9205 -> 0,9057`; `RandomForest`: `0,5451 -> 0,5109`; `SVR`: `0,5562 -> 0,5182`).
  - Redução PCA escolhida por CV para as linhas 8–10 da T1: `pca95` (`7` atributos) para os três modelos (`SelectKBest` mantido na tabela complementar e grade em CV).
  - **Modelo escolhido (exclusivamente por CV, via desempate técnico):** `RandomForest (ajustado | nenhum)` com `13` atributos — `RMSE CV = 0,499 ± 0,037 mg/m³` (empate técnico com `SVR` ajustado `0,480 ± 0,063`, `Δ = 0,0187 < dp = 0,0625`, desempatado por maior interpretabilidade). No teste (leitura única): `RMSE = 0,4873 mg/m³`, `MAE = 0,2700 mg/m³`, `R² = 0,8860`, `MAPE = 19,39%`.
  - Análise complementar temporal (somente treino): `KFold` em blocos temporais elevou o RMSE de `0,4929 ± 0,0330` para `0,5303 ± 0,2040` (`+0,0374 mg/m³`, `+7,6%`).
- **Teste do modo `RODAR_BUSCAS = False`:** executado em `16,26 s` sem erros (modelo campeão reajustado no treino em `2,03 s`), reproduzindo a mesma escolha e todas as tabelas/figuras.
- **Publicação no GitHub:** repositório público criado em `https://github.com/wilsoneng/disciplina-ia-projeto2` contendo `v3/` (sem `.venv`), com `URL_BASE = "https://raw.githubusercontent.com/wilsoneng/disciplina-ia-projeto2/main/resultados/"` e `RODAR_BUSCAS = False` por padrão em `P2_AirQuality_v3.ipynb`.
- **Pendências do usuário antes da entrega final:**
  1. Abrir `P2_AirQuality_v3.ipynb` no Google Colab (ou importar direto do GitHub `wilsoneng/disciplina-ia-projeto2`), ativar compartilhamento público ("Qualquer pessoa com o link: leitor"), rodar "Executar tudo" e testar em janela anônima.
  2. Atualizar o relatório (template REPA) e os slides com os novos números/figuras gerados em `v3/resultados/` e `v3/figuras/`.
