import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Painel de Avaliação e Explicabilidade",
    page_icon="📊",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

h1, h2, h3 {
    margin-bottom: 0.3rem !important;
}

p, .stCaption {
    margin-bottom: 0.35rem !important;
}

div[data-testid="stMetric"] {
    border: 1px solid #E6E6E6;
    border-radius: 10px;
    padding: 10px 12px;
    background: #FAFAFA;
}

div[data-testid="stExpander"] {
    border: 1px solid #EAEAEA;
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# Por padrão, procura os artefatos em ./resultados ao lado deste arquivo.
# Se quiser sobrescrever sem editar o código, defina a variável de ambiente:
# STREAMLIT_WEB_DIR=E:\\mestrado\\web\\resultados
COMMITTEE_NAME = "Comite_Arvores_XGB_LGBM_CatBoost"


def get_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


BASE_DIR = Path(get_secret("STREAMLIT_WEB_DIR", Path(__file__).parent / "resultados")).resolve()
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-5-mini")


# ============================================================
# Classe do comitê para deserialização do pickle
# ============================================================
class PrefitCommittee(BaseEstimator, ClassifierMixin):
    def __init__(self, models: dict, weights: np.ndarray, threshold: float = 0.5, feature_names=None):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        self.threshold_ = float(threshold)
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        s = self.weights.sum()
        if not np.isclose(s, 1.0):
            self.weights = self.weights / (s if s > 0 else 1.0)

    def _to_df(self, X):
        if isinstance(X, (pd.Series, dict)):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.feature_names_ is not None:
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(f"Faltam colunas no input: {missing}")
            X = X[self.feature_names_]
        return X

    def predict_proba(self, X):
        X = self._to_df(X)
        probs = []
        for name in ("XGBoost", "LightGBM", "CatBoost"):
            p = self.models[name].predict_proba(X)[:, 1]
            probs.append(p)
        probs = np.vstack(probs).T
        p1 = np.dot(probs, self.weights)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold_).astype(int)


# ============================================================
# Helpers gerais
# ============================================================
def humaniza_slug(nome: str) -> str:
    texto = str(nome).strip()
    texto = re.sub(r"_+", " ", texto)
    texto = re.sub(r"\s*-\s*", " - ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    palavras_minusculas = {"de", "da", "do", "das", "dos", "e"}
    partes = []
    for palavra in texto.split(" "):
        if palavra == "-":
            partes.append(palavra)
        elif palavra.lower() in palavras_minusculas:
            partes.append(palavra.lower())
        else:
            partes.append(palavra[:1].upper() + palavra[1:].lower())
    return " ".join(partes)


def pasta_curso_periodo(curso_slug: str, periodo: str) -> Path:
    return BASE_DIR / curso_slug / periodo


def lista_cursos() -> list[str]:
    if not BASE_DIR.exists():
        return []
    cursos = [p.name for p in BASE_DIR.iterdir() if p.is_dir()]
    return sorted(cursos, key=lambda x: humaniza_slug(x).lower())


def lista_periodos(curso_slug: Optional[str]) -> list[str]:
    if curso_slug is None:
        return []
    pasta = BASE_DIR / curso_slug
    if not pasta.exists():
        return []
    periodos = [p.name for p in pasta.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(periodos, key=int)


@st.cache_data(show_spinner=False)
def carregar_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def carregar_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def carregar_npy(path: Path):
    return np.load(path, allow_pickle=True)


@st.cache_resource(show_spinner=False)
def carregar_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Helpers de nomes amigáveis
# ============================================================
def _extrai_periodo(nome: str) -> Optional[str]:
    m = re.search(r"periodo_(\d+)_", nome)
    return m.group(1) if m else None


def _ordinal_pt(num: str) -> str:
    return f"{int(num)}º"


ALIAS_FIXOS = {
    "nota_ingresso": "nota de ingresso",
}

ALIAS_BINARIOS = {
    "tipo_ingresso_pism": "ingresso por PISM",
    "tipo_ingresso_sisu": "ingresso por SISU/ENEM",
    "tipo_ingresso_vestibular": "ingresso por vestibular",
    "tipo_ingresso_outros": "ingresso por outras vias",
    "cota_racial": "cota racial",
    "cota_ampla_concorrencia": "ampla concorrência",
    "cota_escola_publica": "cota de escola pública",
    "cota_renda": "cota por renda",
    "cota_pcd": "cota para PCD",
    "genero_masculino": "gênero masculino",
    "genero_feminino": "gênero feminino",
    "genero_outros": "outro gênero",
    "etnia_branca": "etnia branca",
    "etnia_parda": "etnia parda",
    "etnia_preta": "etnia preta",
    "etnia_outra": "outra etnia",
}


def alias_por_periodo(nome: str) -> Optional[str]:
    p = _extrai_periodo(nome)
    if not p:
        return None
    ordp = _ordinal_pt(p)
    base_map = {
        f"periodo_{p}_disciplinas_aprovadas": f"disciplinas aprovadas no {ordp} período",
        f"periodo_{p}_disciplinas_reprovadas": f"disciplinas reprovadas no {ordp} período",
        f"periodo_{p}_disciplinas_ri": f"reprovações por infrequência no {ordp} período",
        f"periodo_{p}_disciplinas_trancadas": f"disciplinas trancadas no {ordp} período",
        f"periodo_{p}_disciplinas_outros_status": f"disciplinas com outros status no {ordp} período",
        f"periodo_{p}_bolsa_remunerada": f"bolsa remunerada no {ordp} período",
        f"periodo_{p}_bolsa_n_remunerada": f"bolsa não remunerada no {ordp} período",
        f"periodo_{p}_ae": f"assistência estudantil no {ordp} período",
    }
    return base_map.get(nome)


def humaniza_variavel(nome: str) -> str:
    nome_l = nome.lower()
    alias_p = alias_por_periodo(nome_l)
    if alias_p:
        return alias_p
    if nome_l in ALIAS_FIXOS:
        return ALIAS_FIXOS[nome_l]
    if nome_l in ALIAS_BINARIOS:
        return ALIAS_BINARIOS[nome_l]
    return nome.replace("_", " ")


# ============================================================
# Carregamento dos artefatos do período
# ============================================================
def carregar_artefatos_periodo(pasta: Path) -> dict:
    arquivos = {
        "comite_model": pasta / "comite_model.pkl",
        "shap_values": pasta / "shap_values_Comite_Arvores.npy",
        "x_shap": pasta / "x_treino_balanceado_Comite_Arvores.csv",
        "ranking": pasta / "shap_ranking_Comite_Arvores.csv",
        "metricas_repeticao": pasta / "02_logs" / "metricas_por_repeticao.csv",
        "resumo_metricas": pasta / "00_dashboard" / "resumo_metricas.csv",
        "melhor_execucao": pasta / "00_dashboard" / "melhor_execucao_por_modelo.csv",
        "tabela_agregada": pasta / "00_dashboard" / "tabela_agregada.csv",
        "ic95": pasta / "00_dashboard" / "ic95_f1_pos.csv",
    }

    faltando = [k for k, v in arquivos.items() if not v.exists()]
    return {"paths": arquivos, "faltando": faltando}


# ============================================================
# SHAP global
# ============================================================
def calcular_top_shap(shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 10):
    importancias = np.mean(np.abs(shap_values), axis=0)
    total = importancias.sum() if np.isfinite(importancias).all() else 0.0
    ordem = np.argsort(importancias)[::-1][:top_n]

    dados = []
    for idx in ordem:
        nome = X.columns[idx]
        pct = (importancias[idx] / total * 100) if total > 0 else 0.0
        dados.append(
            {
                "feature": nome,
                "feature_label": humaniza_variavel(nome),
                "importancia": float(importancias[idx]),
                "peso_pct": float(pct),
            }
        )

    return pd.DataFrame(dados), importancias, total


def gerar_listas_risco_permanencia(shap_values: np.ndarray, X: pd.DataFrame, top_features: list[str]):
    riscos = []
    permanencias = []

    for nome in top_features:
        idx = X.columns.get_loc(nome)
        valores_unicos = set(pd.Series(X[nome]).dropna().unique())
        tipo_binario = valores_unicos.issubset({0, 1})
        base = humaniza_variavel(nome)

        if tipo_binario:
            try:
                shap_1 = float(np.mean(shap_values[X[nome] == 1, idx])) if 1 in valores_unicos else np.nan
                shap_0 = float(np.mean(shap_values[X[nome] == 0, idx])) if 0 in valores_unicos else np.nan
            except Exception:
                shap_1 = np.nan
                shap_0 = np.nan

            if not np.isnan(shap_1):
                if shap_1 > 0.01:
                    riscos.append(f"Presença de {base}")
                elif shap_1 < -0.01:
                    permanencias.append(f"Presença de {base}")

            if not np.isnan(shap_0):
                if shap_0 > 0.01:
                    riscos.append(f"Ausência de {base}")
                elif shap_0 < -0.01:
                    permanencias.append(f"Ausência de {base}")

        else:
            try:
                q75 = X[nome].quantile(0.75)
                q25 = X[nome].quantile(0.25)
                shap_high = float(np.mean(shap_values[X[nome] >= q75, idx]))
                shap_low = float(np.mean(shap_values[X[nome] <= q25, idx]))
            except Exception:
                shap_high = np.nan
                shap_low = np.nan

            if not np.isnan(shap_high):
                if shap_high > 0.01:
                    riscos.append(f"Valores altos de {base}")
                elif shap_high < -0.01:
                    permanencias.append(f"Valores altos de {base}")

            if not np.isnan(shap_low):
                if shap_low > 0.01:
                    riscos.append(f"Valores baixos de {base}")
                elif shap_low < -0.01:
                    permanencias.append(f"Valores baixos de {base}")

    return riscos, permanencias


# ============================================================
# Figuras prontas da melhor execução
# ============================================================
def localizar_meta_comite(pasta_periodo: Path) -> Optional[Path]:
    pasta_modelos = pasta_periodo / "03_modelos" / COMMITTEE_NAME
    if not pasta_modelos.exists():
        return None
    candidatos = sorted(pasta_modelos.glob("*_meta.json"))
    return candidatos[0] if candidatos else None


def localizar_figuras_melhor_execucao(pasta_periodo: Path) -> dict:
    pasta_figs = pasta_periodo / "01_figuras" / COMMITTEE_NAME
    out = {"confusao": None, "roc": None, "pr": None, "calibracao": None}
    if not pasta_figs.exists():
        return out

    for arq in pasta_figs.glob("*.png"):
        nome = arq.name.lower()
        if "matriz_confusao" in nome:
            out["confusao"] = arq
        elif nome.endswith("roc.png"):
            out["roc"] = arq
        elif nome.endswith("pr.png"):
            out["pr"] = arq
        elif "calibracao" in nome:
            out["calibracao"] = arq
    return out


# ============================================================
# Helpers da explicação local do comitê
# ============================================================
def unwrap_for_shap(model):
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "base_estimator") and model.base_estimator is not None:
            return model.base_estimator
        if hasattr(model, "estimator") and model.estimator is not None:
            return model.estimator

        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            cc = calibrated[0]
            if hasattr(cc, "estimator") and cc.estimator is not None:
                return cc.estimator
            if hasattr(cc, "base_estimator") and cc.base_estimator is not None:
                return cc.base_estimator
    return model


def explicacao_local_comite(modelo_comite, X_background: pd.DataFrame, X_new: pd.DataFrame, proba_final: float):
    cols_pred = list(X_new.columns)

    if len(X_background) > 200:
        X_bg = X_background[cols_pred].sample(200, random_state=0)
    else:
        X_bg = X_background[cols_pred].copy()

    xgb_base = unwrap_for_shap(modelo_comite.models["XGBoost"])
    lgbm_base = unwrap_for_shap(modelo_comite.models["LightGBM"])
    cat_base = unwrap_for_shap(modelo_comite.models["CatBoost"])

    exp_xgb = shap.TreeExplainer(xgb_base, data=X_bg, model_output="probability")
    exp_lgbm = shap.TreeExplainer(lgbm_base, data=X_bg, model_output="probability")
    exp_cat = shap.TreeExplainer(cat_base, data=X_bg, model_output="probability")

    def sv_pos(exp, X_):
        sv = exp.shap_values(X_)
        ev = exp.expected_value
        if isinstance(sv, list):
            sv = sv[1]
            ev = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
        return float(np.ravel([ev])[0]), sv

    ev_xgb, sv_xgb = sv_pos(exp_xgb, X_new[cols_pred])
    ev_lgbm, sv_lgbm = sv_pos(exp_lgbm, X_new[cols_pred])
    ev_cat, sv_cat = sv_pos(exp_cat, X_new[cols_pred])

    w = np.asarray(modelo_comite.weights, dtype=float)
    sv_c = w[0] * sv_xgb + w[1] * sv_lgbm + w[2] * sv_cat

    base_val = proba_final - float(np.sum(sv_c[0]))

    explanation = shap.Explanation(
        values=sv_c[0],
        base_values=base_val,
        data=X_new[cols_pred].values[0],
        feature_names=[humaniza_variavel(c) for c in cols_pred],
    )

    impactos = pd.DataFrame(
        {
            "feature": cols_pred,
            "feature_label": [humaniza_variavel(c) for c in cols_pred],
            "shap": sv_c[0],
            "valor": X_new[cols_pred].iloc[0].values,
        }
    )
    impactos["impacto_abs"] = impactos["shap"].abs()
    impactos = impactos.sort_values("impacto_abs", ascending=False)

    return explanation, impactos


# ============================================================
# OpenAI / ChatGPT
# ============================================================
def gerar_prompt_explicacao(curso: str, periodo: str, riscos: list[str], permanencias: list[str], top_df: pd.DataFrame) -> str:
    linhas_top = [f"- {row.feature_label}: {row.peso_pct:.1f}% do peso global" for row in top_df.itertuples()]
    return f"""
Você está ajudando a interpretar um painel acadêmico de predição de evasão.

Curso: {humaniza_slug(curso)}
Período analisado: p{periodo}
Modelo: Comitê calibrado de árvores (XGBoost + LightGBM + CatBoost)

Explique em linguagem objetiva, voltada para gestores educacionais:
1. quais são os principais sinais de risco de evasão;
2. quais fatores se associam à permanência;
3. como interpretar esses achados com cautela, sem tratá-los como causalidade;
4. que tipo de ação institucional poderia ser priorizada a partir desses sinais.

Principais variáveis por peso:
{chr(10).join(linhas_top) if linhas_top else '- sem dados'}

Sinais associados ao risco:
{chr(10).join('- ' + r for r in riscos) if riscos else '- nenhum sinal destacado'}

Sinais associados à permanência:
{chr(10).join('- ' + p for p in permanencias) if permanencias else '- nenhum sinal destacado'}

Observação importante:
- A nota de ingresso pode apresentar distribuição não homogênea no summary plot. Nesses casos, sua leitura deve ser feita com cautela e em conjunto com as demais variáveis.

Use tom acadêmico, claro e sem exageros.
""".strip()


def gerar_explicacao_openai(prompt: str) -> str:
    if OpenAI is None:
        return "A biblioteca openai não está instalada neste ambiente. Instale com: pip install openai"

    api_key = OPENAI_API_KEY
    if not api_key:
        return "Defina OPENAI_API_KEY nos secrets do Streamlit Cloud."

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "Você é um assistente que explica resultados acadêmicos com clareza, precisão e linguagem acessível.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        texto = getattr(response, "output_text", None)
        if texto:
            return texto

        try:
            return response.output[0].content[0].text
        except Exception:
            return "A resposta foi gerada, mas não foi possível extrair o texto automaticamente."
    except Exception as e:
        return f"Erro ao gerar explicação com a OpenAI: {e}"


# ============================================================
# Interface
# ============================================================
st.title("📊 Painel de Avaliação e Explicabilidade")
st.caption("Análise do comitê com foco em desempenho, SHAP global e simulação de perfis.")

cursos = lista_cursos()
if not cursos:
    st.error(f"Nenhuma pasta de curso encontrada em {BASE_DIR}")
    st.stop()

with st.sidebar:
    st.header("Seleção")
    curso_sel = st.selectbox("Curso", options=cursos, index=0, format_func=humaniza_slug)

if curso_sel is None:
    st.error("Nenhum curso foi selecionado.")
    st.stop()

periodos = lista_periodos(curso_sel)
if not periodos:
    st.error(f"Nenhum período encontrado para o curso selecionado: {curso_sel}")
    st.stop()

with st.sidebar:
    periodo_sel = st.selectbox("Período", options=periodos, index=0, format_func=lambda x: f"p{x}")
    st.markdown("---")
    st.caption(f"Base de artefatos: {BASE_DIR}")
    st.caption(f"Modelo OpenAI: {OPENAI_MODEL}")

pasta = pasta_curso_periodo(curso_sel, periodo_sel)
artefatos = carregar_artefatos_periodo(pasta)
if artefatos["faltando"]:
    st.error("Alguns artefatos obrigatórios não foram encontrados:\n- " + "\n- ".join(artefatos["faltando"]))
    st.stop()

paths = artefatos["paths"]

X_shap = carregar_csv(paths["x_shap"])
shap_values = carregar_npy(paths["shap_values"])
if isinstance(shap_values, (list, tuple)):
    shap_values = shap_values[1]

ranking_df = carregar_csv(paths["ranking"])
metricas_rep = carregar_csv(paths["metricas_repeticao"])
tabela_agregada = carregar_csv(paths["tabela_agregada"])
ic95_df = carregar_csv(paths["ic95"])
modelo_comite = carregar_pickle(paths["comite_model"])

meta_path = localizar_meta_comite(pasta)
meta_comite = carregar_json(meta_path) if meta_path else None
figuras_best = localizar_figuras_melhor_execucao(pasta)

if "status" in X_shap.columns:
    X_shap = X_shap.drop(columns=["status"])

ranking_df["feature_label"] = ranking_df["feature"].apply(humaniza_variavel)

agregado_comite = tabela_agregada[tabela_agregada["modelo"] == COMMITTEE_NAME].copy()
ic95_comite = ic95_df[ic95_df["modelo"] == COMMITTEE_NAME].copy()
metricas_comite = metricas_rep[metricas_rep["modelo"] == COMMITTEE_NAME].copy().sort_values("repeticao")

top_df, importancias_abs, total_imp = calcular_top_shap(shap_values, X_shap, top_n=10)
riscos, permanencias = gerar_listas_risco_permanencia(shap_values, X_shap, top_df["feature"].tolist())

mostrar = agregado_comite.copy()
cols_keep = [
    "modelo",
    "n",
    "f1_pos_media",
    "f1_pos_dp",
    "f1_weighted_media",
    "acuracia_media",
    "roc_auc_media",
    "avg_precision_media",
    "brier_cal_media",
    "threshold_media",
]
cols_keep = [c for c in cols_keep if c in mostrar.columns]

total_instancias = None
if not metricas_comite.empty and {"tr_n", "te_n"}.issubset(metricas_comite.columns):
    total_instancias = int(metricas_comite.iloc[0]["tr_n"] + metricas_comite.iloc[0]["te_n"])

abas = st.tabs([
    "📌 Resumo",
    "🧠 Explicabilidade",
    "🧪 Simulação",
    "⚙️ Detalhes técnicos",
])


# ============================================================
# Aba 1 - resumo
# ============================================================
with abas[0]:
    st.subheader(f"{humaniza_slug(curso_sel)} • p{periodo_sel}")

    c1, c2, c3, c4 = st.columns(4)

    if not agregado_comite.empty:
        row = agregado_comite.iloc[0]
        c1.metric("F1 risco", f"{row.get('f1_pos_media', np.nan):.3f}")
        c2.metric("ROC-AUC", f"{row.get('roc_auc_media', np.nan):.3f}")
        c3.metric("Brier", f"{row.get('brier_cal_media', np.nan):.3f}")
    else:
        c1.metric("F1 risco", "-")
        c2.metric("ROC-AUC", "-")
        c3.metric("Brier", "-")

    c4.metric("Estudantes analisados", f"{total_instancias}" if total_instancias is not None else "-")

    st.markdown("#### Variáveis mais importantes")
    plot_df = top_df.sort_values("importancia")

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.2))
    ax_bar.barh(plot_df["feature_label"], plot_df["importancia"])
    ax_bar.set_xlabel("Média do |SHAP|")
    ax_bar.set_ylabel("")
    ax_bar.set_title("Top 10 variáveis")
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sinais de risco**")
        if riscos:
            for item in riscos[:5]:
                st.markdown(f"- {item}")
        else:
            st.info("Nenhum sinal destacado.")

    with col2:
        st.markdown("**Sinais de permanência**")
        if permanencias:
            for item in permanencias[:5]:
                st.markdown(f"- {item}")
        else:
            st.info("Nenhum sinal destacado.")

    with st.expander("Ver ranking completo"):
        st.dataframe(
            ranking_df.head(15)[["feature_label", "importancia_media"]].rename(
                columns={"feature_label": "Variável", "importancia_media": "Importância média"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Explicação assistida")
    st.caption("Gera uma leitura em linguagem natural com base nos principais sinais do período.")

    if st.button("Gerar explicação em linguagem natural", key="btn_openai_resumo"):
        prompt = gerar_prompt_explicacao(curso_sel, periodo_sel, riscos, permanencias, top_df)
        resposta = gerar_explicacao_openai(prompt)
        st.markdown(resposta)

    with st.expander("Mostrar prompt usado"):
        st.code(gerar_prompt_explicacao(curso_sel, periodo_sel, riscos, permanencias, top_df), language="text")


# ============================================================
# Aba 2 - explicabilidade
# ============================================================
with abas[1]:
    st.subheader("Explicabilidade global")
    st.caption(
        "Contribuições positivas se associam ao aumento do risco de evasão; contribuições negativas se associam à permanência."
    )

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False,
        max_display=10,
        plot_size=(8, 4.8),
    )
    st.pyplot(plt.gcf())
    plt.close(plt.gcf())

    if "nota_ingresso" in [f.lower() for f in top_df["feature"].tolist()]:
        st.info(
            "A nota de ingresso pode apresentar distribuição não homogênea no summary plot. Por isso, sua leitura deve ser feita com cautela e sempre em conjunto com as demais variáveis."
        )

    with st.expander("Observações de leitura"):
        st.markdown(
            """
- O gráfico resume as variáveis que mais influenciaram o comitê.
- A direção do efeito deve ser interpretada com cautela.
- Os sinais indicam associação, não causalidade.
            """
        )


# ============================================================
# Aba 3 - simulador
# ============================================================
with abas[2]:
    st.subheader("Simular um novo perfil")
    st.caption("A previsão usa o mesmo comitê calibrado salvo em `comite_model.pkl`.")

    cols_pred = list(getattr(modelo_comite, "feature_names_", X_shap.columns.tolist()))
    cols_pred_lower = {c.lower(): c for c in cols_pred}
    periodos_detectados = sorted(
        {int(m.group(1)) for c in cols_pred for m in [re.search(r"periodo_(\d+)_", c.lower())] if m}
    )
    periodo_max = int(periodo_sel)

    def nome_coluna(col_name: str) -> str:
        return cols_pred_lower.get(col_name.lower(), col_name)

    def tem_coluna(col_name: str) -> bool:
        return col_name.lower() in cols_pred_lower

    def valor_mediano(col: str, default=0):
        col_real = nome_coluna(col)
        if col_real in X_shap.columns:
            try:
                return int(np.nanmedian(X_shap[col_real].values))
            except Exception:
                return default
        return default

    def faixa_nota():
        col_nota = nome_coluna("nota_ingresso")
        if col_nota in X_shap.columns:
            s = X_shap[col_nota]
            return float(s.min()), float(s.max()), float(s.median())
        return 0.0, 1000.0, 650.0

    with st.form("form_simulador"):
        c1, c2, c3 = st.columns(3)

        vmin, vmax, vmed = faixa_nota()
        nota_ingresso = c1.number_input("Nota de ingresso", min_value=vmin, max_value=vmax, value=vmed)

        op_ingresso = []
        if tem_coluna("tipo_ingresso_pism"):
            op_ingresso.append("PISM")
        if tem_coluna("tipo_ingresso_sisu"):
            op_ingresso.append("SISU/ENEM")
        if tem_coluna("tipo_ingresso_vestibular"):
            op_ingresso.append("Vestibular")
        if tem_coluna("tipo_ingresso_outros"):
            op_ingresso.append("Outros")
        tipo_ingresso = c2.selectbox("Tipo de ingresso", op_ingresso or ["PISM"])

        op_genero = []
        if tem_coluna("genero_masculino"):
            op_genero.append("Masculino")
        if tem_coluna("genero_feminino"):
            op_genero.append("Feminino")
        if tem_coluna("genero_outros"):
            op_genero.append("Outros")
        genero = c3.selectbox("Gênero", op_genero or ["Masculino"])

        st.markdown("#### Cotas")
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        cota_racial = cc1.checkbox("Racial", disabled=not tem_coluna("cota_racial"))
        cota_ampla = cc2.checkbox("Ampla concorrência", disabled=not tem_coluna("cota_ampla_concorrencia"))
        cota_escola = cc3.checkbox("Escola pública", disabled=not tem_coluna("cota_escola_publica"))
        cota_renda = cc4.checkbox("Renda", disabled=not tem_coluna("cota_renda"))
        cota_pcd = cc5.checkbox("PCD", disabled=not tem_coluna("cota_pcd"))

        st.markdown("#### Etnia")
        op_etnia = ["Branca", "Preta", "Parda", "Outra"]
        etnia = st.radio("Selecione uma etnia", op_etnia, horizontal=True)
        st.caption("Todas as opções são exibidas na interface. Se alguma categoria não fizer parte do modelo deste curso/período, ela simplesmente não será usada no cálculo.")

        st.markdown(f"#### Dados por período (até o {periodo_max}º)")
        per_vals = {}

        periodos_validos = [pp for pp in periodos_detectados if pp <= periodo_max]
        primeiro_periodo = periodos_validos[0] if periodos_validos else None

        for p in periodos_validos:
            with st.expander(f"{p}º período", expanded=(p == primeiro_periodo)):
                a, b = st.columns(2)

                nomes = {
                    "ap": nome_coluna(f"periodo_{p}_disciplinas_aprovadas"),
                    "rep": nome_coluna(f"periodo_{p}_disciplinas_reprovadas"),
                    "ri": nome_coluna(f"periodo_{p}_disciplinas_ri"),
                    "tr": nome_coluna(f"periodo_{p}_disciplinas_trancadas"),
                    "out": nome_coluna(f"periodo_{p}_disciplinas_outros_status"),
                    "br": nome_coluna(f"periodo_{p}_bolsa_remunerada"),
                    "bnr": nome_coluna(f"periodo_{p}_bolsa_n_remunerada"),
                    "ae": nome_coluna(f"periodo_{p}_ae"),
                }

                per_vals[nomes["ap"]] = a.number_input(
                    "Aprovadas", min_value=0, step=1, value=valor_mediano(nomes["ap"]), key=f"ap_{p}"
                )
                per_vals[nomes["rep"]] = a.number_input(
                    "Reprovadas", min_value=0, step=1, value=valor_mediano(nomes["rep"]), key=f"rep_{p}"
                )
                per_vals[nomes["ri"]] = a.number_input(
                    "RI", min_value=0, step=1, value=valor_mediano(nomes["ri"]), key=f"ri_{p}"
                )
                per_vals[nomes["tr"]] = a.number_input(
                    "Trancadas", min_value=0, step=1, value=valor_mediano(nomes["tr"]), key=f"tr_{p}"
                )

                per_vals[nomes["out"]] = b.number_input(
                    "Outros status", min_value=0, step=1, value=valor_mediano(nomes["out"]), key=f"out_{p}"
                )
                per_vals[nomes["br"]] = b.selectbox("Bolsa remunerada", ["Não", "Sim"], key=f"br_{p}")
                per_vals[nomes["bnr"]] = b.selectbox("Bolsa não remunerada", ["Não", "Sim"], key=f"bnr_{p}")
                per_vals[nomes["ae"]] = b.selectbox("Assistência estudantil", ["Não", "Sim"], key=f"ae_{p}")

        submitted = st.form_submit_button("Calcular risco")

    if submitted:
        valores = []

        for col in cols_pred:
            col_l = col.lower()
            val = 0

            if col_l == "nota_ingresso":
                val = float(nota_ingresso)
            elif col_l == "tipo_ingresso_pism":
                val = 1 if tipo_ingresso == "PISM" else 0
            elif col_l == "tipo_ingresso_sisu":
                val = 1 if tipo_ingresso == "SISU/ENEM" else 0
            elif col_l == "tipo_ingresso_vestibular":
                val = 1 if tipo_ingresso == "Vestibular" else 0
            elif col_l == "tipo_ingresso_outros":
                val = 1 if tipo_ingresso == "Outros" else 0
            elif col_l == "genero_masculino":
                val = 1 if genero == "Masculino" else 0
            elif col_l == "genero_feminino":
                val = 1 if genero == "Feminino" else 0
            elif col_l == "genero_outros":
                val = 1 if genero == "Outros" else 0
            elif col_l == "cota_racial":
                val = int(cota_racial)
            elif col_l == "cota_ampla_concorrencia":
                val = int(cota_ampla)
            elif col_l == "cota_escola_publica":
                val = int(cota_escola)
            elif col_l == "cota_renda":
                val = int(cota_renda)
            elif col_l == "cota_pcd":
                val = int(cota_pcd)
            elif col_l == "etnia_branca":
                val = 1 if etnia == "Branca" else 0
            elif col_l == "etnia_preta":
                val = 1 if etnia == "Preta" else 0
            elif col_l == "etnia_parda":
                val = 1 if etnia == "Parda" else 0
            elif col_l == "etnia_outra":
                val = 1 if etnia == "Outra" else 0
            elif re.match(r"periodo_(\d+)_", col_l):
                v = per_vals.get(col, 0)
                if isinstance(v, str):
                    val = 1 if v == "Sim" else 0
                else:
                    val = int(v)

            valores.append(val)

        X_new = pd.DataFrame([valores], columns=cols_pred)
        proba = float(modelo_comite.predict_proba(X_new)[0, 1])
        yhat = int(proba >= modelo_comite.threshold_)

        m1, m2, m3 = st.columns(3)
        m1.metric("Risco estimado de evasão", f"{proba * 100:.1f}%")
        m2.metric("Classe prevista", "Evasão" if yhat == 1 else "Permanência")
        m3.metric("Threshold do comitê", f"{modelo_comite.threshold_:.3f}")

        st.markdown("#### Explicação da previsão")

        try:
            X_bg_local = X_shap[cols_pred].copy()
            explanation, impactos = explicacao_local_comite(modelo_comite, X_bg_local, X_new[cols_pred], proba)

            fig_w = plt.figure(figsize=(8.2, 4.8))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig_w)
            plt.close(fig_w)

            with st.expander("Principais fatores desta previsão", expanded=True):
                top_locais = impactos.head(6)
                for row in top_locais.itertuples():
                    direcao = "aumentou o risco de evasão" if row.shap > 0 else "reduziu o risco de evasão"
                    st.markdown(f"- **{row.feature_label}** (valor = {row.valor}): {direcao}.")
        except Exception as e:
            st.warning(f"Não foi possível gerar a explicação SHAP local: {e}")


# ============================================================
# Aba 4 - detalhes técnicos
# ============================================================
with abas[3]:
    st.subheader("Detalhes técnicos")

    with st.expander("Desempenho agregado"):
        if not mostrar.empty:
            st.dataframe(mostrar[cols_keep], use_container_width=True)
        else:
            st.info("Tabela agregada do comitê não encontrada.")

    with st.expander("Melhor execução"):
        if meta_comite:
            st.write(f"**Repetição:** {meta_comite.get('melhor_repeticao', '-')}")
            st.write(f"**Threshold:** {meta_comite.get('threshold', np.nan):.3f}")
            st.write(f"**F1_pos:** {meta_comite.get('f1_pos', np.nan):.3f}")
            st.write(f"**ROC-AUC:** {meta_comite.get('roc_auc', np.nan):.3f}")
            st.write(f"**Average Precision:** {meta_comite.get('avg_precision', np.nan):.3f}")
            st.write(f"**Brier calibrado:** {meta_comite.get('brier_cal', np.nan):.3f}")
            if "pesos" in meta_comite:
                pesos = meta_comite["pesos"]
                st.write(f"**Pesos do comitê:** XGB={pesos[0]:.3f} • LGBM={pesos[1]:.3f} • CAT={pesos[2]:.3f}")
        else:
            st.info("Metadados da melhor execução não encontrados.")

    with st.expander("Figuras da melhor execução"):
        opcoes_fig = {
            "Matriz de confusão": figuras_best["confusao"],
            "Curva ROC": figuras_best["roc"],
            "Curva Precision-Recall": figuras_best["pr"],
            "Curva de calibração": figuras_best["calibracao"],
        }
        opcoes_validas = {k: v for k, v in opcoes_fig.items() if v is not None}

        if opcoes_validas:
            fig_sel = st.selectbox("Figura", list(opcoes_validas.keys()))
            st.image(str(opcoes_validas[fig_sel]), caption=fig_sel)
        else:
            st.info("As figuras da melhor execução não foram localizadas.")

    with st.expander("Robustez das repetições"):
        if metricas_comite.empty:
            st.info("Não há métricas por repetição disponíveis para o comitê.")
        else:
            metrica_plot = st.selectbox(
                "Métrica",
                ["f1_pos", "roc_auc", "avg_precision", "brier_cal", "threshold"],
                format_func=lambda x: {
                    "f1_pos": "F1 classe positiva",
                    "roc_auc": "ROC-AUC",
                    "avg_precision": "Average Precision",
                    "brier_cal": "Brier calibrado",
                    "threshold": "Threshold",
                }[x],
                key="metrica_tecnica",
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(metricas_comite["repeticao"], metricas_comite[metrica_plot], marker="o")
            ax.set_xlabel("Repetição")
            ax.set_ylabel(metrica_plot)
            ax.set_title(f"{metrica_plot} ao longo das repetições")
            st.pyplot(fig)
            plt.close(fig)

            cols_rep = [
                "repeticao",
                "f1_pos",
                "f1_pos_val",
                "roc_auc",
                "avg_precision",
                "brier_cal",
                "threshold",
                "w_xgb",
                "w_lgbm",
                "w_cat",
            ]
            cols_rep = [c for c in cols_rep if c in metricas_comite.columns]
            st.dataframe(metricas_comite[cols_rep], use_container_width=True)
