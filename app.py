import sys
import os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from muraqib.data_loader import load_data, get_feature_display_names
from muraqib.model import get_model, predict
from muraqib.i18n import ui, translate_activity, translate_contractor, translate_complexity

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Muraqib | مراقب",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session State ───────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"

lang = st.session_state.lang
is_ar = lang == "ar"
dir_attr = "rtl" if is_ar else "ltr"
text_align = "right" if is_ar else "left"
font_family = "'Tajawal', 'Inter', sans-serif" if is_ar else "'Inter', 'Tajawal', sans-serif"

# ─── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Tajawal:wght@300;400;500;700;800&display=swap" rel="stylesheet">
    <style>
        /* Reset & base */
        html, body, [class*="css"] {{
            font-family: {font_family};
            direction: {dir_attr};
        }}
        .stApp {{
            background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 40%, #1a0a2e 100%);
            min-height: 100vh;
        }}
        /* Hide default streamlit elements */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding: 1.5rem 2rem; max-width: 1400px; }}

        /* Header */
        .muraqib-header {{
            background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 100%);
            border: 1px solid rgba(99,102,241,0.3);
            border-radius: 20px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(10px);
        }}
        .muraqib-title {{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }}
        .muraqib-subtitle {{
            color: rgba(148,163,184,0.85);
            font-size: 0.95rem;
            margin-top: 0.4rem;
            font-weight: 400;
        }}
        .muraqib-badge {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            display: inline-block;
        }}

        /* Metric cards */
        .metric-card {{
            background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,27,75,0.7) 100%);
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            text-align: {text_align};
            transition: all 0.3s ease;
            backdrop-filter: blur(8px);
        }}
        .metric-card:hover {{
            border-color: rgba(99,102,241,0.55);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(99,102,241,0.15);
        }}
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }}
        .metric-label {{
            color: rgba(148,163,184,0.8);
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 0.4rem;
        }}

        /* Section heading */
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #f1f5f9;
            margin: 1.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(99,102,241,0.3);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background: rgba(15,23,42,0.6);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
            border: 1px solid rgba(99,102,241,0.2);
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 10px;
            color: rgba(148,163,184,0.8);
            font-weight: 500;
            padding: 0.6rem 1.4rem;
            font-family: {font_family};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important;
        }}

        /* Result cards */
        .result-high {{
            background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
            border: 1px solid rgba(239,68,68,0.4);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
        }}
        .result-low {{
            background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.08));
            border: 1px solid rgba(34,197,94,0.4);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
        }}
        .result-text {{
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }}
        .result-prob {{
            font-size: 1rem;
            color: rgba(148,163,184,0.9);
        }}

        /* Form inputs */
        .stSelectbox > div > div, .stSlider, .stNumberInput {{
            background: rgba(15,23,42,0.7) !important;
            border-radius: 10px !important;
        }}
        label {{
            color: #cbd5e1 !important;
            font-weight: 500 !important;
            font-family: {font_family} !important;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 2rem;
            font-size: 1rem;
            font-weight: 700;
            width: 100%;
            transition: all 0.3s ease;
            font-family: {font_family};
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99,102,241,0.4);
        }}

        /* Dataframe */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(99,102,241,0.2);
        }}

        /* Chart background */
        .js-plotly-plot .plotly .main-svg {{
            border-radius: 12px;
        }}

        /* Divider */
        hr {{
            border-color: rgba(99,102,241,0.2);
            margin: 1.5rem 0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Load Data & Model ──────────────────────────────────────────────────────────
@st.cache_data
def cached_load():
    return load_data()

df = cached_load()

@st.cache_resource
def cached_model(df_hash):
    return get_model(df)

model, model_acc = cached_model(len(df))

# ─── Header ─────────────────────────────────────────────────────────────────────
col_title, col_lang = st.columns([5, 1])
with col_title:
    st.markdown(
        f"""
        <div class="muraqib-header">
            <div>
                <span class="muraqib-badge">🏗️ AI · Construction</span>
                <div class="muraqib-title">{ui('app_title', lang)}</div>
                <div class="muraqib-subtitle">{ui('app_subtitle', lang)}</div>
            </div>
            <div style="text-align:right; color:rgba(148,163,184,0.6); font-size:0.85rem;">
                Model Accuracy<br/>
                <span style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#34d399,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{model_acc}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_lang:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    if st.button(ui("lang_toggle", lang), key="lang_btn"):
        st.session_state.lang = "ar" if lang == "en" else "en"
        st.rerun()

# ─── KPI Metrics ────────────────────────────────────────────────────────────────
delay_rate = round(df["is_delayed"].mean() * 100, 1)
high_complex = (df["Complexity Level"] == "High").sum()
n_contractors = df["Contractor Name"].nunique()

c1, c2, c3, c4 = st.columns(4)
for col, value, label in [
    (c1, len(df), ui("total_activities", lang)),
    (c2, high_complex, ui("high_risk", lang)),
    (c3, f"{delay_rate}%", ui("delay_rate", lang)),
    (c4, n_contractors, ui("unique_contractors", lang)),
]:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    ui("tab_overview", lang),
    ui("tab_predict", lang),
    ui("tab_analytics", lang),
])

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family=font_family),
    xaxis=dict(gridcolor="rgba(99,102,241,0.1)", zeroline=False),
    yaxis=dict(gridcolor="rgba(99,102,241,0.1)", zeroline=False),
)


# ══════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<div class='section-title'>{ui('table_title', lang)}</div>", unsafe_allow_html=True)

    # Build display dataframe with translated values
    display_df = df[["Activity Name", "Expected Start Date", "Contractor Name",
                      "Complexity Level", "supply_delay_days", "subcontractor_performance",
                      "weather_risk", "labor_availability", "is_delayed"]].copy()

    display_df["Activity Name"] = display_df["Activity Name"].apply(lambda x: translate_activity(x, lang))
    display_df["Contractor Name"] = display_df["Contractor Name"].apply(lambda x: translate_contractor(x, lang))
    display_df["Complexity Level"] = display_df["Complexity Level"].apply(lambda x: translate_complexity(x, lang))
    display_df["Expected Start Date"] = display_df["Expected Start Date"].dt.strftime("%Y-%m-%d")

    col_names = {
        "Activity Name": ui("select_activity", lang),
        "Expected Start Date": ui("select_activity", lang).replace(ui("select_activity", lang), "Date" if lang == "en" else "التاريخ"),
        "Contractor Name": ui("select_contractor", lang),
        "Complexity Level": ui("select_complexity", lang),
        "supply_delay_days": ui("supply_delay", lang),
        "subcontractor_performance": ui("subcontractor_perf", lang),
        "weather_risk": ui("weather_risk", lang),
        "labor_availability": ui("labor_availability", lang),
        "is_delayed": "⚠️ Delayed" if lang == "en" else "⚠️ متأخر",
    }
    display_df.rename(columns=col_names, inplace=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=360,
        hide_index=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        # Activity distribution by complexity
        comp_counts = df["Complexity Level"].value_counts().reset_index()
        comp_counts.columns = ["Complexity", "Count"]
        comp_counts["Complexity_label"] = comp_counts["Complexity"].apply(lambda x: translate_complexity(x, lang))
        fig = px.pie(
            comp_counts,
            names="Complexity_label",
            values="Count",
            hole=0.55,
            color_discrete_sequence=["#6366f1", "#f472b6", "#34d399"],
            title=ui("activity_distribution", lang),
        )
        fig.update_layout(**PLOT_THEME, title_font_size=14, margin=dict(t=40, b=10))
        fig.update_traces(textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        # Delay count by complexity
        delay_comp = df.groupby("Complexity Level")["is_delayed"].mean().mul(100).round(1).reset_index()
        delay_comp.columns = ["Complexity", "Delay Rate"]
        delay_comp["Complexity_label"] = delay_comp["Complexity"].apply(lambda x: translate_complexity(x, lang))
        fig2 = px.bar(
            delay_comp,
            x="Complexity_label",
            y="Delay Rate",
            text="Delay Rate",
            color="Delay Rate",
            color_continuous_scale=["#34d399", "#f472b6", "#ef4444"],
            title=ui("delay_by_complexity", lang),
        )
        fig2.update_traces(texttemplate="%{text}%", textposition="outside", marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(**PLOT_THEME, title_font_size=14, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"<div class='section-title'>{ui('predict_title', lang)}</div>", unsafe_allow_html=True)

    arabic_activities = sorted(df["Activity Name"].unique().tolist())
    arabic_contractors = sorted(df["Contractor Name"].unique().tolist())

    complexity_options_raw = ["Low", "Medium", "High"]
    complexity_options_display = [translate_complexity(c, lang) for c in complexity_options_raw]
    weather_options_raw = ["Low", "Medium", "High"]
    weather_options_display = [
        ui("weather_low", lang),
        ui("weather_medium", lang),
        ui("weather_high", lang),
    ]

    activity_display = [translate_activity(a, lang) for a in arabic_activities]
    contractor_display = [translate_contractor(c, lang) for c in arabic_contractors]

    f1, f2 = st.columns(2)
    with f1:
        act_idx = st.selectbox(ui("select_activity", lang), range(len(activity_display)),
                                format_func=lambda i: activity_display[i], key="act")
        comp_idx = st.selectbox(ui("select_complexity", lang), range(len(complexity_options_display)),
                                 format_func=lambda i: complexity_options_display[i], key="comp")
        supply_delay = st.slider(ui("supply_delay", lang), 0, 30, 5, key="supply")
        weather_idx = st.selectbox(ui("weather_risk", lang), range(len(weather_options_display)),
                                    format_func=lambda i: weather_options_display[i], key="weather")

    with f2:
        cont_idx = st.selectbox(ui("select_contractor", lang), range(len(contractor_display)),
                                 format_func=lambda i: contractor_display[i], key="cont")
        subcontractor_perf = st.slider(ui("subcontractor_perf", lang), 1.0, 10.0, 7.0, step=0.5, key="subperf")
        labor_avail = st.slider(ui("labor_availability", lang), 50, 100, 80, key="labor")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button(ui("predict_btn", lang), key="predict_btn")

    if predict_btn:
        result = predict(
            model,
            complexity_enc=comp_idx,
            supply_delay_days=supply_delay,
            subcontractor_performance=subcontractor_perf,
            weather_enc=weather_idx,
            labor_availability=float(labor_avail),
        )

        res_cls = "result-high" if result["is_delayed"] else "result-low"
        res_text = ui("result_high", lang) if result["is_delayed"] else ui("result_low", lang)
        prob = result["delay_probability"]

        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown(
                f"""
                <div class="{res_cls}">
                    <div class="result-text">{res_text}</div>
                    <div class="result-prob">{ui('probability', lang)}: <strong>{prob}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            # Gauge chart
            gauge_color = "#ef4444" if result["is_delayed"] else "#34d399"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"color": gauge_color, "size": 36}},
                title={"text": ui("risk_gauge", lang), "font": {"color": "#94a3b8", "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "rgba(15,23,42,0.6)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(52,211,153,0.1)"},
                        {"range": [50, 75], "color": "rgba(251,191,36,0.1)"},
                        {"range": [75, 100], "color": "rgba(239,68,68,0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": prob,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family=font_family),
                height=230,
                margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance bar
        st.markdown(f"<div class='section-title' style='margin-top:1rem'>{ui('feature_importance', lang)}</div>",
                    unsafe_allow_html=True)

        feat_names = get_feature_display_names(lang)
        importances = result["feature_importances"]
        fi_df = pd.DataFrame({
            "Feature": [feat_names.get(k, k) for k in importances],
            "Importance": list(importances.values()),
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#6366f1", "#a78bfa", "#f472b6"],
        )
        fig_fi.update_coloraxes(showscale=False)
        fig_fi.update_layout(**PLOT_THEME, height=220, margin=dict(t=10, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"<div class='section-title'>{ui('analytics_title', lang)}</div>", unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        # Delay rate by contractor
        contractor_delay = (
            df.groupby("Contractor Name")["is_delayed"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .sort_values("is_delayed", ascending=False)
        )
        contractor_delay["Contractor_display"] = contractor_delay["Contractor Name"].apply(
            lambda x: translate_contractor(x, lang)
        )
        fig_c = px.bar(
            contractor_delay,
            x="is_delayed",
            y="Contractor_display",
            orientation="h",
            text="is_delayed",
            color="is_delayed",
            color_continuous_scale=["#34d399", "#f472b6", "#ef4444"],
            title=ui("delay_by_contractor", lang),
        )
        fig_c.update_traces(texttemplate="%{text}%", textposition="outside", marker_line_width=0)
        fig_c.update_coloraxes(showscale=False)
        fig_c.update_layout(**PLOT_THEME, title_font_size=14, height=420,
                             margin=dict(t=40, b=10, l=10, r=40))
        st.plotly_chart(fig_c, use_container_width=True)

    with a2:
        # Supply delay distribution
        fig_sd = px.histogram(
            df,
            x="supply_delay_days",
            color="is_delayed",
            barmode="overlay",
            color_discrete_map={0: "#34d399", 1: "#ef4444"},
            title=ui("supply_delay", lang) + " Distribution" if lang == "en" else "توزيع " + ui("supply_delay", lang),
            labels={"supply_delay_days": ui("supply_delay", lang), "is_delayed": "Delayed"},
            nbins=15,
        )
        fig_sd.update_layout(**PLOT_THEME, title_font_size=14, height=200, margin=dict(t=40, b=10))
        st.plotly_chart(fig_sd, use_container_width=True)

        # Labor vs subcontractor scatter
        fig_sc = px.scatter(
            df,
            x="labor_availability",
            y="subcontractor_performance",
            color="is_delayed",
            color_discrete_map={0: "#34d399", 1: "#ef4444"},
            size="supply_delay_days",
            size_max=18,
            opacity=0.75,
            title=ui("labor_availability", lang) + " vs " + ui("subcontractor_perf", lang),
            labels={
                "labor_availability": ui("labor_availability", lang),
                "subcontractor_performance": ui("subcontractor_perf", lang),
            },
        )
        fig_sc.update_layout(**PLOT_THEME, title_font_size=14, height=200, margin=dict(t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Monthly heatmap
    st.markdown(f"<div class='section-title'>{ui('monthly_heatmap', lang)}</div>", unsafe_allow_html=True)

    month_comp = df.pivot_table(
        index="Complexity Level", columns="start_month", values="is_delayed", aggfunc="mean"
    ).fillna(0).mul(100).round(1)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if lang == "ar":
        month_labels = ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
                        "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]

    fig_hm = go.Figure(go.Heatmap(
        z=month_comp.values,
        x=[month_labels[m - 1] for m in month_comp.columns],
        y=[translate_complexity(c, lang) for c in month_comp.index],
        colorscale=[[0, "#1a0a2e"], [0.5, "#6366f1"], [1, "#ef4444"]],
        text=month_comp.values,
        texttemplate="%{text}%",
        textfont={"size": 11, "color": "white"},
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8")),
    ))
    fig_hm.update_layout(
        **PLOT_THEME,
        height=220,
        margin=dict(t=10, b=10),
    )
    fig_hm.update_xaxes(side="bottom")
    st.plotly_chart(fig_hm, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1rem;color:rgba(100,116,139,0.6);font-size:0.8rem;">
        🏗️ Muraqib | مراقب &nbsp;·&nbsp; Powered by RandomForest &nbsp;·&nbsp; Saudi Construction Risk Analytics
    </div>
    """,
    unsafe_allow_html=True,
)
