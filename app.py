import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="UCIC Loan Risk Dashboard By Chandana Reddy",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    .risk-high   { background:#fde8e8; color:#991b1b; padding:4px 12px; border-radius:6px; font-weight:600; font-size:13px; }
    .risk-medium { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:6px; font-weight:600; font-size:13px; }
    .risk-low    { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:6px; font-weight:600; font-size:13px; }
    .metric-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; }
    .metric-val  { font-size:28px; font-weight:700; color:#1e293b; }
    .metric-lbl  { font-size:13px; color:#64748b; margin-top:4px; }
    .section-title { font-size:18px; font-weight:600; color:#1e293b; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model       = joblib.load("model.pkl")
    sector_enc  = joblib.load("sector_encoder.pkl")
    investor_enc= joblib.load("investor_encoder.pkl")
    risk_enc    = joblib.load("risk_encoder.pkl")
    return model, sector_enc, investor_enc, risk_enc

@st.cache_data
def load_data():
    df = pd.read_csv("loan_portfolio.csv")
    return df

model, sector_enc, investor_enc, risk_enc = load_artifacts()
df = load_data()

SECTORS   = list(sector_enc.classes_)
INVESTORS = list(investor_enc.classes_)
RISK_ORDER= ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}

# ── Sidebar filters ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.ucinclusive.in/wp-content/uploads/2019/09/logo.png", width=180)
    st.markdown("## UCIC Risk Dashboard")
    st.markdown("---")
    st.markdown("### Portfolio Filters")

    sel_sectors = st.multiselect("Sector", SECTORS, default=SECTORS)
    sel_risks   = st.multiselect("Risk Level", RISK_ORDER, default=RISK_ORDER)
    loan_range  = st.slider("Loan Amount (₹ Lakh)", 50, 2000, (50, 2000), step=50)

    st.markdown("---")
    st.caption("UC Inclusive Credit Pvt. Ltd. · Prototype")

fdf = df[
    df["sector"].isin(sel_sectors) &
    df["risk_label"].isin(sel_risks) &
    df["loan_amount_lakh"].between(loan_range[0], loan_range[1])
].copy()

# ── Page tabs ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🚨 Risk Flags", "🔍 Assess New Loan"])


# ══════════════════════════════════════════
# TAB 1 — Portfolio Overview
# ══════════════════════════════════════════
with tab1:
    st.markdown("### Portfolio at a Glance")

    total   = len(fdf)
    high_r  = len(fdf[fdf["risk_label"] == "High"])
    avg_loan= fdf["loan_amount_lakh"].mean()
    avg_irr = fdf["irr_percent"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total}</div><div class="metric-lbl">Total Loans</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ef4444">{high_r}</div><div class="metric-lbl">High Risk Loans</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">₹{avg_loan:.0f}L</div><div class="metric-lbl">Avg Loan Size</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_irr:.1f}%</div><div class="metric-lbl">Avg IRR</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = fdf["risk_label"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig_risk = px.bar(
            risk_counts, x="Risk", y="Count",
            color="Risk",
            color_discrete_map=RISK_COLORS,
            text="Count"
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(
            showlegend=False, plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            xaxis_title="", yaxis_title="No. of Loans"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Sector Allocation</div>', unsafe_allow_html=True)
        sec_counts = fdf["sector"].value_counts().reset_index()
        sec_counts.columns = ["Sector", "Count"]
        fig_pie = px.pie(
            sec_counts, names="Sector", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Risk by Sector</div>', unsafe_allow_html=True)
        heat = fdf.groupby(["sector", "risk_label"]).size().unstack(fill_value=0)
        heat = heat.reindex(columns=RISK_ORDER, fill_value=0)
        fig_heat = px.imshow(
            heat,
            color_continuous_scale=["#d1fae5", "#fef3c7", "#fde8e8"],
            text_auto=True, aspect="auto"
        )
        fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">Loan Size vs Promoter Score</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            fdf, x="promoter_score", y="loan_amount_lakh",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            size="repayment_track",
            hover_data=["loan_id", "sector", "irr_percent"],
            labels={"promoter_score": "Promoter Score", "loan_amount_lakh": "Loan Amount (₹L)"}
        )
        fig_sc.update_layout(
            plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            legend_title="Risk"
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════
# TAB 2 — Risk Flag Table
# ══════════════════════════════════════════
with tab2:
    st.markdown("### Risk-Flagged Loan Portfolio")

    sort_col = st.selectbox("Sort by", ["risk_label", "loan_amount_lakh", "promoter_score", "repayment_track"], index=0)
    show_df  = fdf.sort_values(
        sort_col,
        key=lambda x: x.map({"High": 0, "Medium": 1, "Low": 2}) if sort_col == "risk_label" else x
    ).reset_index(drop=True)

    def color_risk(val):
        colors = {"High": "background-color:#fde8e8;color:#991b1b;font-weight:600",
                  "Medium": "background-color:#fef3c7;color:#92400e;font-weight:600",
                  "Low": "background-color:#d1fae5;color:#065f46;font-weight:600"}
        return colors.get(val, "")

    def color_score(val):
        if val >= 75: return "color:#065f46;font-weight:600"
        if val >= 55: return "color:#92400e"
        return "color:#991b1b;font-weight:600"

    display_cols = ["loan_id", "sector", "loan_amount_lakh", "tenure_months",
                    "promoter_score", "repayment_track", "irr_percent",
                    "investor_backing", "years_in_operation", "risk_label"]

    styled = (
        show_df[display_cols]
        .rename(columns={
            "loan_id": "Loan ID", "sector": "Sector",
            "loan_amount_lakh": "Amount (₹L)", "tenure_months": "Tenure",
            "promoter_score": "Promoter", "repayment_track": "Repayment",
            "irr_percent": "IRR%", "investor_backing": "Investor",
            "years_in_operation": "Yrs Ops", "risk_label": "Risk"
        })
        .style
        .map(color_risk, subset=["Risk"])
        .map(color_score, subset=["Promoter", "Repayment"])
        .format({"Amount (₹L)": "{:.0f}", "IRR%": "{:.1f}", "Promoter": "{:.1f}", "Repayment": "{:.1f}", "Yrs Ops": "{:.1f}"})
    )

    st.dataframe(styled, use_container_width=True, height=500)

    high_loans = fdf[fdf["risk_label"] == "High"]
    if len(high_loans) > 0:
        st.markdown(f"#### ⚠️ {len(high_loans)} High-Risk Loans Flagged")
        for _, row in high_loans.iterrows():
            reasons = []
            if row["promoter_score"] < 55:  reasons.append(f"low promoter score ({row['promoter_score']:.0f})")
            if row["repayment_track"] < 60:  reasons.append(f"poor repayment track ({row['repayment_track']:.0f})")
            if row["investor_backing"] == "Weak": reasons.append("weak investor backing")
            if row["years_in_operation"] < 2:    reasons.append(f"early stage ({row['years_in_operation']:.1f} yrs)")
            reason_str = " · ".join(reasons) if reasons else "multiple risk factors"
            st.warning(f"**{row['loan_id']}** | {row['sector']} | ₹{row['loan_amount_lakh']:.0f}L → {reason_str}")


# ══════════════════════════════════════════
# TAB 3 — Predict New Loan
# ══════════════════════════════════════════
with tab3:
    st.markdown("### Assess a New Loan Application")
    st.caption("Enter the borrower's profile below to get an instant risk classification.")

    col1, col2, col3 = st.columns(3)

    with col1:
        sector          = st.selectbox("Impact Sector", SECTORS)
        loan_amount     = st.number_input("Loan Amount (₹ Lakh)", 50, 5000, 300, step=50)
        tenure          = st.selectbox("Tenure (months)", [12, 18, 24, 36, 48, 60], index=2)

    with col2:
        promoter_score  = st.slider("Promoter / Management Score", 0, 100, 65)
        repayment_track = st.slider("Repayment Track Record", 0, 100, 70)
        irr             = st.number_input("Expected IRR (%)", 8.0, 30.0, 16.0, step=0.5)

    with col3:
        investor_backing = st.selectbox("Investor Backing", INVESTORS)
        years_ops        = st.number_input("Years in Operation", 0.5, 30.0, 4.0, step=0.5)
        st.markdown("<br>", unsafe_allow_html=True)
        assess_btn = st.button("🔍 Assess Risk", use_container_width=True, type="primary")

    st.markdown("---")

    if assess_btn:
        sec_enc  = sector_enc.transform([sector])[0]
        inv_enc  = investor_enc.transform([investor_backing])[0]

        features = np.array([[sec_enc, loan_amount, tenure, promoter_score,
                               repayment_track, irr, inv_enc, years_ops]])

        pred_enc   = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        pred_label = risk_enc.inverse_transform([pred_enc])[0]
        confidence = max(pred_proba) * 100

        risk_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}[pred_label]
        risk_bg    = {"High": "#fde8e8", "Medium": "#fef3c7", "Low": "#d1fae5"}[pred_label]
        risk_icon  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[pred_label]

        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            st.markdown(f"""
            <div style="background:{risk_bg};border:2px solid {risk_color};border-radius:12px;padding:24px;text-align:center;">
                <div style="font-size:36px">{risk_icon}</div>
                <div style="font-size:28px;font-weight:700;color:{risk_color};margin:8px 0">{pred_label} Risk</div>
                <div style="font-size:14px;color:#64748b">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 40],  "color": "#f1f5f9"},
                        {"range": [40, 70], "color": "#e2e8f0"},
                        {"range": [70, 100],"color": "#cbd5e1"},
                    ]
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=20, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r3:
            st.markdown("**Probability breakdown**")
            for label in risk_enc.classes_:
                idx  = list(risk_enc.classes_).index(label)
                prob = pred_proba[idx] * 100
                col  = RISK_COLORS[label]
                st.markdown(f"""
                <div style="margin-bottom:8px">
                    <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
                        <span style="color:{col};font-weight:600">{label}</span>
                        <span>{prob:.1f}%</span>
                    </div>
                    <div style="background:#e2e8f0;border-radius:4px;height:8px">
                        <div style="background:{col};width:{prob}%;height:8px;border-radius:4px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**What's driving this assessment:**")
        flags = []
        if promoter_score >= 75:  flags.append(("✅", "Strong promoter/management quality"))
        elif promoter_score < 55: flags.append(("⚠️", f"Weak promoter score ({promoter_score}) — key risk driver"))
        else:                     flags.append(("ℹ️", f"Moderate promoter score ({promoter_score})"))

        if repayment_track >= 80: flags.append(("✅", "Excellent repayment track record"))
        elif repayment_track < 60:flags.append(("⚠️", f"Poor repayment history ({repayment_track}) — highest risk signal"))
        else:                     flags.append(("ℹ️", f"Average repayment track ({repayment_track})"))

        if investor_backing == "Strong": flags.append(("✅", "Backed by strong institutional investors"))
        elif investor_backing == "Weak": flags.append(("⚠️", "Weak investor backing — limited validation"))

        if years_ops < 2: flags.append(("⚠️", f"Early-stage enterprise ({years_ops} yrs) — limited operating history"))
        elif years_ops >= 5: flags.append(("✅", f"Established enterprise ({years_ops:.0f} yrs in operation)"))

        for icon, msg in flags:
            st.markdown(f"{icon} {msg}")
