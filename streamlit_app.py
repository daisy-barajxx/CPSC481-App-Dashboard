import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import requests

st.set_page_config(
    page_title="Incribo Synthetic Cyber Attacks Explorer",
    layout="wide",
)

# style 
st.markdown(
    """
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2.0rem;}
div[data-testid="stMetricValue"] {font-size: 2.1rem;}
div[data-testid="stMetricLabel"] {opacity: 0.8;}
h1, h2, h3 {letter-spacing: -0.02em;}
</style>
""",
    unsafe_allow_html=True,
)

# load and clean data
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "data" / "cybersecurity_attacks.csv"
    if not csv_path.exists():
        st.error("Could not find data/cybersecurity_attacks.csv")
        st.stop()

    df = pd.read_csv(csv_path)

    # keep only columns actually used in the dashboard
    keep = [
        "Timestamp",
        "Attack Type",
        "Protocol",
        "Traffic Type",
        "Severity Level",
        "Action Taken",
        "Network Segment",
        "Attack Signature",
        "Anomaly Scores",
        "Geo-location Data",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # ---- Parse time ----
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()
    df["Date"] = df["Timestamp"].dt.date

    # ---- Numeric cleanup ----
    for c in ["Packet Length", "Anomaly Scores"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Parse Geo-location Data into state (India-only in this dataset) ----
    if "Geo-location Data" in df.columns:
        geo = df["Geo-location Data"].fillna("").astype(str)
        parts = geo.str.split(",", n=1, expand=True)
        df["Geo_State"] = (parts[1].str.strip() if parts.shape[1] > 1 else np.nan)
        df["Geo_State"] = df["Geo_State"].astype(str).str.strip()
        df.loc[df["Geo_State"].isin(["nan", "None", "null", ""]), "Geo_State"] = np.nan
        df["Geo_State"] = df["Geo_State"].str.title()
    else:
        df["Geo_State"] = np.nan

    # ---- Drop rows missing essential categorical fields used in visualizations ----
    essential = [c for c in ["Attack Type", "Protocol", "Severity Level", "Action Taken"] if c in df.columns]
    df = df.dropna(subset=essential).copy()

    # Normalize empty strings
    for c in essential + ["Traffic Type", "Network Segment", "Attack Signature"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["", "nan", "None", "null"]), c] = "unknown"

    return df


@st.cache_data
def load_india_geojson():
    # Prefer local file for Streamlit cloud reliability
    local_path = Path(__file__).parent / "data" / "india_states.geojson"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # fallback fetch
    urls = [
        "https://raw.githubusercontent.com/Anujarya300/bharat-geojson/master/india_states.geojson",
        "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
        "https://raw.githubusercontent.com/udit-001/india-maps-data/master/geojson/india_state.geojson",
    ]
    for url in urls: 
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
        except Exception:
            pass
    return None


def make_state_key(geojson_obj):
    if not geojson_obj:
        return None
    props = geojson_obj["features"][0].get("properties", {})
    for k in ["ST_NM", "st_nm", "STATE", "state", "NAME_1", "name", "STATE_NAME", "State_Name"]:
        if k in props:
            return k
    for k, v in props.items():
        if isinstance(v, str) and len(v) < 40:
            return k
    return None


def build_sankey(df: pd.DataFrame, stage1: str, stage2: str, stage3: str):
    d = df[[stage1, stage2, stage3]].dropna().copy()
    for c in [stage1, stage2, stage3]:
        d[c] = d[c].astype(str).str.strip()
        d.loc[d[c].isin(["", "nan", "None", "null"]), c] = "unknown"

    g12 = d.groupby([stage1, stage2]).size().reset_index(name="value")
    g23 = d.groupby([stage2, stage3]).size().reset_index(name="value")
    if g12.empty or g23.empty:
        return None

    nodes_1 = g12[stage1].astype(str).unique().tolist()
    nodes_2 = pd.unique(pd.concat([g12[stage2], g23[stage2]])).astype(str).tolist()
    nodes_3 = g23[stage3].astype(str).unique().tolist()

    node_labels = (
        [f"{stage1}: {x}" for x in nodes_1]
        + [f"{stage2}: {x}" for x in nodes_2]
        + [f"{stage3}: {x}" for x in nodes_3]
    )

    idx_1 = {x: i for i, x in enumerate(nodes_1)}
    idx_2 = {x: i + len(nodes_1) for i, x in enumerate(nodes_2)}
    idx_3 = {x: i + len(nodes_1) + len(nodes_2) for i, x in enumerate(nodes_3)}

    sources, targets, values = [], [], []

    for _, row in g12.iterrows():
        sources.append(idx_1[str(row[stage1])])
        targets.append(idx_2[str(row[stage2])])
        values.append(int(row["value"]))

    for _, row in g23.iterrows():
        sources.append(idx_2[str(row[stage2])])
        targets.append(idx_3[str(row[stage3])])
        values.append(int(row["value"]))

    fig = go.Figure(
        data=[go.Sankey(
            node=dict(label=node_labels, pad=10, thickness=16),
            link=dict(source=sources, target=targets, value=values)
        )]
    )
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# -----------------------------
# App
# -----------------------------
df = load_data()

st.title("India Cyber Attack Dataset")
st.markdown(
    """
### **Dataset:** 
This dataset is from Kraggle's Incribo Synthetic Cyber Attack Dataset from roughly 2020 to 2023. This dataset offers making various analytical tasks that helps access attacks by finding patterns in these realistic representations. The dataset consist of 25 attributes and 40,000 records. The attributes I am interested to dive deeper into are the following:
- Timestamp,
- Attack Type,
- Protocol,
- Traffic Type: what kind of behavior it is 
- Severity Level,
- Action Taken,
- Network Segment: where the traffic is allowed to exist
- Attack Signature,
- Anomaly Scores: behavior deviates from established normal
- Geo-location Data,



### **Question:** 
How do attacks evolve over time, which pathways dominate, and where do hotspots cluster? 


**Note:** The provided `Geo-location Data` is `City, State`, so the map is **India state-level**.

"""
)

st.divider()


min_d = df["Date"].min()
max_d = df["Date"].max()

st.markdown("**Date range**")
date_start, date_end = st.slider(
    label="",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),
    label_visibility="collapsed",
    key="global_date",
)

df_date = df[(df["Date"] >= date_start) & (df["Date"] <= date_end)].copy()

# attacks over time 

st.subheader("Attacks over time")

st.caption(
    "What to look for: shifts in volume (counts) or shifts in behavior (mean anomaly). "
    "If focus filters are set to All, the chart shows the top attack types for readability."
)

c2, c3, c4 = st.columns([1.2, 1.2, 1.0])

attack_focus = c2.selectbox(
    "Attack Type",
    options=["All"] + sorted(df_date["Attack Type"].unique().tolist()),
    index=0,
    key="ts_attack",
)

protocol_focus = c3.selectbox(
    "Protocol",
    options=["All"] + sorted(df_date["Protocol"].unique().tolist()),
    index=0,
    key="ts_protocol",
)

main_mode = c4.selectbox(
    "Chart Focus",
    options=["Daily attack counts", "Daily mean anomaly"],
    index=0,
    key="ts_mode",
)

ts_df = df_date.copy()
if attack_focus != "All":
    ts_df = ts_df[ts_df["Attack Type"] == attack_focus]
if protocol_focus != "All":
    ts_df = ts_df[ts_df["Protocol"] == protocol_focus]


if ts_df.empty:
    st.warning("No rows match current filters. Widen the date range or reset chart filters.")
else:
    if main_mode == "Daily attack counts":
        ts = ts_df.groupby(["Date", "Attack Type"]).size().reset_index(name="value")
        y_label = "Events per day"
        ts_title = "Daily attack counts"
    else:
        ts = ts_df.groupby(["Date", "Attack Type"])["Anomaly Scores"].mean().reset_index(name="value")
        y_label = "Mean anomaly score"
        ts_title = "Daily mean anomaly score"

    if attack_focus == "All":
        top_types = ts_df["Attack Type"].value_counts().head(6).index.tolist()
        ts = ts[ts["Attack Type"].isin(top_types)]

    fig = px.line(
        ts, x="Date", y="value", color="Attack Type", title=ts_title,
        labels={"Date": "Date", "value": y_label}
    )
    fig_ts = px.line(ts, x="Date", y="value", color="Attack Type", title=ts_title,
                     labels={"Date": "Date", "value": y_label})
    fig_ts.update_xaxes(showgrid=False)
    fig_ts.update_yaxes(showgrid=False)
    fig_ts.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_ts, width="stretch")

    # metrics stay date-global 
    total = len(df_date)
    blocked_rate = df_date["Action Taken"].eq("Blocked").mean() * 100 if total else 0
    avg_anom = df_date["Anomaly Scores"].mean() if total else np.nan
    top_attack = df_date["Attack Type"].value_counts().index[0] if total else "—"
    top_state = df_date["Geo_State"].dropna().value_counts().index[0] if df_date["Geo_State"].notna().any() else "—"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Events", f"{total:,}")
    m2.metric("% Blocked", f"{blocked_rate:.1f}%")
    m3.metric("Avg Anomaly", "—" if np.isnan(avg_anom) else f"{avg_anom:.2f}")
    m4.metric("Top attack", top_attack)
    m5.metric("Top state", top_state)

st.divider()

# attack pathways

st.subheader("Attack pathways")

PATHWAYS = {
    "Responses — Protocol → Attack Type → Action Taken": {
        "stages": ("Protocol", "Attack Type", "Action Taken"),
        "desc": "How protocols tend to end (blocked/allowed/mitigated). Look for thick links into 'Blocked' or other actions."
    },
    "Risk — Traffic Type → Attack Type → Severity Level": {
        "stages": ("Traffic Type", "Attack Type", "Severity Level"),
        "desc": "Which traffic types are associated with higher severity outcomes. Look for links ending in 'High'."
    },
    "Impact lands — Network Segment → Attack Type → Action Taken": {
        "stages": ("Network Segment", "Attack Type", "Action Taken"),
        "desc": "Which network segments receive attacks and what happens to them (response/action).\n\n"
        "\t\tSegmentA (Production/Server): Critical systems and databases. Highly restrictive."
        "\n\n\t\tSegmentB (Employee Workstations): Internal user devices with limited access."
        "\n\n\t\tSegmentC (Guest Wi-Fi): Internet-only access, isolated from internal resources."

    },
    "Signature trail — Protocol → Attack Signature → Attack Type": {
        "stages": ("Protocol", "Attack Signature", "Attack Type"),
        "desc": "See which signatures cluster under protocols and which attack types they represent."
    },
}

path_choice = st.selectbox("Choose a pathway", list(PATHWAYS.keys()), index=0, key="sankey_path",)
stage1, stage2, stage3 = PATHWAYS[path_choice]["stages"]

st.caption(f"**What to look for:** {PATHWAYS[path_choice]['desc']}")

if len(df_date) < 10:
    st.info("Not enough events under current filters to build a stable Sankey. Widen the date range or set focus filters to All.")
else:
    sankey = build_sankey(df_date, stage1, stage2, stage3)
    if sankey is None:
        st.info("No valid pathways for this preset under the current filters.")
    else:
        st.plotly_chart(sankey, width="stretch")

st.divider()

# hotspot map

st.subheader("Geographic hotspots (India state-level)")

mdf = df_date.dropna(subset=["Geo_State"]).copy()
if mdf.empty:
    st.info("No geo rows available under current date range.")
else:
    geojson_obj = load_india_geojson()
    state_key = make_state_key(geojson_obj) if geojson_obj else None

    if not (geojson_obj and state_key):
        st.warning("GeoJSON not available. Add a local geojson file for reliability.")
    else:
        # Aggregate all three metrics per state
        state_stats = (
            mdf.groupby("Geo_State")
            .agg(
                attack_count=("Attack Type", "size"),
                high_severity_count=("Severity Level", lambda x: (x.astype(str).str.lower() == "high").sum()),
                mean_anomaly=("Anomaly Scores", "mean"),
            )
            .reset_index()
        )

        # Normalize (0–1) so they can be combined fairly
        for col in ["attack_count", "high_severity_count", "mean_anomaly"]:
            mx = state_stats[col].max()
            state_stats[col + "_norm"] = (state_stats[col] / mx) if mx and mx > 0 else 0

        # Composite score (you can tune weights)
        w1, w2, w3 = 0.40, 0.35, 0.25
        state_stats["risk_score"] = (
            w1 * state_stats["attack_count_norm"]
            + w2 * state_stats["high_severity_count_norm"]
            + w3 * state_stats["mean_anomaly_norm"]
        )

        st.markdown(
            "**Color = *composite cyber risk score* per state. "
            "It combines (1) attack volume, (2) high-severity volume, and (3) anomaly intensity. "
            "Hover for the raw metrics."
        )

        fig_map = px.choropleth(
            state_stats,
            geojson=geojson_obj,
            locations="Geo_State",
            featureidkey=f"properties.{state_key}",
            color="risk_score",
            hover_name="Geo_State",
            hover_data={
                "attack_count": ":,",
                "high_severity_count": ":,",
                "mean_anomaly": ":.2f",
                "risk_score": ":.2f",
            },
            title="India hotspots — Cyber Risk Score (volume, impact, strangeness)",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(height=820, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_map, width="stretch")

st.divider()
st.markdown(
    """
### **Interpretation:** 
Across the 2020 to 2023 period, cyber attacks exhibit a temporal pattern, characterized by steady daily activity rather than isolated surges. While overall attack volume remains relatively stable over time, shifts in anomaly intensity reveal that the most frequent attacks are not always the most anomalous, indicating that attack frequency and behavioral strangeness evolve independently. The sankey diagram analysis reveals that although attacks originate across multiple protocols and traffic types, they are not evenly distributed across outcomes. Instead, the protocol–attack signature–attack type pathway carries a disproportionately large share of events, indicating that attacks repeatedly follow recognizable structural patterns. This clustering suggests that signature-based behaviors play a central role in shaping observed attack outcomes. This suggests that cyber threats follow repeatable structural patterns rather than random or isolated behaviors. Geographically, attack hotspots cluster unevenly across Indian states, with the highest-risk regions emerging where attack volume, high-severity frequency, and anomaly intensity intersect. These findings indicate that cyber risk is multidimensional and distributed across time, network pathways, and space, emphasizing the importance of integrated analysis rather than reliance on any single metric. 
\n*Those thick links show that most attacks repeatedly follow the same protocol-signature-attack combinations, meaning the threat landscape is structured and predictable at a pattern level, even if it looks noisy over time.*
"""
)       

