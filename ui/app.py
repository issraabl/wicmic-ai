import streamlit as st
import httpx
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Wicmic Energy AI",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Wicmic Energy — Dashboard IA")
st.caption("Prévisions et benchmarking alimentés par RAG + Ollama")

# ══════════════════════════════════════════════════════
# Sidebar — Données d'entrée
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Paramètres")

    st.subheader("Électricité (kWh)")
    elec_m1 = st.number_input("Il y a 3 mois", value=1100, key="e1")
    elec_m2 = st.number_input("Il y a 2 mois",  value=1280, key="e2")
    elec_m3 = st.number_input("Mois précédent", value=1200, key="e3")
    elec_m4 = st.number_input("Mois courant",   value=1450, key="e4")

    st.subheader("Eau (m³)")
    eau_m1 = st.number_input("Il y a 2 mois",  value=340, key="w1")
    eau_m2 = st.number_input("Mois précédent", value=320, key="w2")
    eau_m3 = st.number_input("Mois courant",   value=290, key="w3")

    st.subheader("Gazoil (L)")
    gaz_m1 = st.number_input("Il y a 2 mois",  value=480, key="g1")
    gaz_m2 = st.number_input("Mois précédent", value=520, key="g2")
    gaz_m3 = st.number_input("Mois courant",   value=610, key="g3")

    run = st.button("🚀 Lancer l'analyse IA", use_container_width=True)

# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

def call_previsions():
    payload = {
        "energies": [
            {
                "nom": "Electricité", "unite": "kWh",
                "mois": [
                    {"mois": "M-3", "total": elec_m1},
                    {"mois": "M-2", "total": elec_m2},
                    {"mois": "M-1", "total": elec_m3},
                    {"mois": "M",   "total": elec_m4},
                ],
            },
            {
                "nom": "Eau", "unite": "m³",
                "mois": [
                    {"mois": "M-2", "total": eau_m1},
                    {"mois": "M-1", "total": eau_m2},
                    {"mois": "M",   "total": eau_m3},
                ],
            },
            {
                "nom": "Gazoil", "unite": "L",
                "mois": [
                    {"mois": "M-2", "total": gaz_m1},
                    {"mois": "M-1", "total": gaz_m2},
                    {"mois": "M",   "total": gaz_m3},
                ],
            },
        ]
    }
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{API_URL}/previsions", json=payload)
        return r.json()


def call_benchmark():
    payload = {
        "energies": [
            {
                "nom": "Electricité", "unite": "kWh",
                "moisActuel":    elec_m4,
                "moisPrecedent": elec_m3,
                "moyenne":       round((elec_m1+elec_m2+elec_m3+elec_m4)/4, 1),
            },
            {
                "nom": "Eau", "unite": "m³",
                "moisActuel":    eau_m3,
                "moisPrecedent": eau_m2,
                "moyenne":       round((eau_m1+eau_m2+eau_m3)/3, 1),
            },
            {
                "nom": "Gazoil", "unite": "L",
                "moisActuel":    gaz_m3,
                "moisPrecedent": gaz_m2,
                "moyenne":       round((gaz_m1+gaz_m2+gaz_m3)/3, 1),
            },
        ]
    }
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{API_URL}/benchmark", json=payload)
        return r.json()


def trend_icon(trend: str) -> str:
    return {"up": "📈", "down": "📉", "flat": "➡️"}.get(trend, "➡️")

def position_color(pos: str) -> str:
    return {"better": "🟢", "same": "🟡", "worse": "🔴"}.get(pos, "🟡")

# ══════════════════════════════════════════════════════
# Résultats
# ══════════════════════════════════════════════════════

if run:
    tab1, tab2 = st.tabs(["🔮 Prévisions", "📊 Benchmark"])

    # ── Prévisions ────────────────────────────────────
    with tab1:
        with st.spinner("⏳ Analyse IA en cours... (30-60s)"):
            try:
                prev = call_previsions()

                st.subheader("Prévisions mois prochain")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "⚡ Électricité",
                    f"{prev.get('elec', 0)} kWh",
                    f"{trend_icon(prev.get('elecTrend','flat'))} {prev.get('elecVar','0')}%",
                )
                c2.metric("💧 Eau",    f"{prev.get('eau', 0)} m³")
                c3.metric("🛢️ Gazoil", f"{prev.get('gazoil', 0)} L")
                c4.metric("🎯 Fiabilité", f"{prev.get('fiabilite', 0)}%")

                st.info(f"💬 **Raisonnement IA :** {prev.get('raisonnement', '')}")

                recos = prev.get("recos", [])
                if recos:
                    st.subheader("💡 Recommandations")
                    for r in recos:
                        urgence = r.get("urgence", "normale")
                        color   = "🔴" if urgence == "haute" else "🟡"
                        with st.expander(f"{color} {r.get('titre', '')}"):
                            st.write(r.get("description", ""))
                            eco = r.get("economie", 0)
                            if eco:
                                st.success(f"💰 Économie estimée : {eco} DT")

                with st.expander("🔍 JSON brut"):
                    st.json(prev)

            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    # ── Benchmark ─────────────────────────────────────
    with tab2:
        with st.spinner("⏳ Analyse benchmark IA... (30-60s)"):
            try:
                bench = call_benchmark()

                st.subheader("Analyse comparative par énergie")

                benchmarks = bench.get("benchmarks", [])
                for b in benchmarks:
                    pos   = b.get("position", "same")
                    color = position_color(pos)
                    var   = b.get("variation", 0)

                    with st.expander(
                        f"{color} {b.get('energie','')} — "
                        f"{b.get('moisActuel','')} {b.get('unite','')} "
                        f"({'↑' if var > 0 else '↓'}{abs(var)}%)",
                        expanded=True,
                    ):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mois courant",   f"{b.get('moisActuel','')} {b.get('unite','')}")
                        col2.metric("Mois précédent", f"{b.get('moisPrecedent','')} {b.get('unite','')}", f"{var:+.1f}%")
                        col3.metric("Moyenne",        f"{b.get('moyenne','')} {b.get('unite','')}")
                        st.write(f"**Insight IA :** {b.get('insight','')}")

                resume = bench.get("resumeGlobal", "")
                if resume:
                    st.info(f"📋 **Résumé global :** {resume}")

                with st.expander("🔍 JSON brut"):
                    st.json(bench)

            except Exception as e:
                st.error(f"❌ Erreur : {e}")

else:
    st.info("👈 Configure les données dans la sidebar et clique sur **Lancer l'analyse IA**")

    st.subheader("Comment ça fonctionne ?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ RAG\nTes données sont indexées dans ChromaDB comme base de connaissances.")
    with col2:
        st.markdown("### 2️⃣ Ollama\ngemma3:4b analyse le contexte et génère des insights en français.")
    with col3:
        st.markdown("### 3️⃣ FastAPI\nLes résultats sont exposés via une API REST prête pour Angular.")