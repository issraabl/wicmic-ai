"""
export_data.py — Exporte les vraies données du backend Wicmic
vers le dossier data/ pour que le RAG utilise les données réelles.

Usage:
    python export_data.py --token <votre_jwt_token>
    python export_data.py  (lit le token depuis .env ou vous le demande)
"""

import httpx
import json
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────
BASE_URL  = "https://localhost:7128/api"
DATA_DIR  = "./data"
TIMEOUT   = 30.0


def get_token(args_token: str = None) -> str:
    """Récupère le JWT token."""
    # 1. Depuis les arguments
    if args_token:
        return args_token
    # 2. Depuis .env
    token = os.getenv("WICMIC_TOKEN", "")
    if token:
        return token
    # 3. Demande à l'utilisateur
    print("\n🔑 Token JWT requis pour accéder à l'API Wicmic.")
    print("   (Copiez-le depuis votre navigateur : F12 → Application → localStorage → wicmic_token)")
    token = input("   Token : ").strip()
    return token


def get_headers(token: str) -> dict:
    return {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {token}",
    }


def fetch(client: httpx.Client, endpoint: str) -> list:
    """Appelle un endpoint et retourne la liste."""
    try:
        url = f"{BASE_URL}/{endpoint}"
        print(f"   → GET {url}")
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return [data] if data else []
    except httpx.HTTPStatusError as e:
        print(f"   ⚠️  {endpoint} : HTTP {e.response.status_code}")
        return []
    except Exception as e:
        print(f"   ⚠️  {endpoint} : {e}")
        return []


def normalize_mesure(m: dict) -> dict:
    """Normalise une mesure pour le RAG."""
    energie_raw  = m.get("energie")    or m.get("Energie")    or {}
    equipement_r = m.get("equipement") or m.get("Equipement") or {}
    energie_id   = m.get("energieId")  or m.get("EnergieId")  or energie_raw.get("idEnergie") or 0
    return {
        "idMesure":      m.get("idMesure")     or m.get("IdMesure")     or 0,
        "valeur":        float(m.get("valeur") or m.get("Valeur")       or 0),
        "dateMesure":    m.get("dateMesure")   or m.get("DateMesure")   or "",
        "sourceDonnee":  m.get("sourceDonnee") or m.get("SourceDonnee") or "",
        "energieId":     energie_id,
        "energieNom":    energie_raw.get("nom") or energie_raw.get("Nom") or f"Energie {energie_id}",
        "unite":         energie_raw.get("unite") or energie_raw.get("Unite") or "unité",
        "equipementNom": equipement_r.get("nom") or equipement_r.get("Nom") or "",
        "commentaire":   m.get("commentaire")  or m.get("Commentaire")  or "",
    }


def normalize_alerte(a: dict) -> dict:
    return {
        "idAlerte":     a.get("idAlerte")     or a.get("IdAlerte")     or 0,
        "type":         a.get("type")         or a.get("Type")         or "",
        "message":      a.get("message")      or a.get("Message")      or "",
        "severite":     a.get("severite")     or a.get("Severite")     or "Normale",
        "traite":       a.get("traite")       or a.get("Traite")       or False,
        "dateCreation": a.get("dateCreation") or a.get("DateCreation") or "",
        "seuil":        float(a.get("seuil") or a.get("Seuil") or 0),
    }


def normalize_equipement(e: dict) -> dict:
    energie_raw = e.get("energie") or e.get("Energie") or {}
    zone_raw    = e.get("zone")    or e.get("Zone")    or {}
    return {
        "idEquipement":   e.get("idEquipement")   or e.get("IdEquipement")   or 0,
        "nom":            e.get("nom")            or e.get("Nom")            or "",
        "typeEquipement": e.get("typeEquipement") or e.get("TypeEquipement") or "",
        "statut":         e.get("statut")         or e.get("Statut")         or "Actif",
        "puissance":      float(e.get("puissance") or e.get("Puissance") or 0),
        "localisation":   e.get("localisation")   or e.get("Localisation")   or "",
        "energieNom":     energie_raw.get("nom")  or energie_raw.get("Nom")  or "",
        "zoneNom":        zone_raw.get("nom")     or zone_raw.get("Nom")     or "",
        "dateInstallation": e.get("dateMiseEnService") or e.get("dateInstallation") or "",
    }


def normalize_anomalie(a: dict) -> dict:
    energie_raw = a.get("energie") or a.get("Energie") or {}
    return {
        "id":            a.get("id") or a.get("idAnomalie") or a.get("IdAnomalie") or 0,
        "description":   a.get("description")   or a.get("Description")   or "",
        "dateDetection": a.get("dateDetection") or a.get("DateDetection") or "",
        "resolu":        a.get("resolu")        or a.get("Resolu")        or False,
        "type":          a.get("type")          or a.get("Type")          or "Anomalie",
        "energieNom":    energie_raw.get("nom") or energie_raw.get("Nom") or "",
    }


def save_json(data: list, filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(data)} enregistrements → {path}")


def build_benchmark_json(mesures: list, energies: list) -> list:
    """Calcule les benchmarks mensuels depuis les vraies mesures."""
    from collections import defaultdict
    benchmarks = []

    for energie in energies:
        eid  = energie.get("idEnergie") or energie.get("IdEnergie") or 0
        nom  = energie.get("nom")       or energie.get("Nom")       or ""
        unite = energie.get("unite")    or energie.get("Unite")     or ""

        mes_en = [
            m for m in mesures
            if str(m.get("energieId", "")) == str(eid)
        ]
        if not mes_en:
            continue

        # Groupe par mois
        monthly: dict = defaultdict(list)
        for m in mes_en:
            date = m.get("dateMesure", "")[:7]  # YYYY-MM
            if date:
                monthly[date].append(float(m.get("valeur", 0)))

        # Calcule totaux mensuels
        mois_data = []
        for mois, vals in sorted(monthly.items()):
            mois_data.append({"mois": mois, "total": round(sum(vals), 2)})

        if len(mois_data) >= 2:
            mois_act  = mois_data[-1]["total"]
            mois_prec = mois_data[-2]["total"]
            moyenne   = round(sum(m["total"] for m in mois_data) / len(mois_data), 2)
            variation = round(((mois_act - mois_prec) / mois_prec * 100) if mois_prec else 0, 1)
        elif len(mois_data) == 1:
            mois_act  = mois_data[0]["total"]
            mois_prec = 0
            moyenne   = mois_act
            variation = 0
        else:
            continue

        benchmarks.append({
            "energie":       nom,
            "unite":         unite,
            "moisActuel":    mois_act,
            "moisPrecedent": mois_prec,
            "moyenne":       moyenne,
            "variation":     variation,
            "historique":    mois_data,
        })

    return benchmarks


def main():
    parser = argparse.ArgumentParser(description="Export données Wicmic → data/")
    parser.add_argument("--token", type=str, help="JWT token")
    args = parser.parse_args()

    token = get_token(args.token)
    if not token:
        print("❌ Token requis. Abandon.")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n🚀 Export des données depuis {BASE_URL}")
    print(f"   Dossier cible : {DATA_DIR}/\n")

    # SSL désactivé car localhost avec certificat auto-signé
    with httpx.Client(
        headers=get_headers(token),
        timeout=TIMEOUT,
        verify=False,  # ← HTTPS localhost auto-signé
    ) as client:

        # ── Énergies ──────────────────────────────────────
        print("📡 Récupération des énergies...")
        energies_raw = fetch(client, "Energies")
        energies = energies_raw  # pas besoin de normaliser
        save_json(energies, "energies.json")

        # ── Mesures ───────────────────────────────────────
        print("\n📡 Récupération des mesures...")
        mesures_raw = fetch(client, "Mesures")
        mesures = [normalize_mesure(m) for m in mesures_raw]
        save_json(mesures, "mesures.json")

        # ── Alertes ───────────────────────────────────────
        print("\n📡 Récupération des alertes...")
        alertes_raw = fetch(client, "Alertes")
        alertes = [normalize_alerte(a) for a in alertes_raw]
        save_json(alertes, "alertes.json")

        # ── Équipements ───────────────────────────────────
        print("\n📡 Récupération des équipements...")
        equipements_raw = fetch(client, "Equipements")
        equipements = [normalize_equipement(e) for e in equipements_raw]
        save_json(equipements, "equipements.json")

        # ── Anomalies ─────────────────────────────────────
        print("\n📡 Récupération des anomalies...")
        anomalies_raw = fetch(client, "Anomalies")
        anomalies = [normalize_anomalie(a) for a in anomalies_raw]
        save_json(anomalies, "anomalies.json")

        # ── Benchmarks calculés ───────────────────────────
        print("\n📊 Calcul des benchmarks mensuels...")
        benchmarks = build_benchmark_json(mesures, energies)
        save_json(benchmarks, "benchmarks.json")

    # ── Rapport export ────────────────────────────────────
    print("\n" + "="*50)
    print("✅ Export terminé !")
    print(f"   Mesures     : {len(mesures)}")
    print(f"   Alertes     : {len(alertes)}")
    print(f"   Équipements : {len(equipements)}")
    print(f"   Anomalies   : {len(anomalies)}")
    print(f"   Benchmarks  : {len(benchmarks)} énergies")
    print(f"   Exporté le  : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("="*50)
    print("\n🔄 Lance maintenant la réindexation ChromaDB:")
    print("   python -m rag.embedder  (force_reindex=True)")
    print("   ou: uvicorn api.main:app → POST /reindex\n")


if __name__ == "__main__":
    main()