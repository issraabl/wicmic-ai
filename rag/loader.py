import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    """Charge un fichier JSON et retourne une liste de dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_csv(path: str) -> List[Dict[str, Any]]:
    """Charge un fichier CSV et retourne une liste de dicts."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        return [row for row in reader]


def format_mesure(m: Dict) -> str:
    """Convertit une mesure en texte lisible pour le RAG."""
    date = m.get("dateMesure", m.get("date", "date inconnue"))
    valeur = m.get("valeur", "?")
    unite = m.get("unite", "")
    energie = m.get("energieNom", m.get("energie", "énergie inconnue"))
    equipement = m.get("equipementNom", m.get("equipement", ""))
    equip_str = f" | Équipement: {equipement}" if equipement else ""
    return (
        f"[MESURE] Date: {date} | Énergie: {energie} | "
        f"Valeur: {valeur} {unite}{equip_str}"
    )


def format_alerte(a: Dict) -> str:
    """Convertit une alerte en texte lisible pour le RAG."""
    return (
        f"[ALERTE] Type: {a.get('type','?')} | "
        f"Sévérité: {a.get('severite','?')} | "
        f"Message: {a.get('message','?')} | "
        f"Date: {a.get('dateCreation','?')} | "
        f"Traitée: {'Oui' if a.get('traite') else 'Non'}"
    )


def format_equipement(e: Dict) -> str:
    """Convertit un équipement en texte lisible pour le RAG."""
    return (
        f"[EQUIPEMENT] Nom: {e.get('nom','?')} | "
        f"Type: {e.get('typeEquipement','?')} | "
        f"Statut: {e.get('statut','Actif')} | "
        f"Puissance: {e.get('puissance','?')} kW | "
        f"Localisation: {e.get('localisation','?')}"
    )


def format_benchmark(b: Dict) -> str:
    """Convertit un benchmark en texte lisible pour le RAG."""
    return (
        f"[BENCHMARK] Énergie: {b.get('energie','?')} | "
        f"Mois courant: {b.get('moisActuel','?')} {b.get('unite','')} | "
        f"Mois précédent: {b.get('moisPrecedent','?')} {b.get('unite','')} | "
        f"Variation: {b.get('variation','?')}%"
    )


def build_documents(data_dir: str = "./data") -> List[str]:
    """
    Charge tous les fichiers du dossier data/ et retourne
    une liste de strings prêts à être embeddes.
    """
    documents = []

    # ── Mesures ──────────────────────────────────────────
    mesures = load_json(os.path.join(data_dir, "mesures.json"))
    if not mesures:
        mesures = load_csv(os.path.join(data_dir, "mesures.csv"))
    for m in mesures:
        documents.append(format_mesure(m))

    # ── Alertes ──────────────────────────────────────────
    alertes = load_json(os.path.join(data_dir, "alertes.json"))
    if not alertes:
        alertes = load_csv(os.path.join(data_dir, "alertes.csv"))
    for a in alertes:
        documents.append(format_alerte(a))

    # ── Équipements ───────────────────────────────────────
    equipements = load_json(os.path.join(data_dir, "equipements.json"))
    if not equipements:
        equipements = load_csv(os.path.join(data_dir, "equipements.csv"))
    for e in equipements:
        documents.append(format_equipement(e))

    # ── Benchmarks ────────────────────────────────────────
    benchmarks = load_json(os.path.join(data_dir, "benchmarks.json"))
    for b in benchmarks:
        documents.append(format_benchmark(b))

    print(f"[Loader] {len(documents)} documents chargés.")
    return documents


if __name__ == "__main__":
    docs = build_documents()
    for d in docs[:5]:
        print(d)