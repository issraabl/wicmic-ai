import os
import httpx
import json
import re
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from rag.embedder import search_documents, index_documents

load_dotenv()

OLLAMA_URL    = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
BACKEND_URL   = os.getenv("BACKEND_URL",  "https://localhost:7128/api")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_context(query: str, n_results: int = 3) -> str:
    docs = search_documents(query, n_results=n_results)
    if not docs:
        return "Aucune donnée disponible."
    return "\n".join([f"- {d}" for d in docs])


def ask_ollama(prompt: str, num_predict: int = 600) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    try:
        with httpx.Client(timeout=600.0) as client:
            print(f"[Retriever] Appel Ollama ({OLLAMA_MODEL})...")
            response = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
    except httpx.TimeoutException:
        print("[Retriever] ❌ Timeout Ollama")
        return ""
    except httpx.ConnectError:
        print("[Retriever] ❌ Connexion refusée — lance: ollama serve")
        return ""
    except Exception as e:
        print(f"[Retriever] ❌ Erreur: {str(e)}")
        return ""


def _clean_raw(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    return raw.strip()


def _parse_json_response(raw: str, fallback: dict) -> dict:
    raw = _clean_raw(raw)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        print("[Retriever] ⚠️  Aucun JSON détecté.")
        return fallback
    json_str = raw[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    try:
        open_b  = json_str.count("{")
        close_b = json_str.count("}")
        open_br  = json_str.count("[")
        close_br = json_str.count("]")
        repaired = json_str.rstrip().rstrip(",")
        if open_br > close_br:
            repaired += "]" * (open_br - close_br)
        if open_b > close_b:
            repaired += "}" * (open_b - close_b)
        return json.loads(repaired)
    except Exception:
        pass
    print("[Retriever] ⚠️  JSON invalide — fallback local.")
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# Fetch données réelles depuis le backend .NET
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_backend(path: str) -> list:
    headers = {}
    if BACKEND_TOKEN:
        headers["Authorization"] = f"Bearer {BACKEND_TOKEN}"
    try:
        with httpx.Client(timeout=30.0, verify=False) as client:
            resp = client.get(f"{BACKEND_URL}/{path}", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"[Retriever] ⚠️ Erreur fetch backend /{path}: {e}")
        return []


def _build_historique_depuis_backend() -> dict:
    mesures  = _fetch_backend("Mesures")
    energies = _fetch_backend("Energies")

    energie_map = {}
    for e in energies:
        eid   = str(e.get("idEnergie") or e.get("IdEnergie") or 0)
        nom   = e.get("nom")   or e.get("Nom")   or ""
        unite = e.get("unite") or e.get("Unite") or ""
        energie_map[eid] = {"nom": nom, "unite": unite}

    monthly = defaultdict(lambda: defaultdict(float))
    for m in mesures:
        eid      = str(m.get("energieId") or m.get("EnergieId") or 0)
        valeur   = float(m.get("valeur")  or m.get("Valeur")  or 0)
        date_str = m.get("dateMesure")    or m.get("DateMesure") or ""
        if not date_str or not eid or eid == "0":
            continue
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            mois = date.strftime("%Y-%m")
            monthly[eid][mois] += valeur
        except Exception:
            continue

    result_energies = []
    for eid, mois_data in monthly.items():
        info  = energie_map.get(eid, {"nom": f"Energie {eid}", "unite": ""})
        mois_list = sorted(mois_data.items())
        if len(mois_list) < 2:
            continue
        result_energies.append({
            "nom":   info["nom"],
            "unite": info["unite"],
            "mois":  [{"mois": k, "total": round(v, 1)} for k, v in mois_list],
        })

    print(f"[Retriever] ✅ Historique réel: {len(result_energies)} énergies, {len(mesures)} mesures")
    return {"energies": result_energies}


def _build_benchmark_depuis_backend() -> dict:
    mesures  = _fetch_backend("Mesures")
    energies = _fetch_backend("Energies")

    energie_map = {}
    for e in energies:
        eid   = str(e.get("idEnergie") or e.get("IdEnergie") or 0)
        nom   = e.get("nom")   or e.get("Nom")   or ""
        unite = e.get("unite") or e.get("Unite") or ""
        energie_map[eid] = {"nom": nom, "unite": unite}

    now        = datetime.now()
    mois_act   = now.strftime("%Y-%m")
    prev_month = datetime(now.year, now.month - 1, 1) if now.month > 1 else datetime(now.year - 1, 12, 1)
    mois_prec  = prev_month.strftime("%Y-%m")

    totaux_act  = defaultdict(float)
    totaux_prec = defaultdict(float)
    totaux_all  = defaultdict(list)

    for m in mesures:
        eid      = str(m.get("energieId") or m.get("EnergieId") or 0)
        valeur   = float(m.get("valeur")  or m.get("Valeur")  or 0)
        date_str = m.get("dateMesure")    or m.get("DateMesure") or ""
        if not date_str or not eid or eid == "0":
            continue
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            mois = date.strftime("%Y-%m")
            if mois == mois_act:
                totaux_act[eid]  += valeur
            elif mois == mois_prec:
                totaux_prec[eid] += valeur
            totaux_all[eid].append(valeur)
        except Exception:
            continue

    result = []
    for eid, info in energie_map.items():
        act  = round(totaux_act.get(eid, 0), 1)
        prec = round(totaux_prec.get(eid, 0), 1)
        vals = totaux_all.get(eid, [])
        moy  = round(sum(vals) / len(vals), 1) if vals else 0
        if act == 0 and prec == 0:
            continue
        result.append({
            "nom":           info["nom"],
            "unite":         info["unite"],
            "moisActuel":    act,
            "moisPrecedent": prec,
            "moyenne":       moy,
        })

    print(f"[Retriever] ✅ Benchmark réel: {len(result)} énergies")
    return {"energies": result}


# ══════════════════════════════════════════════════════════════════════════════
# ▶▶▶ VRAIE RÉGRESSION LINÉAIRE Python — calcul pur, sans Ollama ◀◀◀
# ══════════════════════════════════════════════════════════════════════════════

def _regression_lineaire(valeurs: list[float]) -> dict:
    """
    Régression linéaire sur une série temporelle mensuelle.
    Retourne : prévision mois prochain, R², tendance, variation %.
    """
    n = len(valeurs)
    if n < 2:
        return {"prevision": valeurs[-1] if valeurs else 0, "r2": 0.0, "tendance": "flat", "variation_pct": 0.0}

    x = np.arange(n, dtype=float)
    y = np.array(valeurs, dtype=float)

    # Coefficients de régression
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy  = np.sum((x - x_mean) * (y - y_mean))
    ss_xx  = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return {"prevision": round(float(y_mean), 1), "r2": 0.0, "tendance": "flat", "variation_pct": 0.0}

    a = ss_xy / ss_xx          # pente
    b = y_mean - a * x_mean    # ordonnée à l'origine

    # Prévision mois n+1
    prevision = max(0.0, a * n + b)

    # R² (coefficient de détermination)
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r2 = max(0.0, min(1.0, r2))

    # Variation % par rapport au dernier mois réel
    dernier = valeurs[-1]
    variation_pct = ((prevision - dernier) / dernier * 100) if dernier > 0 else 0.0

    # Tendance
    if variation_pct > 3:
        tendance = "up"
    elif variation_pct < -3:
        tendance = "down"
    else:
        tendance = "flat"

    return {
        "prevision":     round(float(prevision), 1),
        "r2":            round(float(r2), 4),
        "tendance":      tendance,
        "variation_pct": round(float(variation_pct), 1),
        "pente":         round(float(a), 4),
        "dernier_reel":  round(float(dernier), 1),
        "moyenne":       round(float(y_mean), 1),
    }


def _calcul_fiabilite(regressions: list[dict], nb_mois: int) -> int:
    """
    Calcule la fiabilité globale du modèle basée sur les R² et le nb de données.
    Retourne un pourcentage 0-100.
    """
    if not regressions:
        return 0

    r2_moyen = np.mean([r["r2"] for r in regressions])

    # Bonus selon le nombre de mois de données
    bonus_data = min(20, (nb_mois - 2) * 3)  # +3% par mois supplémentaire jusqu'à +20%

    fiabilite = int(r2_moyen * 70) + bonus_data + 10  # base 10%
    return max(0, min(99, fiabilite))


def _generer_raisonnement_ollama(energies_data: list[dict]) -> str:
    """
    Demande à Ollama UNIQUEMENT un raisonnement textuel.
    Les chiffres sont déjà calculés par la régression Python.
    """
    resume = ""
    for e in energies_data:
        reg = e["regression"]
        resume += (
            f"- {e['nom']} ({e['unite']}) : dernier mois réel = {reg['dernier_reel']}, "
            f"prévision mois prochain = {reg['prevision']}, "
            f"variation attendue = {reg['variation_pct']:+.1f}%, "
            f"tendance = {reg['tendance']}, R² = {reg['r2']:.2f}\n"
        )

    prompt = f"""Tu es expert en efficacité énergétique industrielle.
Voici les résultats d'une régression linéaire sur les consommations réelles :

{resume}

Rédige en 3-4 phrases un raisonnement analytique professionnel en français qui :
1. Commente les tendances observées
2. Explique les prévisions calculées
3. Suggère des priorités d'action

IMPORTANT: Réponds UNIQUEMENT avec ce JSON valide :
{{"raisonnement": "Ton analyse ici."}}"""

    raw = ask_ollama(prompt, num_predict=400)
    if raw:
        parsed = _parse_json_response(raw, fallback=None)
        if parsed and parsed.get("raisonnement"):
            return parsed["raisonnement"]

    # Fallback textuel si Ollama échoue
    parties = []
    for e in energies_data:
        reg = e["regression"]
        if reg["tendance"] == "up":
            parties.append(
                f"{e['nom']} montre une tendance à la hausse ({reg['variation_pct']:+.1f}%) "
                f"avec une prévision de {reg['prevision']} {e['unite']} pour le mois prochain."
            )
        elif reg["tendance"] == "down":
            parties.append(
                f"{e['nom']} affiche une baisse encourageante ({reg['variation_pct']:+.1f}%) "
                f"avec une prévision de {reg['prevision']} {e['unite']} pour le mois prochain."
            )
        else:
            parties.append(
                f"{e['nom']} reste stable autour de {reg['moyenne']} {e['unite']} par mois "
                f"(prévision : {reg['prevision']} {e['unite']})."
            )
    return " ".join(parties) if parties else "Analyse basée sur régression linéaire des données réelles."


def _generer_recos_ollama(energies_data: list[dict]) -> list[dict]:
    """
    Génère 3 recommandations concrètes via Ollama basées sur les vraies tendances.
    """
    resume = ""
    for e in energies_data:
        reg = e["regression"]
        resume += f"- {e['nom']}: tendance {reg['tendance']}, variation {reg['variation_pct']:+.1f}%\n"

    prompt = f"""Expert énergie industrielle. Données réelles de régression :
{resume}

Génère exactement 3 recommandations concrètes et chiffrées basées sur ces tendances.
IMPORTANT: JSON valide uniquement, sans texte avant ou après.
Format :
{{"recos":[
  {{"titre":"Titre court","description":"Description actionnable et précise.","economie":150,"urgence":"haute"}},
  {{"titre":"Titre court","description":"Description actionnable et précise.","economie":80,"urgence":"normale"}},
  {{"titre":"Titre court","description":"Description actionnable et précise.","economie":40,"urgence":"normale"}}
]}}"""

    raw = ask_ollama(prompt, num_predict=600)
    if raw:
        parsed = _parse_json_response(raw, fallback=None)
        if parsed and parsed.get("recos") and len(parsed["recos"]) > 0:
            return parsed["recos"]

    # Fallback local basé sur les tendances réelles
    recos = []
    for e in energies_data:
        reg = e["regression"]
        if reg["tendance"] == "up" and len(recos) < 3:
            recos.append({
                "titre":       f"Réduire la consommation {e['nom']}",
                "description": f"La consommation {e['nom']} augmente de {reg['variation_pct']:+.1f}% par mois. "
                               f"Prévision : {reg['prevision']} {e['unite']}. "
                               f"Audit des équipements et horaires recommandé.",
                "economie":    int(reg["prevision"] * 0.1),
                "urgence":     "haute",
            })
        elif reg["tendance"] == "down" and len(recos) < 3:
            recos.append({
                "titre":       f"Maintenir la performance {e['nom']}",
                "description": f"La baisse de {abs(reg['variation_pct']):.1f}% sur {e['nom']} est positive. "
                               f"Documenter les actions mises en place pour pérenniser cette tendance.",
                "economie":    int(reg["prevision"] * 0.05),
                "urgence":     "normale",
            })

    if not recos:
        recos = [{
            "titre":       "Suivi mensuel renforcé",
            "description": "Mettre en place des relevés hebdomadaires pour affiner les prévisions.",
            "economie":    50,
            "urgence":     "normale",
        }]

    return recos[:3]


# ══════════════════════════════════════════════════════════════════════════════
# Prévisions — Architecture hybride : Régression Python + Raisonnement Ollama
# ══════════════════════════════════════════════════════════════════════════════

def generate_previsions(historique: dict) -> dict:
    print("[Retriever] ── DÉMARRAGE PRÉVISIONS ──────────────────────────────")

    # 1. Récupération données réelles
    print("[Retriever] Récupération historique depuis le backend...")
    historique_reel = _build_historique_depuis_backend()

    if historique_reel["energies"]:
        historique = historique_reel
        print(f"[Retriever] ✅ {len(historique['energies'])} énergies récupérées du backend")
    else:
        print("[Retriever] ⚠️ Backend inaccessible — données front utilisées")

    energies = historique.get("energies", [])
    if not energies:
        return {
            "elec": 0, "eau": 0, "gazoil": 0,
            "fiabilite": 0, "elecTrend": "flat",
            "elecVar": "0", "hasEnoughData": False,
            "raisonnement": "Aucune donnée disponible.",
            "recos": [],
        }

    # 2. RÉGRESSION LINÉAIRE PYTHON sur chaque énergie
    print("[Retriever] Calcul des régressions linéaires...")
    energies_avec_regression = []
    previsions_par_nom = {}

    for en in energies:
        nom   = en.get("nom",   "?")
        unite = en.get("unite", "")
        mois  = en.get("mois",  [])

        if len(mois) < 2:
            print(f"[Retriever] ⚠️ {nom}: seulement {len(mois)} mois — ignoré")
            continue

        # Série temporelle : valeurs mensuelles ordonnées
        valeurs = [m["total"] for m in sorted(mois, key=lambda x: x["mois"])]
        print(f"[Retriever] {nom}: {len(valeurs)} mois de données → {valeurs}")

        reg = _regression_lineaire(valeurs)
        print(f"[Retriever] {nom}: prévision={reg['prevision']}, R²={reg['r2']:.3f}, tendance={reg['tendance']}")

        energies_avec_regression.append({
            "nom":       nom,
            "unite":     unite,
            "valeurs":   valeurs,
            "regression": reg,
        })
        previsions_par_nom[nom.lower()] = reg

    if not energies_avec_regression:
        return {
            "elec": 0, "eau": 0, "gazoil": 0,
            "fiabilite": 0, "elecTrend": "flat",
            "elecVar": "0", "hasEnoughData": False,
            "raisonnement": "Données insuffisantes pour la régression.",
            "recos": [],
        }

    # 3. Extraction des valeurs calculées pour les 3 énergies principales
    def _find_prevision(keywords: list[str]) -> float:
        for kw in keywords:
            for nom, reg in previsions_par_nom.items():
                if kw in nom:
                    return reg["prevision"]
        return 0.0

    def _find_regression(keywords: list[str]) -> dict | None:
        for kw in keywords:
            for nom, reg in previsions_par_nom.items():
                if kw in nom:
                    return reg
        return None

    elec_reg   = _find_regression(["elec", "électr"])
    eau_reg    = _find_regression(["eau", "water"])
    gazoil_reg = _find_regression(["gazoil", "gaz", "fuel"])

    elec_prev   = elec_reg["prevision"]   if elec_reg   else _find_prevision(["elec", "électr"])
    eau_prev    = eau_reg["prevision"]    if eau_reg    else _find_prevision(["eau", "water"])
    gazoil_prev = gazoil_reg["prevision"] if gazoil_reg else _find_prevision(["gazoil", "gaz"])

    elec_trend  = elec_reg["tendance"]      if elec_reg else "flat"
    elec_var    = str(abs(elec_reg["variation_pct"])) if elec_reg else "0"

    # 4. Fiabilité globale basée sur les R²
    nb_mois_max  = max(len(e["valeurs"]) for e in energies_avec_regression)
    all_regs     = [e["regression"] for e in energies_avec_regression]
    fiabilite    = _calcul_fiabilite(all_regs, nb_mois_max)
    print(f"[Retriever] Fiabilité calculée : {fiabilite}% (R² moyen: {np.mean([r['r2'] for r in all_regs]):.3f})")

    # 5. Raisonnement textuel via Ollama (non-bloquant sur les chiffres)
    print("[Retriever] Génération du raisonnement via Ollama...")
    raisonnement = _generer_raisonnement_ollama(energies_avec_regression)

    # 6. Recommandations via Ollama
    print("[Retriever] Génération des recommandations via Ollama...")
    recos = _generer_recos_ollama(energies_avec_regression)

    result = {
        "elec":          elec_prev,
        "eau":           eau_prev,
        "gazoil":        gazoil_prev,
        "fiabilite":     fiabilite,
        "elecTrend":     elec_trend,
        "elecVar":       elec_var,
        "hasEnoughData": True,
        "raisonnement":  raisonnement,
        "recos":         recos,
        # Données détaillées pour debug / affichage enrichi
        "_details": {
            e["nom"]: {
                "prevision":     e["regression"]["prevision"],
                "r2":            e["regression"]["r2"],
                "variation_pct": e["regression"]["variation_pct"],
                "tendance":      e["regression"]["tendance"],
                "nb_mois":       len(e["valeurs"]),
                "dernier_reel":  e["regression"]["dernier_reel"],
            }
            for e in energies_avec_regression
        },
    }

    print(f"[Retriever] ✅ Prévisions calculées: elec={elec_prev}, eau={eau_prev}, gazoil={gazoil_prev}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_insight(nom: str, unite: str, mois_act: float,
                      mois_prec: float, moyenne: float,
                      variation: float, vs_moy: str) -> str:
    if variation < -3:
        return (
            f"✅ {nom} affiche une baisse significative de {abs(variation):.1f}% ce mois "
            f"({mois_act} vs {mois_prec} {unite}). La consommation est {vs_moy} à la moyenne "
            f"de {moyenne} {unite}. Cette tendance positive indique une meilleure maîtrise "
            f"de la consommation. Il est recommandé d'analyser les facteurs ayant contribué "
            f"à cette baisse afin de les reproduire les prochains mois."
        )
    elif variation > 3:
        return (
            f"⚠️ {nom} enregistre une hausse préoccupante de {variation:.1f}% ce mois "
            f"({mois_act} vs {mois_prec} {unite}). La consommation est {vs_moy} à la moyenne "
            f"de {moyenne} {unite}. Une investigation est recommandée pour identifier les causes : "
            f"vérifier les équipements associés, les horaires d'utilisation et les éventuelles "
            f"fuites ou pannes. Des actions correctives doivent être mises en place rapidement."
        )
    else:
        return (
            f"→ {nom} est stable ce mois avec {mois_act} {unite} (variation {variation:+.1f}% "
            f"vs mois précédent). La consommation est {vs_moy} à la moyenne de {moyenne} {unite}. "
            f"La situation est maîtrisée. Continuer le suivi mensuel et maintenir les bonnes "
            f"pratiques de gestion énergétique en place."
        )


def _generate_resume_global(benchmarks: list, context: str) -> str:
    resume_data = ""
    for b in benchmarks:
        resume_data += (
            f"- {b['energie']}: {b['variation']:+.1f}% "
            f"(courant={b['moisActuel']} prec={b['moisPrecedent']} {b['unite']}, "
            f"statut={b['position']})\n"
        )

    prompt = f"""Expert énergie industrielle. Résumé exécutif global en français basé sur ces données réelles.

DONNÉES RÉELLES:
{resume_data}

Analyse: situation globale, points critiques, points positifs, priorités d action concrètes.
3-4 phrases détaillées basées sur les vraies valeurs ci-dessus.
IMPORTANT: JSON valide uniquement.
Format: {{"resume": "Résumé détaillé ici."}}"""

    raw = ask_ollama(prompt, num_predict=500)
    if raw:
        parsed = _parse_json_response(raw, fallback=None)
        if parsed:
            resume = parsed.get("resume", "")
            if resume:
                return resume

    # Fallback local
    worse  = [b for b in benchmarks if b["position"] == "worse"]
    better = [b for b in benchmarks if b["position"] == "better"]
    same   = [b for b in benchmarks if b["position"] == "same"]
    parts  = []
    if worse:
        noms  = ", ".join([b["energie"] for b in worse])
        vars_ = [f"{b['variation']:+.1f}%" for b in worse]
        parts.append(f"Hausse préoccupante pour {noms} ({', '.join(vars_)}) — investigation urgente recommandée.")
    if better:
        noms  = ", ".join([b["energie"] for b in better])
        vars_ = [f"{abs(b['variation']):.1f}%" for b in better]
        parts.append(f"Bonne performance pour {noms} (baisse de {', '.join(vars_)}) — tendance positive à maintenir.")
    if same:
        noms = ", ".join([b["energie"] for b in same])
        parts.append(f"{noms} stable — continuer le suivi mensuel.")
    return " ".join(parts) if parts else "Analyse énergétique mensuelle complétée."


def generate_benchmark(data: dict) -> dict:
    print("[Retriever] Récupération des données benchmark depuis le backend...")
    benchmark_reel = _build_benchmark_depuis_backend()

    if benchmark_reel["energies"]:
        data = benchmark_reel
        print(f"[Retriever] ✅ Données benchmark réelles ({len(data['energies'])} énergies)")
    else:
        print("[Retriever] ⚠️ Backend inaccessible — données front utilisées")

    context  = build_context("benchmark comparaison consommation equipement", n_results=3)
    energies = data.get("energies", [])

    if not energies:
        return {"benchmarks": [], "resumeGlobal": "Aucune donnée disponible."}

    benchmarks = []
    for en in energies:
        nom       = en.get("nom",           "?")
        unite     = en.get("unite",         "")
        mois_act  = en.get("moisActuel",    0)
        mois_prec = en.get("moisPrecedent", 0)
        moyenne   = en.get("moyenne",       0)
        variation = round(
            ((mois_act - mois_prec) / mois_prec * 100) if mois_prec else 0, 1
        )
        position  = "better" if variation < -3 else "worse" if variation > 3 else "same"
        vs_moy    = "supérieure" if mois_act > moyenne else "inférieure" if mois_act < moyenne else "égale"
        insight   = _fallback_insight(nom, unite, mois_act, mois_prec, moyenne, variation, vs_moy)

        benchmarks.append({
            "energie":       nom,
            "unite":         unite,
            "moisActuel":    mois_act,
            "moisPrecedent": mois_prec,
            "moyenne":       moyenne,
            "variation":     variation,
            "position":      position,
            "insight":       insight,
            "hasData":       mois_act > 0 or mois_prec > 0,
        })

    print("[Retriever] Génération résumé global...")
    resume = _generate_resume_global(benchmarks, context)
    return {"benchmarks": benchmarks, "resumeGlobal": resume}


# ══════════════════════════════════════════════════════════════════════════════
# Chat RAG
# ══════════════════════════════════════════════════════════════════════════════

_SOCIAL_TRIGGERS = {
    "merci","thanks","thank you","super","parfait","ok","okay",
    "bien","bonne journée","bonsoir","bonjour","salut","hello",
    "au revoir","bye","nickel","top","cool","d'accord","daccord",
    "compris","vu","👍","🙏","😊",
}
_SOCIAL_RESPONSES = {
    "merci":         "De rien ! N'hésitez pas si vous avez d'autres questions. 😊",
    "bonjour":       "Bonjour ! Comment puis-je vous aider avec vos données énergétiques ?",
    "salut":         "Bonjour ! Comment puis-je vous aider avec vos données énergétiques ?",
    "hello":         "Bonjour ! Comment puis-je vous aider avec vos données énergétiques ?",
    "super":         "Parfait ! Je reste disponible pour toute analyse énergétique. 👍",
    "au revoir":     "À bientôt ! 👋",
    "bye":           "À bientôt ! 👋",
}
_BUSINESS_KEYWORDS = {
    "consomm","kwh","énergi","energi","alert","équip","equip",
    "mesur","anomali","rapport","score","prévi","previ",
    "benchmark","eau","gazoil","électri","electri","compresseur",
    "réduct","reduc","seuil","tendance","analyse","factur",
}


def _is_social_message(prompt: str) -> bool:
    normalized = prompt.strip().lower().rstrip("! .,?")
    if normalized in _SOCIAL_TRIGGERS:
        return True
    words = normalized.split()
    if len(words) <= 4:
        return not any(kw in normalized for kw in _BUSINESS_KEYWORDS)
    return False


def _chat_with_rag(prompt: str, context: str = "") -> str:
    if _is_social_message(prompt):
        normalized = prompt.strip().lower().rstrip("! .,?")
        return _SOCIAL_RESPONSES.get(normalized, "De rien ! Je suis disponible pour toute question énergétique. 😊")

    rag_context = build_context(prompt, n_results=4)
    full_prompt = f"""Tu es un expert en gestion énergétique industrielle pour WICMIC.
Tu réponds TOUJOURS en français, de manière concise et précise (max 5-6 phrases).

DONNÉES RÉELLES DE LA BD:
{rag_context}

CONTEXTE ADDITIONNEL:
{context}

QUESTION: {prompt}

Réponds en français uniquement, de façon ciblée basée sur le contexte fourni."""

    return ask_ollama(full_prompt, num_predict=400)


# ══════════════════════════════════════════════════════════════════════════════
# Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST — Régression linéaire pure Python")
    print("="*60)

    # Test régression
    valeurs_test = [1200, 1350, 1180, 1420, 1500, 1380, 1600]
    reg = _regression_lineaire(valeurs_test)
    print(f"Valeurs test: {valeurs_test}")
    print(f"Régression → prévision: {reg['prevision']}, R²: {reg['r2']}, tendance: {reg['tendance']}")

    print("\n" + "="*60)
    print("TEST — Données réelles backend")
    print("="*60)
    historique = _build_historique_depuis_backend()
    print(json.dumps(historique, ensure_ascii=False, indent=2))

    print("\n" + "="*60)
    print("TEST — Prévisions complètes")
    print("="*60)
    result = generate_previsions({})
    print(json.dumps(result, ensure_ascii=False, indent=2))