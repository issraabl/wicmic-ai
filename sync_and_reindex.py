"""
sync_and_reindex.py — Export + Réindexation en une seule commande.

Usage:
    python sync_and_reindex.py --token <jwt>
    python sync_and_reindex.py          (lit depuis .env WICMIC_TOKEN)
"""

import subprocess
import sys

if __name__ == "__main__":
    print("="*50)
    print("🔄 SYNC WICMIC — Export + Réindexation")
    print("="*50)

    # Étape 1 — Export
    print("\n📡 Étape 1 : Export des données depuis le backend...")
    result = subprocess.run(
        [sys.executable, "export_data.py"] + sys.argv[1:],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ Export échoué. Vérifiez le token et que le backend tourne.")
        sys.exit(1)

    # Étape 2 — Réindexation
    print("\n🗄️  Étape 2 : Réindexation ChromaDB avec les nouvelles données...")
    from rag.embedder import index_documents
    index_documents(force_reindex=True)

    print("\n✅ Sync terminé ! Les prévisions et benchmarks utilisent maintenant vos données réelles.")