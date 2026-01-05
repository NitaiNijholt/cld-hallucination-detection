#!/usr/bin/env python3
"""
Nederlandse Topic Analyse Runner
Voert LLM-gebaseerde topic clustering uit op je dataset
"""

from topic_analysis import DutchTopicAnalyzer
import sys
import os

def main():
    print("🇳🇱 Nederlandse Topic Analyse Tool")
    print("=" * 50)
    
    # Controleer of het Excel bestand bestaat
    excel_file = "replit_db_backup_08_07_2025_trail.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ Excel bestand '{excel_file}' niet gevonden!")
        print("   Zorg ervoor dat het bestand in deze map staat.")
        return
    
    try:
        # Initialiseer de analyzer
        print(f"📂 Bestand gevonden: {excel_file}")
        analyzer = DutchTopicAnalyzer(excel_file)
        
        # Voer analyse uit
        print("\n🚀 Analyse starten...")
        results = analyzer.run_analysis()
        
        if results:
            print("\n✅ Analyse succesvol voltooid!")
            print("📊 Resultaten zijn weergegeven hierboven")
            print("📈 Visualisatie opgeslagen als 'topic_analysis_results.png'")
        else:
            print("\n❌ Analyse mislukt. Controleer je data en probeer opnieuw.")
            
    except Exception as e:
        print(f"\n❌ Fout opgetreden: {e}")
        print("💡 Tip: Controleer of alle vereiste packages zijn geïnstalleerd:")
        print("   pip install -r requirements.txt")
        
if __name__ == "__main__":
    main() 