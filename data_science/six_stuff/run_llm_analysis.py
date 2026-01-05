#!/usr/bin/env python3
"""
LLM-gebaseerde Nederlandse Topic Analyse Runner
Gebruikt OpenAI GPT voor intelligente topic generatie en classificatie
"""

from llm_topic_analysis import LLMTopicAnalyzer
import sys
import os

def main():
    print("🤖 LLM-gebaseerde Nederlandse Topic Analyse Tool")
    print("=" * 60)
    
    # Controleer of het Excel bestand bestaat
    excel_file = "replit_db_backup_08_07_2025_trail.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ Excel bestand '{excel_file}' niet gevonden!")
        print("   Zorg ervoor dat het bestand in deze map staat.")
        return
    
    # Controleer OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY environment variable niet gevonden!")
        print("💡 Stel je API key in met: export OPENAI_API_KEY='your-api-key'")
        print("💡 Of voeg deze toe aan je .env bestand")
        
        # Probeer uit .env te laden
        try:
            with open('../../.env', 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"\'')
                        os.environ['OPENAI_API_KEY'] = api_key
                        print("✅ API key geladen uit .env bestand")
                        break
        except FileNotFoundError:
            print("❌ Geen .env bestand gevonden")
            return
    
    try:
        # Initialiseer de LLM-gebaseerde analyzer
        print(f"📂 Bestand gevonden: {excel_file}")
        analyzer = LLMTopicAnalyzer(excel_file)
        
        # Voer analyse uit
        print("\n🚀 LLM-gebaseerde analyse starten...")
        print("⏳ Dit kan enkele minuten duren vanwege OpenAI API calls...")
        
        results = analyzer.run_analysis(max_topics=8)
        
        if results:
            print("\n✅ LLM-gebaseerde analyse succesvol voltooid!")
            print("📊 Intelligente topic categorieën gegenereerd door GPT")
            print("🎯 Vragen automatisch geclassificeerd door LLM")
            print("📈 Visualisatie opgeslagen als 'llm_topic_analysis_results.png'")
        else:
            print("\n❌ Analyse mislukt. Controleer je API key en data.")
            
    except Exception as e:
        print(f"\n❌ Fout opgetreden: {e}")
        print("💡 Tips voor probleemoplossing:")
        print("   - Controleer je OpenAI API key")
        print("   - Zorg dat je voldoende credits hebt")
        print("   - Controleer je internetverbinding")
        print("   - Probeer: pip install openai --upgrade")
        
if __name__ == "__main__":
    main() 