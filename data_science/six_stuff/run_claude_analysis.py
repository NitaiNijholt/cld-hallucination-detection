#!/usr/bin/env python3
"""
Claude-gebaseerde Nederlandse Topic Analyse Runner
Gebruikt Claude 4 Sonnet voor intelligente topic generatie en classificatie
"""

from claude_topic_analysis import ClaudeTopicAnalyzer
import sys
import os

def main():
    print("🤖 Claude-gebaseerde Nederlandse Topic Analyse Tool")
    print("=" * 60)
    
    # Controleer of het Excel bestand bestaat
    excel_file = "replit_db_backup_08_07_2025_trail.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ Excel bestand '{excel_file}' niet gevonden!")
        print("   Zorg ervoor dat het bestand in deze map staat.")
        return
    
    # Controleer Claude API key
    if not os.getenv('ANTHROPIC_API_KEY') and not os.getenv('CLAUDE_API_KEY'):
        print("⚠️  ANTHROPIC_API_KEY of CLAUDE_API_KEY environment variable niet gevonden!")
        print("💡 Stel je API key in met: export ANTHROPIC_API_KEY='your-api-key'")
        print("💡 Of voeg deze toe aan je .env bestand")
        
        # Probeer uit .env te laden
        try:
            with open('../../.env', 'r') as f:
                for line in f:
                    if line.startswith('CLAUDE_API_KEY'):
                        api_key = line.split('=', 1)[1].strip().strip('"\'')
                        os.environ['ANTHROPIC_API_KEY'] = api_key
                        print("✅ Claude API key geladen uit .env bestand")
                        break
                    elif line.startswith('ANTHROPIC_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"\'')
                        os.environ['ANTHROPIC_API_KEY'] = api_key
                        print("✅ Anthropic API key geladen uit .env bestand")
                        break
        except FileNotFoundError:
            print("❌ Geen .env bestand gevonden")
            return
    
    try:
        # Initialiseer de Claude-gebaseerde analyzer
        print(f"📂 Bestand gevonden: {excel_file}")
        analyzer = ClaudeTopicAnalyzer(excel_file)
        
        # Voer analyse uit
        print("\n🚀 Claude-gebaseerde analyse starten...")
        print("⏳ Dit kan enkele minuten duren vanwege Anthropic API calls...")
        print("🧠 Claude 4 Sonnet analyseert je vragen...")
        
        results = analyzer.run_analysis(max_topics=8)
        
        if results:
            print("\n✅ Claude-gebaseerde analyse succesvol voltooid!")
            print("📊 Intelligente topic categorieën gegenereerd door Claude")
            print("🎯 Vragen automatisch geclassificeerd door Claude")
            print("📈 Visualisatie opgeslagen als 'claude_topic_analysis_results.png'")
            print("\n🎉 Claude heeft je Nederlandse vragen perfect geanalyseerd!")
        else:
            print("\n❌ Analyse mislukt. Controleer je API key en data.")
            
    except Exception as e:
        print(f"\n❌ Fout opgetreden: {e}")
        print("💡 Tips voor probleemoplossing:")
        print("   - Controleer je Anthropic API key")
        print("   - Zorg dat je voldoende credits hebt")
        print("   - Controleer je internetverbinding")
        print("   - Probeer: pip install anthropic --upgrade")
        
if __name__ == "__main__":
    main() 