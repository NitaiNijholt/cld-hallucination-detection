import pandas as pd
import numpy as np
import json
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import anthropic
import os
import warnings
warnings.filterwarnings('ignore')

class ClaudeTopicAnalyzer:
    def __init__(self, excel_file_path, anthropic_api_key=None):
        """
        Initialiseer de Claude-gebaseerde Nederlandse topic analyzer
        
        Args:
            excel_file_path (str): Pad naar het Excel bestand
            anthropic_api_key (str): Anthropic API key (optioneel, kan ook via .env)
        """
        self.excel_file_path = excel_file_path
        self.df = None
        self.questions = []
        self.topics = []
        self.question_classifications = {}
        
        # Setup Anthropic client
        if anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            # Try to get from environment
            try:
                self.client = anthropic.Anthropic()  # Will use ANTHROPIC_API_KEY from environment
            except:
                print("⚠️  Anthropic API key niet gevonden. Stel ANTHROPIC_API_KEY in als environment variable.")
                self.client = None
        
    def load_data(self):
        """Laad de data uit het Excel bestand"""
        try:
            self.df = pd.read_excel(self.excel_file_path)
            print(f"✅ Data geladen: {len(self.df)} rijen")
            print(f"📊 Kolommen: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Fout bij het laden van data: {e}")
            return False
    
    def extract_questions(self):
        """Extraheer alle vragen uit de dataset"""
        questions = []
        
        # Zoek naar kolommen die vragen bevatten
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                col_questions = self.df[col].dropna().astype(str).tolist()
                # Filter lege strings en te korte teksten
                col_questions = [q.strip() for q in col_questions if len(q.strip()) > 10]
                questions.extend(col_questions)
        
        # Verwijder duplicaten
        unique_questions = list(set(questions))
        
        # Sorteer op lengte (langere vragen zijn vaak meer informatief)
        unique_questions.sort(key=len, reverse=True)
        
        self.questions = unique_questions
        print(f"📄 {len(self.questions)} unieke vragen gevonden")
        return self.questions
    
    def generate_topics_with_claude(self, max_topics=10) -> List[Dict]:
        """
        Gebruik Claude om topics te genereren op basis van alle vragen
        
        Args:
            max_topics (int): Maximum aantal topics
            
        Returns:
            List[Dict]: Lijst met topic informatie
        """
        if not self.client:
            print("❌ Anthropic client niet beschikbaar")
            return []
        
        print("🧠 Claude analyseert alle vragen om topics te genereren...")
        
        # Neem een sample van vragen voor context (max 200 voor token limits)
        sample_questions = self.questions[:200] if len(self.questions) > 200 else self.questions
        
        # Maak een prompt voor topic generatie
        questions_text = "\n".join([f"- {q}" for q in sample_questions])
        
        prompt = f"""
Je bent een expert in Nederlandse tekstanalyse. Analyseer de volgende lijst van {len(sample_questions)} Nederlandse vragen en genereer {max_topics} betekenisvolle topic categorieën.

VRAGEN:
{questions_text}

INSTRUCTIES:
1. Analyseer de vragen en identificeer de hoofdthema's
2. Genereer precies {max_topics} topics die de vragen het best categoriseren
3. Geef elke topic een korte, beschrijvende Nederlandse naam
4. Geef een uitgebreide beschrijving van wat elk topic omvat
5. Zorg ervoor dat de topics elkaar niet overlappen
6. Maak topics die logisch en intuïtief zijn voor Nederlandse gebruikers

ANTWOORD FORMAT (JSON):
{{
    "topics": [
        {{
            "id": 1,
            "name": "Korte topic naam",
            "description": "Uitgebreide beschrijving van wat dit topic omvat",
            "keywords": ["sleutelwoord1", "sleutelwoord2", "sleutelwoord3"]
        }},
        ...
    ]
}}

Zorg ervoor dat je antwoord geldige JSON is zonder extra tekst ervoor of erna.
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse de response
            response_text = response.content[0].text
            
            # Probeer JSON te extraheren
            try:
                # Zoek naar JSON in de response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                    result = json.loads(json_text)
                    self.topics = result.get('topics', [])
                    print(f"✅ {len(self.topics)} topics gegenereerd door Claude")
                    return self.topics
                else:
                    print("❌ Geen geldige JSON gevonden in Claude response")
                    print(f"Response was: {response_text}")
                    return []
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse fout: {e}")
                print(f"Response was: {response_text}")
                return []
                
        except Exception as e:
            print(f"❌ Claude API fout: {e}")
            return []
    
    def classify_questions_with_claude(self, batch_size=15) -> Dict:
        """
        Classificeer alle vragen in de gegenereerde topics met Claude
        
        Args:
            batch_size (int): Aantal vragen per API call
            
        Returns:
            Dict: Mapping van vraag naar topic
        """
        if not self.client or not self.topics:
            print("❌ Claude client of topics niet beschikbaar")
            return {}
        
        print("🎯 Claude classificeert vragen naar topics...")
        
        # Maak topic referentie voor de prompt
        topics_text = "\n".join([
            f"{topic['id']}. {topic['name']}: {topic['description']}"
            for topic in self.topics
        ])
        
        classifications = {}
        
        # Verwerk vragen in batches
        for i in range(0, len(self.questions), batch_size):
            batch_questions = self.questions[i:i + batch_size]
            
            # Maak prompt voor classificatie
            questions_text = "\n".join([f"{j+1}. {q}" for j, q in enumerate(batch_questions)])
            
            prompt = f"""
Je bent een expert in Nederlandse tekstanalyse. Classificeer de volgende vragen naar de juiste topic categorieën.

TOPICS:
{topics_text}

VRAGEN OM TE CLASSIFICEREN:
{questions_text}

INSTRUCTIES:
1. Lees elke vraag zorgvuldig
2. Bepaal welk topic het best past bij elke vraag
3. Geef voor elke vraag het topic ID (1-{len(self.topics)})
4. Als een vraag niet goed past bij een topic, kies dan het meest verwante topic
5. Wees consistent en logisch in je classificaties

ANTWOORD FORMAT (JSON):
{{
    "classifications": {{
        "1": topic_id,
        "2": topic_id,
        ...
    }}
}}

Zorg ervoor dat je antwoord geldige JSON is zonder extra tekst ervoor of erna.
"""
            
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response_text = response.content[0].text
                
                # Parse JSON response
                try:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group()
                        result = json.loads(json_text)
                        batch_classifications = result.get('classifications', {})
                        
                        # Map terug naar echte vragen
                        for q_idx, topic_id in batch_classifications.items():
                            try:
                                question_index = int(q_idx) - 1
                                if 0 <= question_index < len(batch_questions):
                                    question = batch_questions[question_index]
                                    classifications[question] = int(topic_id)
                            except (ValueError, IndexError):
                                continue
                                
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse fout in batch {i//batch_size + 1}: {e}")
                    print(f"Response was: {response_text}")
                    continue
                    
            except Exception as e:
                print(f"❌ Claude API fout in batch {i//batch_size + 1}: {e}")
                continue
                
            print(f"✅ Batch {i//batch_size + 1}/{(len(self.questions) + batch_size - 1)//batch_size} verwerkt")
        
        self.question_classifications = classifications
        print(f"✅ {len(classifications)} vragen geclassificeerd")
        return classifications
    
    def analyze_results(self) -> Dict:
        """
        Analyseer de resultaten en genereer statistieken
        
        Returns:
            Dict: Analyse resultaten
        """
        if not self.topics or not self.question_classifications:
            print("❌ Geen topics of classificaties beschikbaar")
            return {}
        
        print("📊 Resultaten analyseren...")
        
        # Tel vragen per topic
        topic_counts = defaultdict(int)
        topic_questions = defaultdict(list)
        
        for question, topic_id in self.question_classifications.items():
            topic_counts[topic_id] += 1
            topic_questions[topic_id].append(question)
        
        # Maak resultaten dictionary
        results = {
            'topics': self.topics,
            'topic_counts': dict(topic_counts),
            'topic_questions': dict(topic_questions),
            'total_questions': len(self.question_classifications),
            'num_topics': len(self.topics),
            'classification_rate': len(self.question_classifications) / len(self.questions) * 100
        }
        
        return results
    
    def visualize_results(self, results: Dict):
        """
        Visualiseer de resultaten
        
        Args:
            results (Dict): Analyse resultaten
        """
        print("📈 Visualisaties maken...")
        
        # Bereid data voor
        topic_names = [topic['name'] for topic in results['topics']]
        topic_counts = [results['topic_counts'].get(topic['id'], 0) for topic in results['topics']]
        
        # Maak plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Bar chart van topic groottes
        bars = ax1.bar(range(len(topic_names)), topic_counts, color='lightcoral', edgecolor='darkred')
        ax1.set_title('Aantal Vragen per Topic (Claude Analyse)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Topics')
        ax1.set_ylabel('Aantal Vragen')
        ax1.set_xticks(range(len(topic_names)))
        ax1.set_xticklabels([f"T{i+1}" for i in range(len(topic_names))], rotation=45)
        
        # Voeg waarden toe aan bars
        for bar, count in zip(bars, topic_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Pie chart van topic distributie
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_names)))
        wedges, texts, autotexts = ax2.pie(topic_counts, labels=[f"T{i+1}" for i in range(len(topic_names))], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Topic Distributie (Claude Analyse)', fontsize=14, fontweight='bold')
        
        # Voeg legenda toe
        ax2.legend(wedges, [f"T{i+1}: {name[:20]}..." if len(name) > 20 else f"T{i+1}: {name}" 
                           for i, name in enumerate(topic_names)], 
                  title="Topics", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig('claude_topic_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self, results: Dict):
        """
        Print de resultaten in een mooie format
        
        Args:
            results (Dict): Analyse resultaten
        """
        print("\n" + "="*70)
        print("🤖 CLAUDE-GEBASEERDE NEDERLANDSE TOPIC ANALYSE RESULTATEN")
        print("="*70)
        
        print(f"\n📈 Totaal aantal vragen: {len(self.questions)}")
        print(f"🎯 Aantal gevonden topics: {results['num_topics']}")
        print(f"📊 Classificatie rate: {results['classification_rate']:.1f}%")
        print(f"📊 Gemiddeld aantal vragen per topic: {results['total_questions'] / results['num_topics']:.1f}")
        
        print("\n🧠 CLAUDE-GEGENEREERDE TOPICS:")
        print("-" * 70)
        
        for topic in results['topics']:
            topic_id = topic['id']
            count = results['topic_counts'].get(topic_id, 0)
            percentage = (count / results['total_questions']) * 100 if results['total_questions'] > 0 else 0
            
            print(f"\n🏷️  Topic {topic_id}: {topic['name']}")
            print(f"   📝 Beschrijving: {topic['description']}")
            print(f"   🔑 Keywords: {', '.join(topic['keywords'])}")
            print(f"   📊 Aantal vragen: {count} ({percentage:.1f}%)")
            
            if topic_id in results['topic_questions']:
                print(f"   💭 Voorbeelden:")
                examples = results['topic_questions'][topic_id][:3]
                for i, example in enumerate(examples):
                    if len(example) > 120:
                        example = example[:120] + "..."
                    print(f"      {i+1}. {example}")
        
        print("\n" + "="*70)
        print("✅ Claude-gebaseerde analyse voltooid!")
        print("="*70)
    
    def run_analysis(self, max_topics=8):
        """
        Voer de volledige Claude-gebaseerde analyse uit
        
        Args:
            max_topics (int): Maximum aantal topics om te genereren
        """
        print("🚀 Claude-gebaseerde Nederlandse Topic Analyse gestart...")
        
        # Stap 1: Laad data
        if not self.load_data():
            return None
        
        # Stap 2: Extraheer vragen
        if not self.extract_questions():
            print("❌ Geen vragen gevonden om te analyseren")
            return None
        
        # Stap 3: Genereer topics met Claude
        topics = self.generate_topics_with_claude(max_topics)
        if not topics:
            print("❌ Kon geen topics genereren")
            return None
        
        # Stap 4: Classificeer vragen met Claude
        classifications = self.classify_questions_with_claude()
        if not classifications:
            print("❌ Kon vragen niet classificeren")
            return None
        
        # Stap 5: Analyseer resultaten
        results = self.analyze_results()
        
        # Stap 6: Visualiseer resultaten
        self.visualize_results(results)
        
        # Stap 7: Print resultaten
        self.print_results(results)
        
        return results

# Hoofdscript
if __name__ == "__main__":
    # Controleer of Anthropic API key is ingesteld
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  ANTHROPIC_API_KEY environment variable niet gevonden!")
        print("💡 Stel je API key in met: export ANTHROPIC_API_KEY='your-api-key'")
        print("💡 Of voeg deze toe aan je .env bestand")
        exit(1)
    
    # Initialiseer de Claude-gebaseerde analyzer
    analyzer = ClaudeTopicAnalyzer("replit_db_backup_08_07_2025_trail.xlsx")
    
    # Voer analyse uit
    print("🤖 Claude-gebaseerde analyse - dit kan even duren...")
    results = analyzer.run_analysis(max_topics=8)
    
    if results:
        print("\n🎉 Klaar! Check de 'claude_topic_analysis_results.png' voor visualisaties.")
    else:
        print("\n❌ Analyse mislukt. Controleer je API key en data.") 