import pandas as pd
import anthropic
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleTopicAnalyzer:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.df = None
        self.questions = []
        self.categories = []
        self.assignments = {}
        
        # Setup Claude client
        api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
        if not api_key:
            # Try to load from .env
            try:
                with open('../../.env', 'r') as f:
                    for line in f:
                        if 'CLAUDE_API_KEY' in line:
                            api_key = line.split('=', 1)[1].strip().strip('"\'')
                            break
            except:
                pass
        
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            print("❌ Geen Claude API key gevonden")
            self.client = None
    
    def load_data(self):
        """Load the Excel file and extract questions"""
        self.df = pd.read_excel(self.excel_file)
        print(f"📊 Dataset loaded: {len(self.df)} rows")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        # Extract questions from 'Human Message' column
        if 'Human Message' in self.df.columns:
            self.questions = self.df['Human Message'].dropna().astype(str).tolist()
        else:
            print("❌ 'Human Message' column not found")
            return False
        
        print(f"📝 Found {len(self.questions)} questions")
        return True
    
    def create_categories(self):
        """Ask Claude to create categories based on ALL questions"""
        print("🧠 Claude is creating categories from all questions...")
        
        # Prepare all questions for Claude
        questions_text = "\n".join([f"{i+1}. {q[:200]}..." if len(q) > 200 else f"{i+1}. {q}" 
                                   for i, q in enumerate(self.questions)])
        
        prompt = f"""
Je bent een expert in Nederlandse tekstanalyse. Hier zijn {len(self.questions)} Nederlandse vragen uit een dataset. 
Creëer 6-7 logische categorieën die deze vragen het best dekken.

VRAGEN:
{questions_text}

Geef je antwoord als JSON in deze format:
{{
    "categories": [
        {{
            "id": 1,
            "name": "Categorie Naam",
            "description": "Korte beschrijving"
        }},
        ...
    ]
}}
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON
            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_text = response_text[json_start:json_end]
            
            result = json.loads(json_text)
            self.categories = result['categories']
            
            # Add "Overige" category as the last one
            overige_id = len(self.categories) + 1
            self.categories.append({
                "id": overige_id,
                "name": "Overige",
                "description": "Vragen die niet passen in de andere categorieën"
            })
            
            print(f"✅ Created {len(self.categories)} categories:")
            for cat in self.categories:
                print(f"   {cat['id']}. {cat['name']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating categories: {e}")
            return False
    
    def assign_questions(self):
        """Assign each question to a category"""
        print("🎯 Assigning questions to categories...")
        
        # Prepare categories text
        categories_text = "\n".join([f"{cat['id']}. {cat['name']}: {cat['description']}" 
                                   for cat in self.categories])
        
        for i, question in enumerate(self.questions):
            prompt = f"""
Categorieën:
{categories_text}

Vraag: {question}

Geef alleen het categorie nummer (1-{len(self.categories)}) dat het best bij deze vraag past. Geef alleen het getal, geen tekst:
"""
            
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract category number
                response_text = response.content[0].text.strip()
                # Extract just the number (in case Claude returns "1 (Category Name)")
                import re
                match = re.search(r'(\d+)', response_text)
                if match:
                    category_id = int(match.group(1))
                    self.assignments[question] = category_id
                else:
                    print(f"❌ Could not parse category for question {i+1}: {response_text}")
                    continue
                
                if (i + 1) % 20 == 0:
                    print(f"   Processed {i + 1}/{len(self.questions)} questions")
                    
            except Exception as e:
                print(f"❌ Error assigning question {i+1}: {e}")
                continue
        
        print(f"✅ Assigned {len(self.assignments)} questions")
        return True
    
    def analyze_results(self):
        """Analyze and display results"""
        print("\n" + "="*60)
        print("📊 TOPIC ANALYSIS RESULTS")
        print("="*60)
        
        # Count questions per category
        category_counts = defaultdict(int)
        for question, category_id in self.assignments.items():
            category_counts[category_id] += 1
        
        print(f"\n📈 Total questions: {len(self.assignments)}")
        print(f"🎯 Categories: {len(self.categories)}")
        
        print("\n📋 RESULTS:")
        print("-" * 60)
        
        for category in self.categories:
            cat_id = category['id']
            count = category_counts[cat_id]
            percentage = (count / len(self.assignments)) * 100 if self.assignments else 0
            
            print(f"\n{cat_id}. {category['name']}")
            print(f"   📊 {count} questions ({percentage:.1f}%)")
            print(f"   📝 {category['description']}")
            
            # Show examples
            examples = [q for q, c in self.assignments.items() if c == cat_id][:3]
            if examples:
                print("   💭 Examples:")
                for j, example in enumerate(examples):
                    short_example = example[:100] + "..." if len(example) > 100 else example
                    print(f"      {j+1}. {short_example}")
        
        print("\n" + "="*60)
        print("✅ Analysis complete!")
        print("="*60)
        
        return category_counts
    
    def create_visualization(self, category_counts):
        """Create bar chart and pie chart visualizations of the results"""
        print("📊 Creating visualization...")
        
        # Prepare data for visualization
        category_names = [cat['name'] for cat in self.categories]
        counts = [category_counts[cat['id']] for cat in self.categories]
        
        # Create subplots: bar chart and pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar chart
        bars = ax1.bar(category_names, counts, color='steelblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Nederlandse Vragen per Topic Categorie', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Categorieën', fontsize=12)
        ax1.set_ylabel('Aantal Vragen', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        colors = plt.cm.Set3(range(len(category_names)))
        wedges, texts, autotexts = ax2.pie(counts, labels=category_names, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        
        # Improve pie chart readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.set_title('Topic Distributie (Percentages)', fontsize=14, fontweight='bold')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('topic_analysis_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'topic_analysis_visualization.png' (bar chart + pie chart)")
        plt.show()
    
    def export_to_excel(self):
        """Export results to Excel with category assignments"""
        print("📁 Exporting results to Excel...")
        
        # Create a mapping from category ID to category name
        category_mapping = {cat['id']: cat['name'] for cat in self.categories}
        
        # Add category columns to the dataframe
        category_ids = []
        category_names = []
        
        for _, row in self.df.iterrows():
            question = str(row['Human Message'])
            if question in self.assignments:
                cat_id = self.assignments[question]
                category_ids.append(cat_id)
                category_names.append(category_mapping[cat_id])
            else:
                category_ids.append(None)
                category_names.append(None)
        
        # Add new columns
        self.df['Categorie_ID'] = category_ids
        self.df['Categorie_Naam'] = category_names
        
        # Save to new Excel file
        output_filename = 'topic_analysis_results.xlsx'
        self.df.to_excel(output_filename, index=False)
        
        print(f"✅ Results exported to '{output_filename}'")
        print(f"📊 Added columns: 'Categorie_ID' and 'Categorie_Naam'")
        
        return output_filename
    
    def run(self):
        """Run the complete analysis"""
        print("🚀 Starting Simple Topic Analysis...")
        
        if not self.client:
            print("❌ No Claude client available")
            return
        
        if not self.load_data():
            return
        
        if not self.create_categories():
            return
        
        if not self.assign_questions():
            return
        
        category_counts = self.analyze_results()
        
        # Create visualization
        self.create_visualization(category_counts)
        
        # Export to Excel
        self.export_to_excel()

if __name__ == "__main__":
    analyzer = SimpleTopicAnalyzer("replit_db_backup_08_07_2025_trail.xlsx")
    analyzer.run() 