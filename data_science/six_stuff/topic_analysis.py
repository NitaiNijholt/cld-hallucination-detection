import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

class DutchTopicAnalyzer:
    def __init__(self, excel_file_path):
        """
        Initialiseer de Nederlandse topic analyzer
        
        Args:
            excel_file_path (str): Pad naar het Excel bestand
        """
        self.excel_file_path = excel_file_path
        self.df = None
        self.embeddings = None
        self.clusters = None
        self.model = None
        self.topic_labels = {}
        
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
    
    def preprocess_text(self, text):
        """
        Preprocessing van Nederlandse tekst
        
        Args:
            text (str): Ruwe tekst
            
        Returns:
            str: Schone tekst
        """
        if pd.isna(text):
            return ""
        
        # Basis text cleaning
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)  # Meerdere spaties naar één spatie
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Behoud alleen letters, cijfers en basis punctuatie
        
        return text
    
    def extract_questions_and_content(self):
        """
        Identificeer en extraheer vragen en relevante content uit de dataset
        """
        # Probeer verschillende kolommen te identificeren die vragen bevatten
        potential_question_columns = []
        
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['question', 'vraag', 'msg', 'message', 'content', 'text']):
                potential_question_columns.append(col)
        
        print(f"🔍 Potentiële vraag kolommen: {potential_question_columns}")
        
        # Als geen duidelijke kolommen, neem de kolom met de meeste tekst
        if not potential_question_columns:
            text_lengths = {}
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    avg_length = self.df[col].astype(str).str.len().mean()
                    text_lengths[col] = avg_length
            
            if text_lengths:
                main_text_col = max(text_lengths, key=text_lengths.get)
                potential_question_columns = [main_text_col]
                print(f"📝 Hoofdtekst kolom gekozen: {main_text_col}")
        
        # Combineer alle relevante tekst kolommen
        all_text = []
        for col in potential_question_columns:
            if col in self.df.columns:
                all_text.extend(self.df[col].dropna().astype(str).tolist())
        
        # Preprocessing
        cleaned_text = [self.preprocess_text(text) for text in all_text]
        
        # Filter lege teksten en te korte teksten
        meaningful_text = [text for text in cleaned_text if len(text) > 10]
        
        print(f"📄 {len(meaningful_text)} betekenisvolle teksten gevonden")
        
        return meaningful_text
    
    def create_embeddings(self, texts):
        """
        Maak embeddings van de teksten met een meertalig model
        
        Args:
            texts (list): Lijst met teksten
            
        Returns:
            numpy.ndarray: Embeddings matrix
        """
        print("🧠 Embeddings maken...")
        
        # Gebruik een meertalig model dat goed werkt met Nederlands
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        
        try:
            self.model = SentenceTransformer(model_name)
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"✅ Embeddings gemaakt: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"❌ Fout bij embeddings: {e}")
            # Fallback naar een eenvoudiger model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = self.model.encode(texts, show_progress_bar=True)
                print(f"✅ Embeddings gemaakt met fallback model: {embeddings.shape}")
                return embeddings
            except Exception as e2:
                print(f"❌ Ook fallback model mislukt: {e2}")
                return None
    
    def find_optimal_clusters(self, embeddings, max_clusters=15):
        """
        Vind het optimale aantal clusters met elbow methode
        
        Args:
            embeddings (numpy.ndarray): Embeddings matrix
            max_clusters (int): Maximum aantal clusters om te testen
            
        Returns:
            int: Optimaal aantal clusters
        """
        print("🔍 Optimaal aantal clusters zoeken...")
        
        inertias = []
        K_range = range(2, min(max_clusters + 1, len(embeddings) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Zoek elbow punt
        if len(inertias) > 2:
            # Simpele elbow detectie
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2
                optimal_k = K_range[min(elbow_idx, len(K_range) - 1)]
            else:
                optimal_k = K_range[len(K_range) // 2]
        else:
            optimal_k = max(K_range) if K_range else 5
            
        print(f"📊 Optimaal aantal clusters: {optimal_k}")
        return optimal_k
    
    def perform_clustering(self, embeddings, n_clusters=None):
        """
        Voer clustering uit op de embeddings
        
        Args:
            embeddings (numpy.ndarray): Embeddings matrix
            n_clusters (int): Aantal clusters (None voor automatische detectie)
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(embeddings)
        
        print(f"🔄 Clustering uitvoeren met {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        print(f"✅ Clustering voltooid")
        return cluster_labels
    
    def generate_topic_labels(self, texts, cluster_labels):
        """
        Genereer betekenisvolle topic labels voor elke cluster
        
        Args:
            texts (list): Originele teksten
            cluster_labels (numpy.ndarray): Cluster labels
            
        Returns:
            dict: Mapping van cluster ID naar topic label
        """
        print("🏷️ Topic labels genereren...")
        
        topic_labels = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
            
            # Zoek meest voorkomende woorden in de cluster
            all_words = []
            for text in cluster_texts:
                # Simpele Nederlandse woord extractie
                words = re.findall(r'\b\w+\b', text.lower())
                # Filter stopwoorden (basis Nederlandse stopwoorden)
                dutch_stopwords = {'de', 'het', 'en', 'een', 'van', 'is', 'op', 'te', 'zijn', 'met', 'voor', 'dat', 'die', 'aan', 'als', 'door', 'kan', 'wordt', 'wat', 'we', 'je', 'heeft', 'maar', 'niet', 'werd', 'naar', 'ook', 'tot', 'er', 'bij', 'om', 'onder', 'zou', 'hem', 'tegen', 'na', 'twee', 'meer', 'geen', 'nog', 'waar', 'veel', 'hun', 'deze', 'andere', 'heb', 'had', 'hoe', 'was', 'dus', 'wel', 'zijn', 'haar', 'dan', 'nu', 'zo', 'uit', 'al', 'over', 'zij', 'ja', 'wie', 'zal', 'of', 'hier', 'moet', 'omdat', 'kunnen', 'hij', 'zich', 'zelf', 'want', 'wanneer', 'ging', 'kom', 'komt', 'tijdens', 'binnen', 'buiten', 'echter', 'terwijl', 'zowel', 'beide', 'namelijk', 'volgens', 'waarbij', 'waarom', 'waarbij', 'ieder', 'elk', 'elke', 'alle', 'alles', 'deze', 'dit', 'daar', 'deze', 'dat', 'den', 'der', 'des', 'dien', 'diens', 'dier', 'dezen', 'deze', 'die', 'dit', 'do', 'doe', 'doet', 'door', 'drie', 'du', 'dus', 'een', 'eens', 'eer', 'eerst', 'eerste', 'eigen', 'eind', 'elk', 'elke', 'en', 'enkel', 'er', 'erg', 'ergens', 'ge', 'geen', 'geweest', 'had', 'heb', 'hebben', 'heeft', 'heel', 'hem', 'hen', 'het', 'hier', 'hij', 'hun', 'ik', 'ieder', 'iemand', 'iets', 'ik', 'in', 'is', 'ja', 'jaar', 'je', 'jij', 'jou', 'jouw', 'jullie', 'kan', 'kon', 'kunnen', 'maar', 'mag', 'maken', 'me', 'meer', 'men', 'met', 'mij', 'mijn', 'moet', 'na', 'naar', 'niet', 'niets', 'nog', 'nu', 'of', 'om', 'omdat', 'onder', 'ons', 'ook', 'op', 'over', 'reeds', 'te', 'tegen', 'toch', 'toen', 'tot', 'u', 'uit', 'uw', 'van', 'veel', 'voor', 'want', 'waren', 'was', 'wat', 'we', 'wel', 'werd', 'wezen', 'waar', 'wie', 'wij', 'wil', 'worden', 'wordt', 'zal', 'ze', 'zei', 'zelf', 'zich', 'zij', 'zijn', 'zo', 'zonder', 'zou'}
                filtered_words = [word for word in words if word not in dutch_stopwords and len(word) > 2]
                all_words.extend(filtered_words)
            
            # Vind meest voorkomende woorden
            if all_words:
                word_counts = Counter(all_words)
                top_words = [word for word, count in word_counts.most_common(3)]
                topic_label = f"Topic {cluster_id + 1}: {' + '.join(top_words)}"
            else:
                topic_label = f"Topic {cluster_id + 1}"
            
            topic_labels[cluster_id] = topic_label
        
        return topic_labels
    
    def analyze_topics(self, texts, cluster_labels):
        """
        Analyseer de topics en tel vragen per topic
        
        Args:
            texts (list): Originele teksten
            cluster_labels (numpy.ndarray): Cluster labels
            
        Returns:
            dict: Topic analyse resultaten
        """
        print("📊 Topic analyse uitvoeren...")
        
        # Genereer topic labels
        topic_labels = self.generate_topic_labels(texts, cluster_labels)
        
        # Tel vragen per topic
        cluster_counts = Counter(cluster_labels)
        
        # Maak resultaten dictionary
        results = {
            'topic_labels': topic_labels,
            'question_counts': cluster_counts,
            'total_questions': len(texts),
            'num_topics': len(np.unique(cluster_labels))
        }
        
        # Voeg voorbeelden toe per topic
        results['examples'] = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
            # Neem max 3 voorbeelden per topic
            results['examples'][cluster_id] = cluster_texts[:3]
        
        return results
    
    def visualize_results(self, embeddings, cluster_labels, topic_labels):
        """
        Visualiseer de clustering resultaten
        
        Args:
            embeddings (numpy.ndarray): Embeddings matrix
            cluster_labels (numpy.ndarray): Cluster labels
            topic_labels (dict): Topic labels
        """
        print("📈 Visualisaties maken...")
        
        # PCA voor dimensie reductie
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Maak plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cluster visualisatie
        scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.6)
        ax1.set_title('Nederlandse Vraag Topics - Cluster Visualisatie')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        
        # Plot 2: Topic grootte
        topic_counts = Counter(cluster_labels)
        topics = [topic_labels[i] for i in sorted(topic_counts.keys())]
        counts = [topic_counts[i] for i in sorted(topic_counts.keys())]
        
        ax2.bar(range(len(topics)), counts)
        ax2.set_title('Aantal Vragen per Topic')
        ax2.set_xlabel('Topics')
        ax2.set_ylabel('Aantal Vragen')
        ax2.set_xticks(range(len(topics)))
        ax2.set_xticklabels([f"T{i+1}" for i in range(len(topics))], rotation=45)
        
        plt.tight_layout()
        plt.savefig('topic_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self, results):
        """
        Print de resultaten in een mooie format
        
        Args:
            results (dict): Analyse resultaten
        """
        print("\n" + "="*60)
        print("📊 NEDERLANDSE TOPIC ANALYSE RESULTATEN")
        print("="*60)
        
        print(f"\n📈 Totaal aantal vragen: {results['total_questions']}")
        print(f"🎯 Aantal gevonden topics: {results['num_topics']}")
        print(f"📊 Gemiddeld aantal vragen per topic: {results['total_questions'] / results['num_topics']:.1f}")
        
        print("\n🏷️ TOPIC OVERZICHT:")
        print("-" * 60)
        
        for cluster_id in sorted(results['question_counts'].keys()):
            topic_label = results['topic_labels'][cluster_id]
            count = results['question_counts'][cluster_id]
            percentage = (count / results['total_questions']) * 100
            
            print(f"\n{topic_label}")
            print(f"   📊 Aantal vragen: {count} ({percentage:.1f}%)")
            print(f"   📝 Voorbeelden:")
            
            for i, example in enumerate(results['examples'][cluster_id]):
                if len(example) > 100:
                    example = example[:100] + "..."
                print(f"      {i+1}. {example}")
        
        print("\n" + "="*60)
        print("✅ Analyse voltooid!")
        print("="*60)
    
    def run_analysis(self, n_clusters=None):
        """
        Voer de volledige analyse uit
        
        Args:
            n_clusters (int): Aantal clusters (None voor automatische detectie)
        """
        print("🚀 Nederlandse Topic Analyse gestart...")
        
        # Stap 1: Laad data
        if not self.load_data():
            return
        
        # Stap 2: Extraheer tekst
        texts = self.extract_questions_and_content()
        if not texts:
            print("❌ Geen teksten gevonden om te analyseren")
            return
        
        # Stap 3: Maak embeddings
        embeddings = self.create_embeddings(texts)
        if embeddings is None:
            return
        
        # Stap 4: Clustering
        cluster_labels = self.perform_clustering(embeddings, n_clusters)
        
        # Stap 5: Analyseer topics
        results = self.analyze_topics(texts, cluster_labels)
        
        # Stap 6: Visualiseer resultaten
        self.visualize_results(embeddings, cluster_labels, results['topic_labels'])
        
        # Stap 7: Print resultaten
        self.print_results(results)
        
        return results

# Hoofdscript
if __name__ == "__main__":
    # Initialiseer de analyzer
    analyzer = DutchTopicAnalyzer("replit_db_backup_08_07_2025_trail.xlsx")
    
    # Voer analyse uit
    results = analyzer.run_analysis()
    
    print("\n🎉 Klaar! Check de 'topic_analysis_results.png' voor visualisaties.") 