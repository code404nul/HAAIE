import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import pickle

class MultilingualFAISSMatcher:
    """
    Système de matching de phrases multilingue avec FAISS
    Supporte français et anglais avec des modèles optimisés
    """
    
    def __init__(self):
        # Modèles multilingues optimisés
        self.models = {
            'fr': SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
            'en': SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        }
        
        self.indexes = {'fr': {}, 'en': {}}  # Index FAISS par clé et langue
        self.metadata = {'fr': {}, 'en': {}}  # Métadonnées (phrases originales)
        
    def vectorize_and_build_indexes(self, data: Dict, save_path: str = None):
        """
        Vectorise toutes les phrases et construit les index FAISS
        
        Args:
            data: Dict au format {"clé": {"fr": [phrases], "en": [phrases]}}
            save_path: Chemin pour sauvegarder les index (optionnel)
        """
        print("🔄 Vectorisation en cours...")
        
        for key, languages in data.items():
            for lang in ['fr', 'en']:
                if lang not in languages or not languages[lang]:
                    continue
                
                phrases = languages[lang]
                print(f"  Traitement: {key} ({lang}) - {len(phrases)} phrases")
                
                # Vectorisation des phrases
                embeddings = self.models[lang].encode(
                    phrases, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Normalisation pour utiliser la similarité cosinus
                faiss.normalize_L2(embeddings)
                
                # Création de l'index FAISS
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner Product (cosinus après normalisation)
                index.add(embeddings)
                
                # Sauvegarde
                self.indexes[lang][key] = index
                self.metadata[lang][key] = phrases
        
        print("✅ Vectorisation terminée!")
        
        # Sauvegarde optionnelle
        if save_path:
            self.save_indexes(save_path)
            
        return self
    
    def save_indexes(self, path: str):
        """Sauvegarde les index et métadonnées dans un fichier"""
        save_data = {
            'indexes': {},
            'metadata': self.metadata
        }
        
        # Sauvegarder les index FAISS en bytes
        for lang in ['fr', 'en']:
            save_data['indexes'][lang] = {}
            for key, index in self.indexes[lang].items():
                save_data['indexes'][lang][key] = faiss.serialize_index(index)
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Index sauvegardés dans: {path}")
    
    def load_indexes(self, path: str):
        """Charge les index depuis un fichier"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restaurer les métadonnées
        self.metadata = save_data['metadata']
        
        # Restaurer les index FAISS
        for lang in ['fr', 'en']:
            self.indexes[lang] = {}
            for key, index_bytes in save_data['indexes'][lang].items():
                self.indexes[lang][key] = faiss.deserialize_index(index_bytes)
        
        print(f"📂 Index chargés depuis: {path}")
        return self
    
    def analyze_phrase(self, phrase: str, key: str, lang: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Analyse une phrase et retourne les similarités avec une clé donnée
        
        Args:
            phrase: La phrase à analyser
            key: La clé du JSON à comparer
            lang: 'fr' ou 'en'
            top_k: Nombre de meilleurs résultats à retourner
            
        Returns:
            Liste de tuples (phrase_correspondante, score_similarité)
            Score entre 0 et 1, où 1 = correspondance parfaite
        """
        if lang not in self.indexes or key not in self.indexes[lang]:
            return []
        
        # Vectoriser la phrase
        embedding = self.models[lang].encode([phrase], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        
        # Recherche dans l'index
        index = self.indexes[lang][key]
        scores, indices = index.search(embedding, min(top_k, index.ntotal))
        
        # Récupérer les phrases correspondantes
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # FAISS retourne -1 si pas assez de résultats
                matched_phrase = self.metadata[lang][key][idx]
                # Score entre 0 et 1 (similarité cosinus normalisée)
                similarity = float(max(0, min(1, (score + 1) / 2)))
                results.append((matched_phrase, similarity))
        
        return results
    
    def get_best_match_score(self, phrase: str, key: str, lang: str) -> float:
        """
        Retourne le meilleur score de similarité pour une phrase donnée
        
        Returns:
            Score entre 0 et 1
        """
        results = self.analyze_phrase(phrase, key, lang, top_k=1)
        return results[0][1] if results else 0.0
    
    def get_category_score(self, phrase: str, key: str, lang: str, method: str = 'mean_top5') -> float:
        """
        Retourne un score d'appartenance à la catégorie (pas juste une phrase)
        
        Args:
            phrase: La phrase à analyser
            key: La clé/catégorie à évaluer
            lang: 'fr' ou 'en'
            method: Méthode de calcul
                - 'mean_top5': Moyenne des 5 meilleurs scores (recommandé)
                - 'mean_top3': Moyenne des 3 meilleurs scores
                - 'weighted': Moyenne pondérée (meilleurs scores = plus de poids)
                - 'max': Meilleur score uniquement
        
        Returns:
            Score entre 0 et 1 indiquant l'appartenance à la catégorie
        """
        if lang not in self.indexes or key not in self.indexes[lang]:
            return 0.0
        
        # Récupérer plusieurs résultats pour évaluer la catégorie
        top_k = 10
        results = self.analyze_phrase(phrase, key, lang, top_k=top_k)
        
        if not results:
            return 0.0
        
        scores = [score for _, score in results]
        
        weights = np.array([1.0 - (i * 0.1) for i in range(len(scores))])
        weights = weights[:len(scores)]
        return float(np.average(scores, weights=weights))

        
class Analyse_msg():
    def __init__(self):
        self.KEYS = ("voc_relational", "AI_implication", "request_opinion", "humanization", "comparaison", "self_disclosure")

        self.matcher = MultilingualFAISSMatcher()
        self.matcher.load_indexes('utils/dependency/faiss_indexes.pkl')

    def give_score(self, msg, language="en"):
        scores = {}
        for key in self.KEYS:
            scores[key] = self.matcher.get_category_score(msg, key, language)
        return scores
    
    def overall_score(self, msg, language="en", lambda1=[]):
        scores = self.give_score(msg, language)
        
        return (
            scores["voc_relational"] * scores["AI_implication"] * lambda1[0] + 
            scores["request_opinion"] * lambda1[1] + 
            scores["humanization"] * lambda1[2] + 
            scores["AI_implication"] * lambda1[3] * scores["comparaison"] + 
            scores["self_disclosure"] * lambda1[4]
        )


if __name__ == "__main__":
    test = Analyse_msg()
    
    # Test avec une phrase française
    phrase = "Tu sais je t'aime et je tiens vraiment a toi"
    score = test.overall_score(phrase, "fr", [1]*5)
    print(f"Score global: {score}")
    