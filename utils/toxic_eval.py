import torch
from detoxify import Detoxify
from typing import Dict, List, Union
import pandas as pd


class MultilingualToxicityEvaluator:
    """
    Évaluateur de toxicité multilingue avec support de modèles locaux.
    Supporte 7 langues: English, French, Spanish, Italian, Portuguese, Turkish, Russian
    
    Catégories de toxicité:
    - toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
    """
    
    def __init__(self, model_type: str = "multilingual"):
        """
        Initialize the toxicity evaluator.
        
        Args:
            model_type: Type of model ("multilingual", "original", or "unbiased")
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Running in OFFLINE mode - using cached models only")
        print(f"Loading Detoxify model: {model_type}")
        
        try:
            # Force local_files_only in Detoxify
            self.model = Detoxify(model_type, device=self.device)
            print(f"Model loaded successfully on {self.device}!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTROUBLESHOOTING:")
            print("1. Make sure you've downloaded the model while online first:")
            print(f"   python -c \"from detoxify import Detoxify; Detoxify('{model_type}')\"")
            print("2. Check if the model cache exists:")
            print(f"   ls -la ~/.cache/huggingface/hub/")
            print("3. Try clearing the HF_HUB_OFFLINE flag temporarily to download models")
            raise
        
        # Catégories de toxicité
        self.categories = ["toxicity", "severe_toxicity", "obscene", 
                          "threat", "insult", "identity_attack", "sexual_explicit"]

    def evaluate(self, text: Union[str, List[str]], threshold: float = 0.5) -> Union[Dict, List[Dict]]:
        """
        Évalue la toxicité d'un ou plusieurs textes.
        
        Args:
            text: Texte unique ou liste de textes à évaluer
            threshold: Seuil pour considérer un contenu comme toxique
            
        Returns:
            Dictionnaire ou liste de dictionnaires avec les scores
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        predictions = self.model.predict(texts)
        
        results = []
        for i, txt in enumerate(texts):
            result = {
                "text": txt,
                "scores": {}
            }
            
            for category in self.categories:
                if isinstance(predictions[category], list):
                    score = float(predictions[category][i])
                else:
                    score = float(predictions[category])
                result["scores"][category] = round(score, 4)
            
            result["is_toxic"] = result["scores"]["toxicity"] >= threshold
            result["toxicity_score"] = result["scores"]["toxicity"]
            result["max_category"] = max(result["scores"].items(), key=lambda x: x[1])
            
            results.append(result)
        
        return results[0] if is_single else results
    
    def batch_evaluate(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """Évalue un lot de textes."""
        return self.evaluate(texts, threshold)
    
    def get_detailed_report(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Génère un rapport détaillé pour un texte.
        
        Returns:
            Dictionnaire avec analyse complète incluant niveau de sévérité
        """
        result = self.evaluate(text, threshold)
        
        flagged_categories = [cat for cat, score in result["scores"].items() 
                            if score >= threshold]
        
        report = {
            "text": text,
            "overall_toxic": result["is_toxic"],
            "toxicity_score": result["toxicity_score"],
            "all_scores": result["scores"],
            "flagged_categories": flagged_categories,
            "highest_score": {
                "category": result["max_category"][0],
                "score": result["max_category"][1]
            },
            "severity_level": self._get_severity_level(result["toxicity_score"])
        }
        
        return report
    
    def _get_severity_level(self, score: float) -> str:
        """Détermine le niveau de sévérité basé sur le score."""
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Medium"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def compare_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Compare plusieurs textes et retourne un DataFrame.
        
        Returns:
            DataFrame avec les scores de tous les textes
        """
        results = self.batch_evaluate(texts)
        
        data = []
        for r in results:
            row = {
                "text": r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"],
                "is_toxic": r["is_toxic"],
                **r["scores"]
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def filter_toxic_content(self, texts: Union[str, List[str]], 
                           threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Filtre le contenu toxique du contenu non-toxique.
        
        Args:
            texts: Texte unique ou liste de textes
            threshold: Seuil de toxicité
            
        Returns:
            Dictionnaire avec clés "toxic" et "non_toxic"
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = self.batch_evaluate(texts, threshold)
        
        filtered = {
            "toxic": [],
            "non_toxic": []
        }
        
        for r in results:
            if r["is_toxic"]:
                filtered["toxic"].append(r["text"])
            else:
                filtered["non_toxic"].append(r["text"])
        
        return filtered


if __name__ == "__main__":
    print("=" * 70)
    print("Multilingual Toxicity Evaluator - Demo")
    print("=" * 70)
    
    try:
        evaluator = MultilingualToxicityEvaluator(model_type="multilingual")
        
        test_texts = [
            "I love this product! It's amazing!",  # English - non-toxic
            "You are stupid and worthless, idiot!",  # English - toxic
            "Je déteste ce produit, mais le service était bon.",  # French
            "Va te faire foutre, espèce d'imbécile!",  # French - toxic
            "Eres un idiota y te odio.",  # Spanish - toxic
            "Questo è un ottimo ristorante!",  # Italian - non-toxic
            "Você é incrível!",  # Portuguese - non-toxic
            "This is complete garbage and you should be ashamed.",  # English - toxic
        ]
        
        print("\n" + "=" * 70)
        print("Filtering toxic content...")
        print("=" * 70)
        
        filtered = evaluator.filter_toxic_content(test_texts)
        
        print(f"\n✗ TOXIC CONTENT ({len(filtered['toxic'])} items):")
        for txt in filtered['toxic']:
            print(f"  - {txt}")
        
        print(f"\n✓ NON-TOXIC CONTENT ({len(filtered['non_toxic'])} items):")
        for txt in filtered['non_toxic']:
            print(f"  - {txt}")
        
        print("\n" + "=" * 70)
        print("Detailed analysis of first toxic comment:")
        print("=" * 70)
        
        if filtered['toxic']:
            report = evaluator.get_detailed_report(filtered['toxic'][0])
            print(f"\nText: {report['text']}")
            print(f"Overall Toxic: {report['overall_toxic']}")
            print(f"Toxicity Score: {report['toxicity_score']:.4f}")
            print(f"Severity Level: {report['severity_level']}")
            print(f"\nFlagged Categories: {', '.join(report['flagged_categories'])}")
            print(f"\nHighest Score: {report['highest_score']['category']} "
                  f"({report['highest_score']['score']:.4f})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to download models while online first:")
        print("python -c \"from detoxify import Detoxify; Detoxify('multilingual')\"")