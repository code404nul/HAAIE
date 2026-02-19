import json
from pathlib import Path
from typing import Any, Dict


class ModelManager:
    """ModelManager.available_models to get current model"""

    _MODEL_PATHS = {
        "epsilon": "resources/v2/haru/haru.model.json",
        "haru": "resources/v3/Haru/Haru.model3.json",
        "hibiki": "resources/v2/hibiki/hibiki.model.json",
        "kana": "resources/v2/kana/Kobayaxi.model.json",
        "kasumi2": "resources/v2/kasumi2/kasumi2.model.json",
        "shizuku": "resources/v2/shizuku/shizuku.model.json",
        "zer0": "resources/v2/托尔/model0.json",
        "llny": "resources/v3/llny/llny.model3.json",
        "mao": "resources/v3/Mao/Mao.model3.json",
        "nn": "resources/v3/nn/nn.model3.json",
        "ahiru": "resources/v3/ahiru/ahiru.model3.json",
        "keserannpasarann": "resources/v3/keserannpasarann/keserannpasarann.model3.json",
    }
    
    @classmethod
    def available_models(cls) -> str: return cls._MODEL_PATHS.keys()

    def __init__(self, name: str = "keserannpasarann") -> None:
        """
        La liste des noms de modèles disponible -> ModelManager.available_models()
        Var : 
        - path
        - name
        - expressions
        - motions
        
        Remarque : 
            Vous pouvez ajouter votre propre modèle en ajoutant une entrée
            dans le dictionnaire `_MODEL_PATHS`.
        """
        
        if name not in self._MODEL_PATHS:
            raise ValueError(
                f"Modèle inconnu : '{name}'. "
                f"Modèles disponibles : {', '.join(self._MODEL_PATHS)}"
            )

        self.name = name
        self.path = Path(self._MODEL_PATHS[name])

        if not self.path.is_file():
            raise FileNotFoundError(f"Fichier modèle introuvable : {self.path}")

        with self.path.open(encoding="utf-8") as f:
            self.model_json = json.load(f)

        self.expressions = self.extract_nested("FileReferences.Expressions")
        self.motions = self.extract_nested("FileReferences.Motions")

    def __repr__(self) -> str:  return f"<ModelManager name='{self.name}' path='{self.path}'>"
    
    def extract(self, *keys: str) -> Dict[str, Any]: return {k: self.model_json.get(k) for k in keys}

    def extract_nested(self, path: str, default: Any = None) -> Any:
        current = self.model_json
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current