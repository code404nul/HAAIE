import os
from pathlib import Path
from typing import List, Dict, Optional
from llama_cpp import Llama
import atexit

class GemmaGGUFChat:
    def __init__(
        self, 
        model_path: str = "models/gemma-3-12b-it-q4_0.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int = None
    ):
        """
        Initialiser le chat avec le modèle Gemma GGUF.
        
        Args:
            model_path: Chemin vers le fichier GGUF
            n_ctx: Taille du contexte
            n_gpu_layers: Nombre de couches sur GPU (-1 = toutes)
            n_threads: Nombre de threads CPU (None = auto)
        """
        # Vérifier si le fichier existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle '{model_path}' n'existe pas.")
        
        print(f"⏳ Chargement du modèle: {model_path}")
        
        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format="gemma",
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
        
        self.chat_history: List[Dict[str, str]] = []
        self._closed = False
        
        # Enregistrer la fermeture propre à la sortie
        atexit.register(self._cleanup)
        
        print("✓ Modèle chargé avec succès!\n")

    def _cleanup(self):
        """Nettoyer les ressources proprement."""
        if not self._closed and hasattr(self, 'llm'):
            try:
                if hasattr(self.llm, 'close'):
                    self.llm.close()
                self._closed = True
            except Exception:
                # Ignorer les erreurs lors du nettoyage
                pass

    def __del__(self):
        """Destructeur pour nettoyer les ressources."""
        self._cleanup()

    def __enter__(self):
        """Support du context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fermeture propre avec context manager."""
        self._cleanup()
        return False

    def add_message(self, role: str, content: str):
        """Ajouter un message à l'historique."""
        self.chat_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Effacer tout l'historique."""
        self.chat_history = []
        print("✓ Historique effacé.\n")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Obtenir l'historique actuel."""
        return self.chat_history.copy()
    
    def count_tokens(self, text: str) -> int:
        """Compter le nombre de tokens dans un texte."""
        if self._closed:
            raise RuntimeError("Le modèle a été fermé.")
        return len(self.llm.tokenize(text.encode()))
    
    def generate_response(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        use_history: bool = True,
        stream: bool = False
    ) -> str:
        """
        Générer une réponse au message utilisateur.
        
        Args:
            user_message: Message de l'utilisateur
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température d'échantillonnage (0.0-2.0)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repeat_penalty: Pénalité de répétition
            use_history: Utiliser l'historique de chat
            stream: Afficher la réponse en streaming
        
        Returns:
            Texte de la réponse générée
        """
        if self._closed:
            raise RuntimeError("Le modèle a été fermé.")
        
        # Ajouter le message utilisateur à l'historique
        self.add_message("user", user_message)
        
        # Préparer les messages
        if use_history:
            messages = self.chat_history.copy()
        else:
            messages = [{"role": "user", "content": user_message}]
        
        try:
            # Générer la réponse
            if stream:
                # Mode streaming
                response_text = ""
                for chunk in self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stream=True
                ):
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        content = delta['content']
                        print(content, end='', flush=True)
                        response_text += content
                print()  # Nouvelle ligne après le streaming
            else:
                # Mode normal
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty
                )
                response_text = response['choices'][0]['message']['content']
        
        except Exception as e:
            # Retirer le dernier message utilisateur de l'historique en cas d'erreur
            if self.chat_history and self.chat_history[-1]['role'] == 'user':
                self.chat_history.pop()
            raise RuntimeError(f"Erreur lors de la génération: {e}")
        
        # Ajouter la réponse à l'historique
        self.add_message("assistant", response_text)
        
        return response_text
    
    def print_history(self):
        """Afficher l'historique formaté."""
        print("\n" + "="*60)
        print("HISTORIQUE DE LA CONVERSATION")
        print("="*60)
        for i, msg in enumerate(self.chat_history, 1):
            role = "Vous" if msg['role'] == "user" else "Gemma"
            print(f"\n[{i}] {role}:")
            print(msg['content'])
        print("\n" + "="*60 + "\n")
    
    def chat_loop(self, stream: bool = True):
        """Boucle de chat interactive."""
        print("\n" + "="*60)
        print("Interface de Chat Gemma-3")
        print("="*60)
        print("Commandes:")
        print("  'quit' ou 'exit' : Quitter")
        print("  'clear'          : Effacer l'historique")
        print("  'history'        : Afficher l'historique")
        print("  'tokens <text>'  : Compter les tokens")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n🧑 Vous: ").strip()
                
                if not user_input:
                    continue
                
                # Commandes
                if user_input.lower() in ['quit', 'exit']:
                    print("\nAu revoir! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                if user_input.lower() == 'history':
                    self.print_history()
                    continue
                
                if user_input.lower().startswith('tokens '):
                    text = user_input[7:]
                    count = self.count_tokens(text)
                    print(f"Nombre de tokens: {count}")
                    continue
                
                # Générer et afficher la réponse
                print("\n🤖 Gemma: ", end="", flush=True)
                response = self.generate_response(
                    user_input,
                    stream=stream
                )
                
                if not stream:
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\nAu revoir! 👋")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}\n")


def main():
    """Fonction principale."""
    # Initialiser le chat avec context manager pour une fermeture propre
    try:
        with GemmaGGUFChat(
            model_path="gemma3:12b-it-qat",
            n_ctx=8192,
            n_gpu_layers=-1  # Utiliser GPU si disponible
        ) as chat:
            # Exemple 1: Message simple sans historique
            print("--- Exemple 1: Message simple ---")
            response = chat.generate_response(
                "Bonjour! Qui es-tu?",
                use_history=False,
                stream=False
            )
            print(f"Réponse: {response}\n")
            
            # Effacer pour recommencer
            chat.clear_history()
            
            # Exemple 2: Conversation multi-tour avec historique
            print("--- Exemple 2: Conversation avec contexte ---")
            
            print("\n🧑 Vous: Je m'appelle Alice et j'adore la programmation Python.")
            print("🤖 Gemma: ", end="", flush=True)
            chat.generate_response(
                "Je m'appelle Alice et j'adore la programmation Python.",
                stream=True
            )
            
            print("\n🧑 Vous: Quel est mon nom?")
            print("🤖 Gemma: ", end="", flush=True)
            chat.generate_response(
                "Quel est mon nom?",
                stream=True
            )
            
            print("\n🧑 Vous: Qu'est-ce que j'aime?")
            print("🤖 Gemma: ", end="", flush=True)
            chat.generate_response(
                "Qu'est-ce que j'aime?",
                stream=True
            )
            
            # Afficher l'historique
            chat.print_history()
            
            # Lancer le chat interactif
            chat.clear_history()
            chat.chat_loop(stream=True)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nAssurez-vous que le fichier GGUF est dans le bon chemin.")
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()