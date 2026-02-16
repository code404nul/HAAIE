import os
from typing import Union
from collections.abc import Generator
from typing import List, Dict
from llama_cpp import Llama
import atexit

class GemmaGGUFChat:
    def __init__(
        self, 
        model_path: str = "models/gemma-3-12b-it-q4_0.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int = None,
        n_batch: int = 1024,
        max_history_messages: int = 20
    ):
        """
        Initialiser le chat avec le modèle Gemma GGUF optimisé.
        
        Args:
            model_path: Chemin vers le fichier GGUF
            n_ctx: Taille du contexte
            n_gpu_layers: Nombre de couches sur GPU (-1 = toutes)
            n_threads: Nombre de threads CPU (None = auto)
            n_batch: Taille du batch pour prompt processing (plus grand = plus rapide)
            max_history_messages: Nombre max de messages à garder en historique
        """
        # Vérifier si le fichier existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle '{model_path}' n'existe pas.")
        
        print(f"⏳ Chargement du modèle: {model_path}")
        print(f"🔧 Optimisations: n_batch={n_batch}, GPU layers={n_gpu_layers}")
        
        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format="gemma",
                verbose=True,  # Afficher les stats de chargement GPU
                
                # OPTIMISATIONS CRITIQUES POUR VITESSE
                use_mmap=True,           # Memory mapping (déjà par défaut)
                use_mlock=True,          # Verrouiller en RAM (évite swap)
                n_batch=n_batch,         # Batch size pour prompt processing
                logits_all=False,        # Ne calculer que le dernier token
                embedding=False,         # Désactiver embeddings (non nécessaires)
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
        
        self.chat_history: List[Dict[str, str]] = []
        self.max_history_messages = max_history_messages
        self._closed = False
        
        # Enregistrer la fermeture propre à la sortie
        atexit.register(self._cleanup)
        
        print("✓ Modèle chargé avec succès!\n")
        
        # Optionnel: Préchauffer le cache avec un prompt vide
        self._warmup_cache()

    def _warmup_cache(self):
        """Préchauffer le cache du modèle pour accélérer la première requête."""
        try:
            print("🔥 Préchauffage du cache...", end="", flush=True)
            _ = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0.1
            )
            print(" ✓")
        except Exception:
            print(" (ignoré)")

    def _cleanup(self):
        """Nettoyer les ressources proprement."""
        if not self._closed and hasattr(self, 'llm'):
            try:
                if hasattr(self.llm, 'close'):
                    self.llm.close()
                self._closed = True
            except Exception:
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
        
        # Limiter automatiquement la taille de l'historique
        if len(self.chat_history) > self.max_history_messages:
            # Garder seulement les N derniers messages
            self.chat_history = self.chat_history[-self.max_history_messages:]
    
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
    
    def _prepare_messages(self, user_message: str, use_history: bool, max_history: int = None) -> List[Dict[str, str]]:
        """
        Préparer les messages en limitant l'historique pour accélérer le prompt processing.
        
        Args:
            user_message: Message de l'utilisateur
            use_history: Utiliser l'historique ou non
            max_history: Nombre max de messages d'historique à inclure (None = tout)
        """
        if not use_history:
            return [{"role": "user", "content": user_message}]
        
        # Si max_history est défini, limiter l'historique
        if max_history is not None and len(self.chat_history) > max_history:
            # Prendre les N derniers messages (sans compter le message actuel)
            messages = self.chat_history[-max_history:].copy()
        else:
            messages = self.chat_history.copy()
        
        return messages
    
    def generate_response(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        use_history: bool = True,
        stream: bool = False,
        max_history: int = None
    ) -> Union[str, Generator[str, None, None]]:
        """
        Générer une réponse au message utilisateur.
        
        Args:
            max_history: Limiter à N derniers messages (None = tout l'historique)
                         Réduire cette valeur accélère le premier token
        
        Returns:
            Si stream=False: str (réponse complète)
            Si stream=True: Generator[str, None, None] (chunks de texte)
        """
        if self._closed:
            raise RuntimeError("Le modèle a été fermé.")
        
        # Ajouter le message utilisateur à l'historique
        self.add_message("user", user_message)
        
        # Préparer les messages avec limitation d'historique
        messages = self._prepare_messages(user_message, use_history, max_history)
        
        try:
            if stream:
                # Retourner un générateur pour le streaming
                return self._stream_response(
                    messages, max_tokens, temperature, 
                    top_p, top_k, repeat_penalty
                )
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
                self.add_message("assistant", response_text)
                return response_text
        
        except Exception as e:
            # Retirer le dernier message de l'historique en cas d'erreur
            if self.chat_history and self.chat_history[-1]['role'] == 'user':
                self.chat_history.pop()
            raise RuntimeError(f"Erreur lors de la génération: {e}")

    def _stream_response(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float
    ) -> Generator[str, None, None]:
        """Générateur pour le streaming de la réponse."""
        response_text = ""
        
        try:
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
                    response_text += content
                    yield content
            
            # Ajouter la réponse complète à l'historique à la fin
            self.add_message("assistant", response_text)
            
        except Exception as e:
            # Retirer le message utilisateur en cas d'erreur
            if self.chat_history and self.chat_history[-1]['role'] == 'user':
                self.chat_history.pop()
            raise RuntimeError(f"Erreur lors du streaming: {e}")
    
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
    
    def get_stats(self):
        """Afficher les statistiques du modèle."""
        print("\n📊 Statistiques:")
        print(f"  - Messages en historique: {len(self.chat_history)}")
        print(f"  - Limite historique: {self.max_history_messages}")
        if self.chat_history:
            total_tokens = sum(self.count_tokens(msg['content']) for msg in self.chat_history)
            print(f"  - Tokens total historique: ~{total_tokens}")
        print()
    
    def chat_loop(self, stream: bool = True, max_history: int = 10):
        """
        Boucle de chat interactive.
        
        Args:
            stream: Activer le streaming
            max_history: Limiter à N derniers messages pour accélérer (recommandé: 10-15)
        """
        print("\n" + "="*60)
        print("Interface de Chat Gemma-3 (Optimisée)")
        print("="*60)
        print("Commandes:")
        print("  'quit' ou 'exit' : Quitter")
        print("  'clear'          : Effacer l'historique")
        print("  'history'        : Afficher l'historique")
        print("  'stats'          : Afficher les statistiques")
        print("  'tokens <text>'  : Compter les tokens")
        print("="*60)
        print(f"ℹ️  Historique limité à {max_history} derniers messages pour vitesse optimale")
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
                
                if user_input.lower() == 'stats':
                    self.get_stats()
                    continue
                
                if user_input.lower().startswith('tokens '):
                    text = user_input[7:]
                    count = self.count_tokens(text)
                    print(f"Nombre de tokens: {count}")
                    continue
                
                # Générer et afficher la réponse
                print("\n🤖 Gemma: ", end="", flush=True)
                
                import time
                start_time = time.time()
                first_token_time = None
                
                response = self.generate_response(
                    user_input,
                    stream=stream,
                    max_history=max_history  # Limiter l'historique pour vitesse
                )
                
                if stream:
                    for i, chunk in enumerate(response):
                        if i == 0:
                            first_token_time = time.time() - start_time
                        print(chunk, end="", flush=True)
                    print()  # Nouvelle ligne
                    
                    if first_token_time:
                        print(f"\n⏱️  Premier token: {first_token_time:.2f}s")
                else:
                    print(response)
                    elapsed = time.time() - start_time
                    print(f"\n⏱️  Temps total: {elapsed:.2f}s")
                
            except KeyboardInterrupt:
                print("\n\nAu revoir! 👋")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {e}\n")


def main():
    """Fonction principale."""
    try:
        with GemmaGGUFChat(
            model_path="gemma3:12b-it-qat",
            n_ctx=8192,
            n_gpu_layers=-1,      # Tout sur GPU
            n_batch=1024,         # Batch plus grand pour vitesse
            max_history_messages=20  # Limiter l'historique global
        ) as chat:
            
            # Exemple 1: Message simple sans historique
            print("\n" + "="*60)
            print("EXEMPLE 1: Message simple (sans historique)")
            print("="*60)
            
            import time
            start = time.time()
            response = chat.generate_response(
                "Bonjour! Qui es-tu?",
                use_history=False,
                stream=False
            )
            elapsed = time.time() - start
            
            print(f"Réponse: {response}")
            print(f"⏱️  Temps: {elapsed:.2f}s\n")
            
            # Effacer pour recommencer
            chat.clear_history()
            
            # Exemple 2: Conversation avec streaming
            print("\n" + "="*60)
            print("EXEMPLE 2: Conversation avec streaming")
            print("="*60)
            
            messages = [
                "Je m'appelle Alice et j'adore la programmation Python.",
                "Quel est mon nom?",
                "Qu'est-ce que j'aime?"
            ]
            
            for msg in messages:
                print(f"\n🧑 Vous: {msg}")
                print("🤖 Gemma: ", end="", flush=True)
                
                start = time.time()
                first_token = None
                
                for i, chunk in enumerate(chat.generate_response(msg, stream=True, max_history=10)):
                    if i == 0:
                        first_token = time.time() - start
                    print(chunk, end="", flush=True)
                
                print()
                if first_token:
                    print(f"⏱️  Premier token: {first_token:.2f}s")
            
            # Afficher statistiques
            chat.get_stats()
            
            # Lancer le chat interactif
            print("\n" + "="*60)
            print("LANCEMENT DU CHAT INTERACTIF")
            print("="*60)
            chat.clear_history()
            chat.chat_loop(stream=True, max_history=10)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nAssurez-vous que le fichier GGUF est dans le bon chemin.")
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()