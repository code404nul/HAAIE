import os
import re
from typing import Union
from collections.abc import Generator
from typing import List, Dict
from llama_cpp import Llama
import atexit

# Regex pour supprimer les blocs <think>...</think> générés par Gemma 4
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_thinking(text: str) -> str:
    """Supprime les blocs de réflexion interne de Gemma 4 avant l'envoi au TTS."""
    return _THINK_PATTERN.sub("", text).strip()


class GemmaGGUFChat:
    def __init__(
        self,
        model_path: str = "models/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: int = None,
        n_batch: int = 1024,
        max_history_messages: int = 20
    ):
        """
        Chat avec Gemma 4 26B-A4B via llama-cpp-python.

        Gemma 4 utilise le même format de chat que Gemma 3 (rôles system/user/assistant),
        mais peut émettre des blocs <think>...</think> qui doivent être filtrés
        avant d'être envoyés au TTS.

        Paramètres recommandés par Google/Unsloth :
            temperature=1.0, top_p=0.95, top_k=64, repeat_penalty=1.0
        """
        # Initialisé EN PREMIER pour éviter le crash du destructeur si __init__ échoue
        self._closed = False
        self.chat_history: List[Dict[str, str]] = []
        self.max_history_messages = max_history_messages

        # Résolution du chemin : on essaie absolu, puis relatif au script, puis tel quel
        resolved = self._resolve_model_path(model_path)
        if resolved is None:
            raise FileNotFoundError(
                f"Modèle introuvable : '{model_path}'\n"
                f"Cherché dans :\n"
                f"  - {model_path}\n"
                f"  - {os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)}\n"
                f"  - {os.path.join(os.getcwd(), model_path)}\n"
                f"Vérifiez le chemin et relancez."
            )
        model_path = resolved

        print(f"⏳ Chargement de Gemma 4 : {model_path}")
        print(f"🔧 n_batch={n_batch}, GPU layers={n_gpu_layers}")

        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format="gemma",   # Gemma 4 conserve le même format que Gemma 3
                verbose=True,
                use_mmap=True,
                use_mlock=True,
                n_batch=n_batch,
                logits_all=False,
                embedding=False,
            )
        except Exception as e:
            raise RuntimeError(f"Erreur chargement modèle : {e}")

        atexit.register(self._cleanup)
        print("✓ Gemma 4 chargé !\n")
        self._warmup_cache()

    # ------------------------------------------------------------------
    # Résolution du chemin modèle
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_path: str):
        """
        Cherche le fichier GGUF à plusieurs endroits :
          1. Tel quel (absolu ou relatif au CWD)
          2. Relatif au dossier du script Gemma4.py
          3. Relatif au répertoire de travail courant
        """
        candidates = [
            model_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),
            os.path.join(os.getcwd(), model_path),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def _warmup_cache(self):
        try:
            print("🔥 Préchauffage...", end="", flush=True)
            _ = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=1.0,
            )
            print(" ✓")
        except Exception:
            print(" (ignoré)")

    def _cleanup(self):
        if not self._closed and hasattr(self, "llm"):
            try:
                if hasattr(self.llm, "close"):
                    self.llm.close()
                self._closed = True
            except Exception:
                pass

    def __del__(self):
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False

    # ------------------------------------------------------------------
    # Gestion de l'historique
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str):
        """Ajoute un message. Le contenu doit déjà être nettoyé (sans blocs <think>)."""
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > self.max_history_messages:
            self.chat_history = self.chat_history[-self.max_history_messages:]

    def clear_history(self):
        self.chat_history = []
        print("✓ Historique effacé.\n")

    def get_history(self) -> List[Dict[str, str]]:
        return self.chat_history.copy()

    def count_tokens(self, text: str) -> int:
        if self._closed:
            raise RuntimeError("Modèle fermé.")
        return len(self.llm.tokenize(text.encode()))

    def _prepare_messages(
        self, user_message: str, use_history: bool, max_history: int = None
    ) -> List[Dict[str, str]]:
        if not use_history:
            return [{"role": "user", "content": user_message}]
        if max_history is not None and len(self.chat_history) > max_history:
            messages = self.chat_history[-max_history:].copy()
        else:
            messages = self.chat_history.copy()
        return messages

    # ------------------------------------------------------------------
    # Génération
    # ------------------------------------------------------------------

    def generate_response(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 1.0,      # Valeur recommandée pour Gemma 4
        top_p: float = 0.95,           # Valeur recommandée pour Gemma 4
        top_k: int = 64,               # Valeur recommandée pour Gemma 4
        repeat_penalty: float = 1.0,   # Garder à 1.0 pour Gemma 4 (pas de boucles)
        use_history: bool = True,
        stream: bool = False,
        max_history: int = None,
        strip_think: bool = True,      # Filtre automatique des blocs <think>
    ) -> Union[str, Generator[str, None, None]]:
        """
        Génère une réponse.

        strip_think=True (défaut) supprime les blocs <think>...</think>
        avant de retourner le texte et de l'ajouter à l'historique.
        Idéal pour le TTS : le VTuber ne lira pas les pensées internes.
        """
        if self._closed:
            raise RuntimeError("Modèle fermé.")

        self.add_message("user", user_message)
        messages = self._prepare_messages(user_message, use_history, max_history)

        try:
            if stream:
                return self._stream_response(
                    messages, max_tokens, temperature,
                    top_p, top_k, repeat_penalty, strip_think
                )
            else:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                )
                raw_text = response["choices"][0]["message"]["content"]
                clean_text = strip_thinking(raw_text) if strip_think else raw_text
                # On sauvegarde la version propre dans l'historique
                self.add_message("assistant", clean_text)
                return clean_text

        except Exception as e:
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            raise RuntimeError(f"Erreur génération : {e}")

    def _stream_response(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        strip_think: bool,
    ) -> Generator[str, None, None]:
        """
        Streaming avec filtrage à la volée des blocs <think>.

        Le bloc de réflexion est accumulé en mémoire tampon et jamais émis.
        Une fois </think> reçu, on reprend l'émission normale.
        """
        raw_buffer = ""       # tout le texte brut (pour l'historique)
        output_buffer = ""    # texte à émettre (hors <think>)
        in_think = False
        think_buf = ""

        try:
            for chunk in self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"]
                if "content" not in delta:
                    continue
                content = delta["content"]
                raw_buffer += content

                if not strip_think:
                    output_buffer += content
                    yield content
                    continue

                # --- Filtrage des blocs <think> en streaming ---
                think_buf += content

                while think_buf:
                    if in_think:
                        end_idx = think_buf.find("</think>")
                        if end_idx != -1:
                            # Fin du bloc de réflexion
                            think_buf = think_buf[end_idx + len("</think>"):]
                            in_think = False
                        else:
                            # Toujours dans le bloc, on attend
                            break
                    else:
                        start_idx = think_buf.find("<think>")
                        if start_idx != -1:
                            # Émettre ce qui précède <think>
                            before = think_buf[:start_idx]
                            if before:
                                output_buffer += before
                                yield before
                            think_buf = think_buf[start_idx + len("<think>"):]
                            in_think = True
                        else:
                            # Pas de <think> en vue, mais le début pourrait en cacher un
                            safe_len = max(0, len(think_buf) - len("<think>"))
                            if safe_len > 0:
                                safe_part = think_buf[:safe_len]
                                output_buffer += safe_part
                                yield safe_part
                                think_buf = think_buf[safe_len:]
                            break

            # Émettre ce qu'il reste dans le tampon (hors think)
            if think_buf and not in_think and strip_think:
                output_buffer += think_buf
                yield think_buf

            # Historique : version propre uniquement
            clean = strip_thinking(raw_buffer) if strip_think else raw_buffer
            self.add_message("assistant", clean.strip())

        except Exception as e:
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            raise RuntimeError(f"Erreur streaming : {e}")

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def print_history(self):
        print("\n" + "=" * 60)
        print("HISTORIQUE")
        print("=" * 60)
        for i, msg in enumerate(self.chat_history, 1):
            role = "Vous" if msg["role"] == "user" else "Gemma 4"
            print(f"\n[{i}] {role}:\n{msg['content']}")
        print("\n" + "=" * 60 + "\n")

    def get_stats(self):
        print("\n📊 Stats Gemma 4 :")
        print(f"  Messages en historique : {len(self.chat_history)}")
        print(f"  Limite historique      : {self.max_history_messages}")
        if self.chat_history:
            total = sum(self.count_tokens(m["content"]) for m in self.chat_history)
            print(f"  Tokens total           : ~{total}")
        print()

    def chat_loop(self, stream: bool = True, max_history: int = 10):
        print("\n" + "=" * 60)
        print("Chat Gemma 4 — 26B A4B MoE")
        print("=" * 60)
        print("  quit/exit  : quitter")
        print("  clear      : effacer l'historique")
        print("  history    : afficher l'historique")
        print("  stats      : statistiques")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\n🧑 Vous : ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("\nAu revoir ! 👋")
                    break
                if user_input.lower() == "clear":
                    self.clear_history()
                    continue
                if user_input.lower() == "history":
                    self.print_history()
                    continue
                if user_input.lower() == "stats":
                    self.get_stats()
                    continue

                import time
                print("\n🤖 Gemma 4 : ", end="", flush=True)
                start = time.time()
                first_token = None

                resp = self.generate_response(
                    user_input, stream=stream, max_history=max_history
                )

                if stream:
                    for i, chunk in enumerate(resp):
                        if i == 0:
                            first_token = time.time() - start
                        print(chunk, end="", flush=True)
                    print()
                    if first_token:
                        print(f"\n⏱  Premier token : {first_token:.2f}s")
                else:
                    print(resp)
                    print(f"\n⏱  Total : {time.time() - start:.2f}s")

            except KeyboardInterrupt:
                print("\n\nAu revoir ! 👋")
                break
            except Exception as e:
                print(f"\n❌ Erreur : {e}\n")


def main():
    try:
        with GemmaGGUFChat(
            model_path="models/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf",
            n_ctx=8192,
            n_gpu_layers=-1,
            n_batch=1024,
            max_history_messages=20,
        ) as chat:
            chat.clear_history()
            chat.chat_loop(stream=True, max_history=10)

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Erreur : {e}")


if __name__ == "__main__":
    main()