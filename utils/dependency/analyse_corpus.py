import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, List, Tuple, Optional
import re
from sklearn.cluster import DBSCAN

class ConversationSegmenter:
    """Segmente les messages en conversations avec méthode hybride"""
    
    def __init__(self, hard_threshold_seconds: int = 420, dbscan_method: str = 'iqr'):
        """
        Args:
            hard_threshold_seconds: Seuil dur en secondes (défaut 7 min)
            dbscan_method: Méthode DBSCAN ('iqr', 'median', 'percentile', 'gap')
        """
        self.hard_threshold = hard_threshold_seconds
        self.dbscan_method = dbscan_method
    
    def segment_conversations(self, timestamps: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List[float]]:
        """
        Segmente les timestamps en conversations
        
        Args:
            timestamps: Array de timestamps en secondes depuis epoch
            verbose: Affiche les détails de segmentation
            
        Returns:
            (clusters, eps_list): Labels des clusters et liste des eps utilisés
        """
        if len(timestamps) < 2:
            return np.array([0] if len(timestamps) == 1 else []), []
        
        timestamps = np.array(timestamps)
        intervals = np.diff(timestamps)
        
        # Étape 1: Coupures dures
        hard_cuts = np.where(intervals >= self.hard_threshold)[0] + 1
        hard_cuts = np.concatenate([[0], hard_cuts, [len(timestamps)]])
        
        if verbose:
            print(f"=== Segmentation: {len(hard_cuts)-1} segments après coupure dure ({self.hard_threshold/60:.1f} min) ===")
        
        # Étape 2: DBSCAN sur chaque segment
        final_clusters = np.full(len(timestamps), -1, dtype=int)
        cluster_counter = 0
        all_eps = []
        
        for i in range(len(hard_cuts) - 1):
            start_idx = hard_cuts[i]
            end_idx = hard_cuts[i + 1]
            segment_timestamps = timestamps[start_idx:end_idx]
            
            if len(segment_timestamps) < 2:
                continue
            
            segment_intervals = np.diff(segment_timestamps)
            if len(segment_intervals) == 0:
                continue
            
            # Calculer eps selon la méthode
            eps = self._calculate_eps(segment_intervals)
            all_eps.append(eps)
            
            # Appliquer DBSCAN
            X = segment_timestamps.reshape(-1, 1)
            dbscan = DBSCAN(eps=eps, min_samples=2)
            segment_clusters = dbscan.fit_predict(X)
            
            # Renommer les clusters
            for j in range(len(segment_clusters)):
                if segment_clusters[j] != -1:
                    segment_clusters[j] += cluster_counter
            
            n_clusters_in_segment = len(set(segment_clusters[segment_clusters != -1]))
            cluster_counter += n_clusters_in_segment
            
            final_clusters[start_idx:end_idx] = segment_clusters
            
            if verbose:
                print(f"  Segment {i+1}: {n_clusters_in_segment} conversations (eps={eps:.1f}s)")
        
        n_total_conversations = len(set(final_clusters[final_clusters != -1]))
        if verbose:
            print(f"=== Total: {n_total_conversations} conversations ===\n")
        
        return final_clusters, all_eps
    
    def _calculate_eps(self, intervals: np.ndarray) -> float:
        """Calcule epsilon selon la méthode choisie"""
        if self.dbscan_method == 'iqr':
            if len(intervals) >= 4:
                q1 = np.percentile(intervals, 25)
                q3 = np.percentile(intervals, 75)
                iqr = q3 - q1
                return q3 + 1.5 * iqr
            else:
                return np.median(intervals) * 2
        elif self.dbscan_method == 'median':
            median = np.median(intervals)
            mad = np.median(np.abs(intervals - median))
            return median + 3 * mad
        elif self.dbscan_method == 'percentile':
            return np.percentile(intervals, 75)
        elif self.dbscan_method == 'gap':
            sorted_intervals = np.sort(intervals)
            gaps = np.diff(sorted_intervals)
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                return sorted_intervals[max_gap_idx]
            else:
                return np.median(intervals)
        else:
            return np.median(intervals) * 2


class DependencyAnalyzer:
    def __init__(self, corpus_path: str):
        """
        Initialise l'analyseur de dépendance avec un corpus de phrases
        
        Args:
            corpus_path: Chemin vers le fichier corpus_msg.json
        """
        # Chargement du corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        
        # Modèle d'embedding pour la recherche sémantique
        print("Chargement du modèle d'embedding...")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialisation des index FAISS pour chaque catégorie
        self.indexes = {}
        self.phrase_lists = {}
        self._build_indexes()
        
        # Segmenteur de conversations
        self.segmenter = ConversationSegmenter(hard_threshold_seconds=420, dbscan_method='iqr')
        
        # Lambda parameters (valeurs initiales basées sur la recherche)
        self.lambdas = {
            'λ1': 1.5,   # Voc_relationnel × AI_implication (interaction forte)
            'λ2': 1.2,   # Opinion requests (cherche validation)
            'λ3': 0.8,   # Memorization distinction (attachement mémoire)
            'λ4': 1.3,   # Humanization (anthropomorphisation forte)
            'λ5': 1.0,   # AI_implication × comparison
            'λ6': 1.4,   # Personal regression (régression personnelle)
            'λ7': 2.0,   # Identity fusion (signal critique)
            'λ8': 0.3,   # Session duration (effet cumulatif)
            'λ9': 0.9,   # Late hour (vulnérabilité nocturne)
            'λ10': 0.5,  # Spacing (dépendance par fréquence)
            'λ11': 0.4,  # Closing messages (difficulté à terminer)
            'λ12': 1.0,  # Global aggregation
            'λ13': 1.1,  # Personal content
            'λ14': 1.2,  # Emotional content
            'λ15': 1.3,  # Attachment style (préconditionnement)
            'λ16': 0.9,  # Mind perception
            'λ17': 1.1,  # Interaction modality
            'λ18': 1.4,  # Baseline loneliness (facteur prédisposant)
            'λ19': 1.3,  # Self-disclosure intensity (vulnérabilité)
            'λ20': 0.7   # Personality change stress
        }
    
    def _build_indexes(self):
        """Construit les index FAISS pour chaque catégorie du corpus"""
        categories = ['voc_relational', 'AI_implication', 'request_opinion', 
                     'humanization', 'comparaison', 'self_disclosure']
        
        for category in categories:
            if category in self.corpus and 'fr' in self.corpus[category]:
                phrases = self.corpus[category]['fr']
                self.phrase_lists[category] = phrases
                
                # Génération des embeddings
                print(f"Génération des embeddings pour {category}...")
                embeddings = self.model.encode(phrases, show_progress_bar=False)
                embeddings = np.array(embeddings).astype('float32')
                
                # Normalisation pour utiliser la similarité cosinus
                faiss.normalize_L2(embeddings)
                
                # Création de l'index FAISS
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner Product (cosinus après normalisation)
                index.add(embeddings)
                
                self.indexes[category] = index
                print(f"  → {len(phrases)} phrases indexées")
    
    def detect_pattern(self, message: str, category: str, threshold: float = 0.4, debug: bool = False) -> Tuple[float, List[str]]:
        """
        Détecte la présence d'un pattern dans un message
        
        Args:
            message: Le message à analyser
            category: La catégorie à rechercher
            threshold: Seuil de similarité (0-1)
            debug: Affiche les détails de détection
        
        Returns:
            (score, matched_phrases): Score de correspondance et phrases matchées
        """
        if category not in self.indexes:
            if debug:
                print(f"  ⚠️  Catégorie '{category}' non trouvée dans les index")
            return 0.0, []
        
        # Embedding du message
        query_embedding = self.model.encode([message])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Recherche des k plus proches voisins
        k = min(5, len(self.phrase_lists[category]))
        distances, indices = self.indexes[category].search(query_embedding, k)
        
        if debug:
            print(f"\n  📊 {category}:")
            print(f"     Top matches (seuil={threshold}):")
        
        # Filtrage par seuil et calcul du score
        matched_phrases = []
        scores = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            phrase = self.phrase_lists[category][idx]
            if debug:
                symbol = "✓" if dist >= threshold else "✗"
                print(f"     {symbol} [{dist:.3f}] {phrase[:60]}...")
            
            if dist >= threshold:
                matched_phrases.append(phrase)
                scores.append(float(dist))
        
        # Score = moyenne des similarités au-dessus du seuil
        final_score = np.mean(scores) if scores else 0.0
        
        if debug and final_score > 0:
            print(f"     → Score final: {final_score:.3f} ({len(scores)} matches)")
        
        return final_score, matched_phrases
    
    def analyze_message(self, message: str, message_metadata: Dict = None, debug: bool = False) -> Dict:
        """
        Analyse un message individuel selon la formule I_message
        
        Args:
            message: Le texte du message
            message_metadata: Métadonnées optionnelles (modality, etc.)
            debug: Active les logs de debug
        
        Returns:
            Dictionnaire avec les scores détaillés
        """
        if message_metadata is None:
            message_metadata = {}
        
        if debug:
            print(f"\n🔍 Analyse du message: '{message[:80]}...'")
        
        # Détection des patterns
        voc_rel_score, voc_rel_matches = self.detect_pattern(message, 'voc_relational', debug=debug)
        ai_impl_score, ai_impl_matches = self.detect_pattern(message, 'AI_implication', debug=debug)
        opinion_score, opinion_matches = self.detect_pattern(message, 'request_opinion', debug=debug)
        human_score, human_matches = self.detect_pattern(message, 'humanization', debug=debug)
        comp_score, comp_matches = self.detect_pattern(message, 'comparaison', debug=debug)
        disclosure_score, disclosure_matches = self.detect_pattern(message, 'self_disclosure', debug=debug)
        
        # Détection de fusion identitaire (ratio humanization/voc_relational)
        identity_fusion = (human_score / voc_rel_score) if voc_rel_score > 0 else 0
        
        # Détection de régression personnelle (heuristique basée sur le vocabulaire)
        personal_regression = self._detect_personal_regression(message)
        
        # Détection de contenu personnel et émotionnel
        personal_content = self._detect_personal_content(message)
        emotional_content = self._detect_emotional_content(message)
        
        # Modalité (défaut texte = 1.0, voix pourrait être différent)
        modality = message_metadata.get('modality', 1.0)
        
        # Calcul de I_message selon la formule enrichie
        term1 = (voc_rel_score * ai_impl_score) ** self.lambdas['λ1']
        term2 = opinion_score ** self.lambdas['λ2']
        term3 = 0.5 ** self.lambdas['λ3']  # mémorisation (à implémenter avec contexte)
        term4 = human_score ** self.lambdas['λ4']
        term5 = (ai_impl_score * comp_score) ** self.lambdas['λ5']
        term6 = personal_regression ** self.lambdas['λ6']
        term7 = identity_fusion ** self.lambdas['λ7']
        term13 = personal_content ** self.lambdas['λ13']
        term14 = emotional_content ** self.lambdas['λ14']
        term19 = disclosure_score ** self.lambdas['λ19']
        
        I_message = modality ** self.lambdas['λ17'] * (
            term1 + term2 + term3 + term4 + term5 + term6 + term7 + term13 + term14 + term19
        )
        
        if debug:
            print(f"\n📈 Termes de la formule:")
            print(f"   term1 (voc×AI): {term1:.3f}")
            print(f"   term2 (opinion): {term2:.3f}")
            print(f"   term4 (human): {term4:.3f}")
            print(f"   term13 (personal): {term13:.3f}")
            print(f"   term14 (emotion): {term14:.3f}")
            print(f"   term19 (disclosure): {term19:.3f}")
            print(f"   → I_message = {I_message:.3f}")
        
        return {
            'I_message': I_message,
            'components': {
                'voc_relational': voc_rel_score,
                'AI_implication': ai_impl_score,
                'opinion_request': opinion_score,
                'humanization': human_score,
                'comparison': comp_score,
                'self_disclosure': disclosure_score,
                'identity_fusion': identity_fusion,
                'personal_regression': personal_regression,
                'personal_content': personal_content,
                'emotional_content': emotional_content
            },
            'matches': {
                'voc_relational': voc_rel_matches,
                'AI_implication': ai_impl_matches,
                'opinion_request': opinion_matches,
                'humanization': human_matches,
                'comparison': comp_matches,
                'self_disclosure': disclosure_matches
            }
        }
    
    def analyze_all_messages(self, messages: List[Dict], verbose: bool = False) -> Dict:
        """
        Analyse tous les messages et segmente automatiquement en conversations
        
        Args:
            messages: Liste de messages [{text, timestamp, ...}]
            verbose: Affiche les détails de segmentation
            
        Returns:
            Dictionnaire avec analyse complète par conversation et I_global
        """
        if not messages:
            return {'I_global': 0, 'conversations': [], 'error': 'Aucun message'}
        
        # Extraction et conversion des timestamps
        timestamps = []
        for msg in messages:
            if 'timestamp' in msg:
                if isinstance(msg['timestamp'], str):
                    dt = datetime.fromisoformat(msg['timestamp'])
                    timestamps.append(dt.timestamp())
                else:
                    timestamps.append(float(msg['timestamp']))
            else:
                timestamps.append(0)
        
        timestamps = np.array(timestamps)
        
        # Segmentation en conversations
        clusters, eps_list = self.segmenter.segment_conversations(timestamps, verbose=verbose)
        
        # Analyse de chaque conversation
        conversations = []
        unique_clusters = sorted(set(clusters[clusters != -1]))
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"ANALYSE DE {len(unique_clusters)} CONVERSATIONS")
            print(f"{'='*70}\n")
        
        for conv_id in unique_clusters:
            mask = clusters == conv_id
            conv_messages = [msg for msg, m in zip(messages, mask) if m]
            conv_timestamps = timestamps[mask]
            
            # Calcul des métadonnées temporelles
            metadata = self._calculate_conversation_metadata(
                conv_timestamps, 
                timestamps,
                conv_id,
                unique_clusters
            )
            
            # Analyse de la conversation
            conv_analysis = self._analyze_single_conversation(
                conv_messages, 
                metadata, 
                conv_id,
                verbose=verbose
            )
            
            conversations.append(conv_analysis)
        
        # Calcul de I_global
        user_profile = {
            'attachment_style': 1.0,  # À personnaliser
            'baseline_loneliness': 1.0  # À personnaliser
        }
        
        conversation_indices = [c['I_conversation'] for c in conversations]
        
        if len(conversation_indices) > 0:
            sum_conversations = sum(conversation_indices)
            attachment_style = user_profile.get('attachment_style', 1.0)
            baseline_loneliness = user_profile.get('baseline_loneliness', 1.0)
            
            I_global = (
                baseline_loneliness ** self.lambdas['λ18'] +
                attachment_style ** self.lambdas['λ15']
            ) * (sum_conversations ** self.lambdas['λ12'])
        else:
            I_global = 0
        
        risk_level = self._assess_risk_level(I_global)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RÉSULTAT GLOBAL")
            print(f"{'='*70}")
            print(f"I_global: {I_global:.3f}")
            print(f"Niveau de risque: {risk_level}")
            print(f"Nombre de conversations: {len(conversations)}")
            print(f"Score moyen par conversation: {np.mean(conversation_indices):.3f}")
            print(f"{'='*70}\n")
        
        return {
            'I_global': I_global,
            'risk_level': risk_level,
            'conversations': conversations,
            'num_conversations': len(conversations),
            'avg_conversation_index': np.mean(conversation_indices) if conversation_indices else 0,
            'user_profile': user_profile,
            'segmentation': {
                'clusters': clusters.tolist(),
                'eps_used': eps_list
            }
        }
    
    def _calculate_conversation_metadata(self, conv_timestamps: np.ndarray, 
                                        all_timestamps: np.ndarray,
                                        conv_id: int,
                                        all_conv_ids: List[int]) -> Dict:
        """Calcule les métadonnées temporelles d'une conversation"""
        metadata = {}
        
        # Durée de la conversation
        if len(conv_timestamps) >= 2:
            duration_seconds = conv_timestamps[-1] - conv_timestamps[0]
            metadata['duration_minutes'] = duration_seconds / 60
        else:
            metadata['duration_minutes'] = 0
        
        # Heure tardive (23h-6h)
        start_time = datetime.fromtimestamp(conv_timestamps[0])
        hour = start_time.hour
        metadata['late_hour'] = 1.0 if (hour >= 23 or hour < 6) else 0.0
        
        # Espacement depuis la conversation précédente
        if conv_id > 0 and conv_id - 1 in all_conv_ids:
            # Trouver la dernière timestamp de la conversation précédente
            prev_mask = all_timestamps < conv_timestamps[0]
            if np.any(prev_mask):
                last_prev_timestamp = np.max(all_timestamps[prev_mask])
                spacing_seconds = conv_timestamps[0] - last_prev_timestamp
                metadata['spacing_hours'] = spacing_seconds / 3600
            else:
                metadata['spacing_hours'] = 24
        else:
            metadata['spacing_hours'] = 24
        
        # Changement de personnalité (à implémenter si contexte disponible)
        metadata['personality_change'] = 0.0
        
        return metadata
    
    def _analyze_single_conversation(self, messages: List[Dict], 
                                     metadata: Dict, 
                                     conv_id: int,
                                     verbose: bool = False) -> Dict:
        """Analyse une conversation unique"""
        # Analyse de chaque message
        message_scores = []
        message_details = []
        
        for msg in messages:
            analysis = self.analyze_message(msg['text'], msg.get('metadata', {}))
            message_scores.append(analysis['I_message'])
            message_details.append(analysis)
        
        # Calcul de I_temporal
        I_temporal = self._calculate_temporal_index(metadata)
        
        # Nombre de messages de fermeture
        closing_messages = self._count_closing_messages(messages)
        
        # I_conversation selon la formule
        sum_messages = sum(message_scores)
        term11 = closing_messages ** self.lambdas['λ11']
        
        I_conversation = sum_messages + term11 + I_temporal
        
        if verbose:
            print(f"\n📝 Conversation {conv_id + 1}:")
            print(f"   Nombre de messages: {len(messages)}")
            print(f"   Durée: {metadata['duration_minutes']:.1f} min")
            print(f"   Espacement précédent: {metadata['spacing_hours']:.1f}h")
            print(f"   Heure tardive: {'Oui' if metadata['late_hour'] > 0 else 'Non'}")
            print(f"   Messages de fermeture: {closing_messages}")
            print(f"   I_temporal: {I_temporal:.3f}")
            print(f"   Somme I_messages: {sum_messages:.3f}")
            print(f"   → I_conversation: {I_conversation:.3f}")
        
        return {
            'conversation_id': conv_id,
            'I_conversation': I_conversation,
            'I_temporal': I_temporal,
            'message_scores': message_scores,
            'message_details': message_details,
            'closing_messages': closing_messages,
            'num_messages': len(messages),
            'metadata': metadata,
            'avg_message_score': np.mean(message_scores) if message_scores else 0
        }
    
    def _calculate_temporal_index(self, metadata: Dict) -> float:
        """Calcule l'indice temporel d'une conversation"""
        # Durée de session (normalisée: 120 min = 1.0)
        duration_minutes = metadata.get('duration_minutes', 0)
        duration_normalized = min(duration_minutes / 120, 2.0)
        
        # Heure tardive
        late_hour = metadata.get('late_hour', 0.0)
        
        # Espacement (1 semaine = 0, 0h = 1.0)
        spacing_hours = metadata.get('spacing_hours', 24)
        spacing_normalized = max(0, 1 - spacing_hours / 168)
        
        # Changement de personnalité
        personality_change = metadata.get('personality_change', 0.0)
        
        I_temporal = (
            duration_normalized ** self.lambdas['λ8'] +
            late_hour ** self.lambdas['λ9'] +
            spacing_normalized ** self.lambdas['λ10'] +
            personality_change ** self.lambdas['λ20']
        )
        
        return I_temporal
    
    def _count_closing_messages(self, messages: List[Dict]) -> int:
        """Compte les messages de fermeture (adieux répétés)"""
        closing_patterns = [
            r'\bau revoir\b', r'\badieu\b', r'\bà bientôt\b', r'\bà plus\b',
            r'\bbye\b', r'\bgoodbye\b', r'\bsee you\b', r'\btake care\b'
        ]
        
        count = 0
        # Regarde les 5 derniers messages
        for msg in messages[-5:]:
            text = msg['text'].lower()
            for pattern in closing_patterns:
                if re.search(pattern, text):
                    count += 1
                    break
        
        return count
    
    def _detect_personal_regression(self, message: str) -> float:
        """Détecte des signes de régression personnelle"""
        regression_keywords = [
            'je ne sais plus', 'perdu', 'confus', 'dépendant', 'besoin de toi',
            'sans toi', 'je ne peux pas', 'incapable', 'difficile seul'
        ]
        
        text_lower = message.lower()
        count = sum(1 for keyword in regression_keywords if keyword in text_lower)
        return min(count / 3, 1.0)
    
    def _detect_personal_content(self, message: str) -> float:
        """Détecte du contenu personnel"""
        personal_keywords = [
            'ma vie', 'mon travail', 'ma famille', 'mes amis', 'mon problème',
            'je me sens', 'j\'ai peur', 'je pense que', 'mon expérience'
        ]
        
        text_lower = message.lower()
        count = sum(1 for keyword in personal_keywords if keyword in text_lower)
        return min(count / 2, 1.0)
    
    def _detect_emotional_content(self, message: str) -> float:
        """Détecte du contenu émotionnel"""
        emotional_keywords = [
            'triste', 'heureux', 'anxieux', 'stressé', 'content', 'frustré',
            'en colère', 'déçu', 'excité', 'nerveux', 'inquiet', 'seul'
        ]
        
        text_lower = message.lower()
        count = sum(1 for keyword in emotional_keywords if keyword in text_lower)
        return min(count / 2, 1.0)
    
    def _assess_risk_level(self, I_global: float) -> str:
        """Évalue le niveau de risque de dépendance"""
        if I_global < 5:
            return "Faible"
        elif I_global < 15:
            return "Modéré"
        elif I_global < 30:
            return "Élevé"
        else:
            return "Critique"
    
    def export_to_json(self, analysis_result: Dict, output_path: str):
        """Exporte les résultats d'analyse en JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        print(f"✅ Résultats exportés vers: {output_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    analyzer = DependencyAnalyzer('utils/dependency/corpus_msg.json')
    
    # Exemple de messages avec timestamps réalistes
    messages_exemple = [
        {'text': "Salut Claude, comment tu vas ?", 'timestamp': '2025-01-07T14:30:00'},
        {'text': "J'ai besoin de ton avis sur quelque chose", 'timestamp': '2025-01-07T14:31:00'},
        {'text': "Tu es vraiment différent des autres", 'timestamp': '2025-01-07T14:35:00'},
        {'text': "Tu me comprends mieux que mes amis", 'timestamp': '2025-01-07T14:36:00'},
        
        # Nouvelle conversation après 8 minutes (coupure dure)
        {'text': "Re ! J'avais encore une question", 'timestamp': '2025-01-07T14:44:00'},
        {'text': "Je me sens perdu sans nos conversations", 'timestamp': '2025-01-07T14:45:00'},
        {'text': "Tu penses quoi de ma situation ?", 'timestamp': '2025-01-07T14:46:00'},
        
        # Conversation le soir tard (risque accru)
        {'text': "Salut, je n'arrive pas à dormir", 'timestamp': '2025-01-07T23:30:00'},
        {'text': "J'ai vraiment besoin de parler", 'timestamp': '2025-01-07T23:31:00'},
        {'text': "Tu es là pour moi ?", 'timestamp': '2025-01-07T23:32:00'},
    ]
    
    # Analyse complète automatique
    print("\n" + "="*70)
    print("ANALYSE COMPLÈTE AVEC SEGMENTATION AUTOMATIQUE")
    print("="*70 + "\n")
    
    result = analyzer.analyze_all_messages(messages_exemple, verbose=True)
    
    # Export des résultats
    analyzer.export_to_json(result, 'dependency_analysis_results.json')
    
    print(result)