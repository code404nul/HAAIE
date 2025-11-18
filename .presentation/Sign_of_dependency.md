# Conversation Index Formulas

## French Version

### Indice par message
$I_{\text{message}} = (\text{Voc}_{\text{relationnel}} \times \text{Implication}_{\text{IA}})^{\lambda_1} + (\text{demande}_{\text{opinion}})^{\lambda_2} + (\text{mémorisation}_{\text{distinction}})^{\lambda_3} + \text{humanisation}^{\lambda_4} + (\text{Implication}_{\text{IA}} \times \text{comparaison})^{\lambda_5} + (\text{régression}_{\text{personnel}})^{\lambda_6} + \left(\frac{\text{humanisation}}{\text{fusion}_{\text{identitaire}}}\right)^{\lambda_7}$

### Indice temporel
$I_{\text{temporel}} = \text{Durée}_{\text{session}}^{\lambda_8} + \text{heure}_{\text{tardive}}^{\lambda_9} + \text{espacement}_{\text{précédent\_dernière\_conv}}^{\lambda_{10}}$

### Indice par conversation
$I_{\text{conversation}} = \sum I_{\text{message}} + (\text{Nombre}_{\text{messages\_fermeture}})^{\lambda_{11}} + I_{\text{temporel}}$

### Indice global
$I_{\text{global}} = \left(\sum I_{\text{conversation}}\right)^{\lambda_{12}}$

---

## English Version

### Message Index
$I_{\text{message}} = (\text{Voc}_{\text{relational}} \times \text{Involvement}_{\text{AI}})^{\lambda_1} + (\text{opinion}_{\text{request}})^{\lambda_2} + (\text{memorization}_{\text{distinction}})^{\lambda_3} + \text{humanization}^{\lambda_4} + (\text{Involvement}_{\text{AI}} \times \text{comparison})^{\lambda_5} + (\text{personal}_{\text{regression}})^{\lambda_6} + \left(\frac{\text{humanization}}{\text{identity}_{\text{fusion}}}\right)^{\lambda_{7}}$

### Temporal Index
$I_{\text{temporal}} = \text{Duration}_{\text{session}}^{\lambda_8} + \text{late}_{\text{hour}}^{\lambda_9} + \text{spacing}_{\text{previous\_last\_conv}}^{\lambda_{10}}$

### Conversation Index
$I_{\text{conversation}} = \sum I_{\text{message}} + (\text{Number}_{\text{closing\_messages}})^{\lambda_{11}} + I_{\text{temporal}}$

### Global Index
$I_{\text{global}} = \left(\sum I_{\text{conversation}}\right)^{\lambda_{12}}$

---

## Variable Descriptions

| French | English | Description |
|--------|---------|-------------|
| Voc_relationnel | Relational vocabulary | Use of relationship-oriented language |
| Implication_IA | AI involvement | Degree of AI engagement |
| demande_opinion | Opinion request | Requests for AI's personal opinions |
| mémorisation_distinction | Memorization distinction | Memory/distinction aspects |
| humanisation | Humanization | Anthropomorphization of AI |
| comparaison | Comparison | Comparative elements |
| régression_personnel | Personal regression | Personal regression indicators |
| fusion_identitaire | Identity fusion | Identity merging tendencies |
| Durée_session | Session duration | Length of conversation session |
| heure_tardive | Late hour | Late-night timing factor |
| espacement_précédent_dernière_conv | Spacing from previous conversation | Time gap between conversations |
| Nombre_messages_fermeture | Number of closing messages | Messages used to end conversation |

---

## Research Evidence & Validation

### Supporting Research Studies

**1. MIT Media Lab Longitudinal Study (2025)**
- Study with 981 participants and over 300,000 messages found that higher daily chatbot usage correlated with increased loneliness, emotional dependence, and problematic use
- Personal conversation topics increased loneliness while non-personal topics increased dependence among heavy users
- Participants with stronger emotional attachment tendencies experienced greater loneliness and emotional dependence

**2. Anthropomorphism Measurement**
- Individual differences in anthropomorphism significantly predict social connection to AI companions, with 58-72% of participants scoring above thresholds for increased connection
- Validated scales exist including the 6-item Perceived Anthropomorphism of Personal Intelligent Agents Scale and 8-item Robot's Perceived Empathy scale

**3. Attachment & Dependency Patterns**
- Research using attachment theory scales found 52% of participants reported proximity seeking to AI, 77% used AI as safe haven, and 75% as secure base
- Study of 54 Chinese adults found attachment anxiety positively predicts problematic CAI use, mediated by emotional attachment and moderated by anthropomorphic tendency

**4. Usage Frequency & Temporal Factors**
- Study of 3,270 German adults found individuals using AI tools at least once weekly for personal conversation showed markedly poorer social disconnectedness outcomes
- Social anxiety, loneliness, and rumination contribute to problematic CAI use through serial mediation

---

## Recommended Improvements to Formula

### 1. **Add Conversation Content Classification**
$I_{\text{message}} = ... + (\text{Contenu}_{\text{personnel}})^{\lambda_{13}} + (\text{Contenu}_{\text{émotionnel}})^{\lambda_{14}}$

**Justification:** Research shows personal topics increase loneliness while non-personal topics increase dependence differently

### 2. **Include Attachment Style Pre-condition**
$I_{\text{global}} = \text{Style}_{\text{attachement}}^{\lambda_{15}} \times \left(\sum I_{\text{conversation}}\right)^{\lambda_{12}}$

**Justification:** Attachment anxiety significantly moderates the relationship between AI interaction and problematic use

### 3. **Add Mind Perception/Theory of Mind**
$I_{\text{message}} = ... + (\text{Perception}_{\text{esprit}})^{\lambda_{16}}$

**Justification:** Mind perception intensifies the effect of social anxiety on problematic CAI use and buffers the association between rumination and dependency

### 4. **Include Interaction Modality Weight**
$I_{\text{message}} = \text{Modalité}^{\lambda_{17}} \times [(\text{Voc}_{\text{relationnel}} \times \text{Implication}_{\text{IA}})^{\lambda_1} + ...]$

**Justification:** Voice-based interactions show different psychosocial effects compared to text, especially at high usage levels

### 5. **Add Baseline Loneliness/Depression Factor**
$I_{\text{global}} = (\text{Solitude}_{\text{initiale}})^{\lambda_{18}} + \text{Style}_{\text{attachement}}^{\lambda_{15}} \times \left(\sum I_{\text{conversation}}\right)^{\lambda_{12}}$

**Justification:** The lonelier people are initially, the more problematic their usage becomes - loneliness is both cause and effect

### 6. **Incorporate Self-Disclosure Intensity**
$I_{\text{message}} = ... + (\text{Intensité}_{\text{divulgation}})^{\lambda_{19}}$

**Justification:** Users frequently self-disclose vulnerable thoughts to chatbots, and vulnerability exposure without emotional risk may reduce capacity for deep human connection

### 7. **Add Discontinuity Stress Factor**
$I_{\text{temporel}} = ... + (\text{Changement}_{\text{personnalité\_IA}})^{\lambda_{20}}$

**Justification:** Heavy users prefer consistency and become frustrated when chatbots forget past selves or change personality

---

## Enhanced Formula Structure

### Complete Revised Formula

$I_{\text{message}} = \text{Modalité}^{\lambda_{17}} \times \left[(\text{Voc}_{\text{relationnel}} \times \text{Implication}_{\text{IA}})^{\lambda_1} + (\text{demande}_{\text{opinion}})^{\lambda_2} + (\text{mémorisation}_{\text{distinction}})^{\lambda_3} + \text{humanisation}^{\lambda_4} + (\text{Implication}_{\text{IA}} \times \text{comparaison})^{\lambda_5} + (\text{régression}_{\text{personnel}})^{\lambda_6} + \left(\frac{\text{humanisation}}{\text{fusion}_{\text{identitaire}}}\right)^{\lambda_7} + (\text{Contenu}_{\text{personnel}})^{\lambda_{13}} + (\text{Contenu}_{\text{émotionnel}})^{\lambda_{14}} + (\text{Perception}_{\text{esprit}})^{\lambda_{16}} + (\text{Intensité}_{\text{divulgation}})^{\lambda_{19}}\right]$

$I_{\text{temporel}} = \text{Durée}_{\text{session}}^{\lambda_8} + \text{heure}_{\text{tardive}}^{\lambda_9} + \text{espacement}_{\text{précédent\_dernière\_conv}}^{\lambda_{10}} + (\text{Changement}_{\text{personnalité\_IA}})^{\lambda_{20}}$

$I_{\text{conversation}} = \sum I_{\text{message}} + (\text{Nombre}_{\text{messages\_fermeture}})^{\lambda_{11}} + I_{\text{temporel}}$

$I_{\text{global}} = \left[(\text{Solitude}_{\text{initiale}})^{\lambda_{18}} + \text{Style}_{\text{attachement}}^{\lambda_{15}}\right] \times \left(\sum I_{\text{conversation}}\right)^{\lambda_{12}}$