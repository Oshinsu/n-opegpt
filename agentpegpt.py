import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import requests
import os

# Définir les modèles Pydantic pour structurer les données
class LieuTravail(BaseModel):
    libelle: str = Field(description="Libellé du lieu de travail")
    latitude: float = Field(description="Latitude du lieu de travail")
    longitude: float = Field(description="Longitude du lieu de travail")
    codePostal: str = Field(description="Code postal du lieu de travail")
    commune: str = Field(description="Code INSEE de la commune")

class Entreprise(BaseModel):
    nom: str = Field(description="Nom de l'entreprise")
    description: str = Field(description="Description de l'entreprise")
    logo: str = Field(description="URL du logo de l'entreprise")
    url: str = Field(description="URL du site de l'entreprise")
    entrepriseAdaptee: bool = Field(description="Indique si l'entreprise est adaptée")

class Salaire(BaseModel):
    libelle: str = Field(description="Libellé du salaire")
    commentaire: Optional[str] = Field(None, description="Commentaire sur le salaire")
    complement1: Optional[str] = Field(None, description="Complément 1 de rémunération")
    complement2: Optional[str] = Field(None, description="Complément 2 de rémunération")

class OffreEmploi(BaseModel):
    id: str = Field(description="Identifiant de l'offre d'emploi")
    intitule: str = Field(description="Intitulé de l'offre")
    description: str = Field(description="Description de l'offre")
    dateCreation: str = Field(description="Date de création de l'offre")
    dateActualisation: str = Field(description="Date de dernière actualisation de l'offre")
    lieuTravail: LieuTravail
    romeCode: str = Field(description="Code ROME de l’offre")
    romeLibelle: str = Field(description="Libellé associé au code ROME")
    appellationlibelle: str = Field(description="Libellé de l’appellation ROME de l’offre")
    entreprise: Entreprise
    typeContrat: str = Field(description="Code du type de contrat proposé")
    typeContratLibelle: str = Field(description="Libellé du type de contrat proposé")
    salaire: Salaire
    alternance: bool = Field(description="Indique si l'offre est pour de l'alternance")
    nombrePostes: int = Field(description="Nombre de postes disponibles pour cette offre")
    accessibleTH: bool = Field(description="Vrai si l’offre est accessible aux travailleurs handicapés")

# Initialisation des variables de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token" not in st.session_state:
    st.session_state.token = None

# Fonction pour obtenir le token d'accès
def obtenir_token(client_id, client_secret):
    url = "https://entreprise.pole-emploi.fr/connexion/oauth2/access_token?realm=/partenaire"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "api_offresdemploiv2 o2dsoffre"
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

# Fonction pour actualiser le token si nécessaire
def get_token():
    if st.session_state.token is None:
        st.session_state.token = obtenir_token(os.environ["POLE_EMPLOI_CLIENT_ID"], os.environ["POLE_EMPLOI_CLIENT_SECRET"])
    return st.session_state.token

# Définir les outils avec Pydantic
class SearchOffers(BaseModel):
    """Rechercher des offres d'emploi selon divers critères."""
    motsCles: str = Field(..., description="Mots-clés pour la recherche d'emploi")
    commune: Optional[str] = Field(None, description="Code INSEE de la commune")
    codeROME: Optional[str] = Field(None, description="Code ROME pour le poste")

class GetOfferDetails(BaseModel):
    """Obtenir les détails d'une offre d'emploi spécifique."""
    offer_id: str = Field(..., description="L'ID de l'offre d'emploi")

# Fonctions pour les outils
def search_offers(motsCles: str, commune: Optional[str] = None, codeROME: Optional[str] = None) -> List[dict]:
    token = get_token()
    url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    params = {
        "motsCles": motsCles
    }
    if commune:
        params["commune"] = commune
    if codeROME:
        params["codeROME"] = codeROME

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('resultats', [])[:10]
    elif response.status_code == 204:
        return []
    else:
        response.raise_for_status()

def get_offer_details(offer_id: str) -> dict:
    token = get_token()
    url = f"https://api.francetravail.io/partenaire/offresdemploi/v2/offres/{offer_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

# Définir les outils sous forme de fonctions
tools = [search_offers, get_offer_details]

# Instancier le modèle de chat avec outils liés
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
llm_with_tools = llm.bind_tools(tools)

# Définir le prompt avec un placeholder pour l'historique
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Vous êtes un assistant utile. Répondez à toutes les questions du mieux possible."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# Créer la chaîne avec le prompt et le modèle de chat
chain = prompt | llm_with_tools

# Créer l'histoire des messages
chat_history = ChatMessageHistory()

# Créer le Runnable avec l'histoire des messages
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Fonction pour exécuter les appels aux outils
def execute_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']
        if tool_name == 'search_offers':
            offers = search_offers(**args)
            # Transformer les offres en objets Pydantic
            structured_offres = [OffreEmploi(**offre) for offre in offers]
            results.append({"tool": "search_offers", "result": structured_offres})
        elif tool_name == 'get_offer_details':
            details = get_offer_details(**args)
            structured_details = OffreEmploi(**details)
            results.append({"tool": "get_offer_details", "result": structured_details})
        # Ajoutez d'autres outils ici si nécessaire
    return results

# Fonction pour résumer l'historique des messages
def summarize_messages(chain_input):
    stored_messages = chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("human", "Résume les messages de chat ci-dessus en un seul message concis."),
        ]
    )
    summarization_chain = summarization_prompt | llm

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    # Effacer l'historique et ajouter le résumé
    chat_history.clear()
    chat_history.add_message(summary_message)

    return True

# Créer une chaîne avec résumé
chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_memory
)

# Optionnel : Tronquer l'historique pour limiter les tokens
from operator import itemgetter

trimmer = trim_messages(strategy="last", max_tokens=1000, token_counter=lambda msg: len(msg.content.split()))

chain_with_trimming = (
    RunnablePassthrough.assign(chat_history=lambda history: trimmer(history))
    | prompt
    | llm_with_tools
)

chain_with_trimmed_history = RunnableWithMessageHistory(
    chain_with_trimming,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Interface Streamlit
st.set_page_config(page_title='💼 Pôle Emploi GPT')
st.title('💼 Pôle Emploi GPT')

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Interface de recherche avancée
with st.expander("🔍 Critères de Recherche Avancés"):
    with st.form("form_recherche"):
        motsCles = st.text_input("Mots Clés", value="développeur python")
        commune = st.text_input("Code INSEE de la Commune", help="Exemple: 75056 pour Paris")
        codeROME = st.text_input("Code ROME", help="Exemple: M1805")
        submit_button = st.form_submit_button("Rechercher")

    if submit_button:
        with st.spinner('Recherche des offres en cours...'):
            try:
                offres = search_offers(motsCles=motsCles, commune=commune if commune else None, codeROME=codeROME if codeROME else None)
                if not offres:
                    st.warning("Aucune offre trouvée avec les critères spécifiés.")
                else:
                    # Convertir les résultats en objets Pydantic
                    structured_offres = [OffreEmploi(**offre) for offre in offres]
                    for offre in structured_offres:
                        st.markdown(f"### {offre.intitule}")
                        st.markdown(f"**Entreprise:** {offre.entreprise.nom}")
                        st.markdown(f"**Lieu:** {offre.lieuTravail.libelle}")
                        st.markdown(f"**Type de Contrat:** {offre.typeContratLibelle}")
                        st.markdown(f"**Salaire:** {offre.salaire.libelle}")
                        st.markdown(f"**Description:** {offre.description[:500]}...")
                        st.markdown(f"**URL de Postulation:** [Postuler]({offre.url})")
                        st.markdown("---")
            except requests.exceptions.HTTPError as err:
                st.error(f"Erreur lors de la requête: {err}")
            except Exception as e:
                st.error(f"Une erreur est survenue: {e}")

# Entrée utilisateur et réponse de l'assistant avec mémoire et résumé
user_input = st.text_input("Vous: ", "")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner('Le chatbot réfléchit...'):
        try:
            # Générer les appels aux outils avec mémoire et résumé
            tool_response = chain_with_summarization.invoke({"input": user_input}, {"configurable": {"session_id": "unused"}})
            tool_calls = tool_response.tool_calls
            invalid_tool_calls = tool_response.invalid_tool_calls

            # Gérer les appels aux outils invalides
            if invalid_tool_calls:
                for invalid_call in invalid_tool_calls:
                    error_message = invalid_call.get('error', 'Erreur inconnue lors de l\'appel à l\'outil.')
                    st.error(f"Erreur dans l'appel à l'outil '{invalid_call.get('name', 'inconnu')}': {error_message}")

            # Exécuter les outils si des appels sont générés
            if tool_calls:
                tool_results = execute_tool_calls(tool_calls)
                # Traiter les résultats des outils et formuler une réponse
                response_content = "Voici les résultats de votre recherche :\n"
                for result in tool_results:
                    if result["tool"] == "search_offers":
                        if not result["result"]:
                            response_content += "Aucune offre trouvée avec les critères spécifiés.\n"
                        else:
                            for offre in result["result"]:
                                response_content += f"### {offre.intitule}\n"
                                response_content += f"**Entreprise:** {offre.entreprise.nom}\n"
                                response_content += f"**Lieu:** {offre.lieuTravail.libelle}\n"
                                response_content += f"**Type de Contrat:** {offre.typeContratLibelle}\n"
                                response_content += f"**Salaire:** {offre.salaire.libelle}\n"
                                response_content += f"**Description:** {offre.description[:200]}...\n"
                                response_content += f"**URL de Postulation:** [Postuler]({offre.url})\n\n"
                    elif result["tool"] == "get_offer_details":
                        offre = result["result"]
                        response_content += f"### Détails de l'offre : {offre.intitule}\n"
                        response_content += f"**Entreprise:** {offre.entreprise.nom}\n"
                        response_content += f"**Lieu:** {offre.lieuTravail.libelle}\n"
                        response_content += f"**Type de Contrat:** {offre.typeContratLibelle}\n"
                        response_content += f"**Salaire:** {offre.salaire.libelle}\n"
                        response_content += f"**Description:** {offre.description}\n"
                        response_content += f"**URL de Postulation:** [Postuler]({offre.url})\n\n"

                st.session_state.messages.append({"role": "assistant", "content": response_content})
                with st.chat_message("assistant"):
                    st.markdown(response_content)
            else:
                # Si aucun appel aux outils n'est généré, utiliser la réponse directe du modèle
                response = tool_response.generations[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

            # Optionnel : Résumer l'historique après chaque interaction
            summarize_messages(None)

        except Exception as e:
            st.error(f"Erreur dans le chatbot: {e}")
