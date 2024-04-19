import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader  # Assurez-vous que c'est le bon import

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            return tmpfile.name
    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement du fichier: {e}")
        return None

def main():
    st.title("Extracteur et Résumeur de PDF")
    
    # Widget pour uploader un fichier PDF
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=['pdf'])
    
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        
        if file_path:
            try:
                start_page = st.number_input("Page de début (indexation à base zéro)", min_value=0, value=211)
                end_page = st.number_input("Dernière page à inclure (indexation à base zéro)", min_value=0, value=218)
                system_prompt = st.text_area("Message système", value="Fais moi un résumé du livre entre les pages {start_page} et {end_page} pour me donner l'idée globale de cette partie en français.")
                assistant_prompt = st.text_area("Réponse initiale de l'assistant", value="En tant qu'assistant de résumé de livres, je vais faire de mon mieux pour résumer le contenu clé de manière concise et informative.")

                submit_button = st.button("Exécuter")

                if submit_button:
                    # Charge les variables d'environnement depuis `.env`
                    load_dotenv()

                    # Accède à la variable d'environnement
                    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

                    # Initialisation du chat avec LangChain Anthropic
                    chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

                    # Chargement du document PDF
                    loader = PyPDFLoader(file_path)
                    pages = loader.load_and_split()

                    # Validation des indices de pages
                    if start_page < 0 or end_page >= len(pages) or start_page > end_page:
                        st.error("Erreur: Les indices de page spécifiés sont invalides.")
                        return

                    # Extraire les pages souhaitées
                    selected_pages = pages[start_page:min(end_page + 1, len(pages))]
                    texte = " ".join(page.page_content for page in selected_pages)

                    # Création du template de dialogue
                    system = system_prompt.format(start_page=start_page, end_page=end_page)
                    human = "{text}"
                    assistant = assistant_prompt

                    # Création du template de dialogue
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system),
                        ("human", human),
                        ("assistant", assistant)
                    ], template_format='f-string')

                    # Configuration de la chaîne de traitement avec le template de dialogue
                    chain = prompt | chat

                    # Préparer les données d'entrée pour le modèle
                    input_data = {
                        "text": texte,
                        "start_page": start_page,
                        "end_page": end_page
                    }

                    # Envoyer le dictionnaire au modèle pour invoquer le traitement
                    response = chain.invoke(input_data)

                    # Afficher la réponse
                    st.text_area("Réponse", response.content, height=300)

            finally:
                # Nettoyer le fichier temporaire
                if file_path:
                    os.unlink(file_path)

if __name__ == "__main__":
    main()