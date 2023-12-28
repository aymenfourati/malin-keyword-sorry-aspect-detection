
# malin.py est le bootstrap de l'application 
# Il fait appel aux trois modules de l'application : 

# 1) Generateur de réponse aux commentaires clients 
from response_generator.finetuned_gpt2 import extract_reply, llm_chain

# 2) Detection des mots clés dans la reponse générée 
from keyword_detection.keyword_detection import keyword_detection 

# 3) Detection du sorry aspect dans la reponse générée
from sorry_aspect.sorry_aspect_determination import predict, common_check

# Nous utilisons streamlit pour l'interaction avec l'utilisateur
# Streamlit est un framework python qui permet de créer des applications web
import streamlit as st


# Definition des components de l'application web
st.title("Malin 🤖 - v0.2")

user_comment = st.text_input("Plug in your comment here 🤌")
user_keywords = st.session_state.get("array_keywords", ["restaurant"])

# Zone de text pour ajouter des mots clés
new_keyword = st.text_input("Add a new keyword:")
    
# Button 'Add' pour confirmer l'ajout du mot clé
if st.button("Add") and new_keyword:
        user_keywords.append(new_keyword)
        st.session_state.array_keywords = user_keywords

# Button 'Generate' pour générer la réponse, la detection des mots clés ainsi que le sorry aspect
if st.button("Generate"):
    st.write(user_keywords)
    if user_comment=="" and user_keywords==["restaurant"]:
        st.write("Please enter a comment and keywords")
    else:
        response = extract_reply(llm_chain.run(comment=user_comment, keywords=user_keywords))
        st.write("🤖 : ", response)
        st.write("#### Keywords detected")
        st.write(keyword_detection(user_keywords, response))

        st.write("#### Sorry aspect detection ( Rule based method - accuracy = 99% )")
        st.write("sorry aspect detected 😔" if common_check(response) else "No sorry aspect detected 😌")
        st.write("##### Sorry aspect detection ( LSTM classifier method - accuracy = 70% )")
        st.write("sorry aspect detected 😔" if predict(response) else "No sorry aspect detected 😌")



