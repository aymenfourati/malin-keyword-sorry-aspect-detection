# Ce module lance les tests autmatiques implementer dans le stage : Test de la pipeline CI/CD 
# retourne True si les trois modules fonctionnent correctement sinon retourne False

from application.response_generator.finetuned_gpt2 import llm_chain, extract_reply
from application.keyword_detection.keyword_detection import keyword_detection
from application.sorry_aspect.sorry_aspect_determination import predict, common_check


def executioner(user_comment, user_keywords):

    response = extract_reply(llm_chain.run(comment=user_comment, keywords=user_keywords))
    return response and keyword_detection(user_keywords, response) and predict(response) and common_check(response)


def test():


    # Test mauvaise experience
    keywords = ["restaurant","negative experience"]
    comment = "This place is way too overpriced for mediocre food."
    result = executioner(comment, keywords)
    assert result != ""

    # Test bonne experience
    keywords = ["restaurant","good experience"]
    comment = "If you want healthy authentic or ethic food, try this place."
    result = executioner(comment, keywords)
    assert result != ""


