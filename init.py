from dotenv import load_dotenv      # python-dotenv
import os
from datetime import datetime
from werkzeug.security import generate_password_hash
from langchain.chains.query_constructor.base import AttributeInfo
from flask_httpauth import HTTPBasicAuth
from Processing.qna_lch_mod import KBAQnA

# Format the current date and time as DD.MM.YYYY HH:MM:SS
formatted_datetime = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
print(f"*** Start KBA {formatted_datetime} ***")

load_dotenv()

# application API Key definition
################################
users = {
    "admin": generate_password_hash(os.getenv("FLASK_ADMIN_API_KEY")),  # administrator API key
    "app": generate_password_hash(os.getenv("FLASK_APP_API_KEY")),  # application API key
}

auth = HTTPBasicAuth()

# starting KBAQnA
#################

system_msg = "You are AI Assistant named Vanda."
qa = KBAQnA(
    db_type = "qdrant",
    db_dir = "db",  
    system_msg = system_msg,
    answer_time = False,
    verbose = False,
    )

#----------------------------------------------------------------------------------------------

# setup www.mulouny.cz
######################

sekce_list = qa.get_list_from_metadata("www.mulouny.cz", "sekce")            
situace_list = qa.get_list_from_metadata("www.mulouny.cz", "situace")     

system_msg1 = """Jsi AI asistent na webu městského úřadu Louny a řešíš pouze agendy městského úřadu. \
Odpovídáš na základě Vědomostí dodaných uživatelem. \
Pokud není na základě uvedených vědomostí možné jednoznačně odpovědět na otázku, odpověz "Nevím".

Příklad, pokud informace není obsažena ve vědomostech:
Uživatel: kde najdu vysokou školu v Lounech?
Asistent: Nevím"""

self_doc_descr1 = "Informace o službách (životních situacích) obsluhovaných městským úřadem Louny."

sekce = ""
for item in sekce_list:
    sekce += f"'{item}', "  

situace = ""
for item in situace_list:
    situace += f"'{item}', "    

self_metadata1 = [
    AttributeInfo(
        name="sekce",
        description="Název sekce, do které spadá služba (životní situace). Pouze jedna z [" + sekce + "].",
        type="string",
    ),
    AttributeInfo(
        name="situace",
        description="Název služby. Pouze jedna z [" + situace + "].",
        type="string",
    ),
]


qa.set_project_par(project="www.mulouny.cz", system_msg = system_msg1, api_model="gpt35_1106", citation=False, self_doc_descr = self_doc_descr1,
                   self_metadata = self_metadata1, k = 20, metadata_parent_field = "cislo_zs")
# (EmbeddingRetriever, SelfRetriever, BM25, MultiRetriever, SelfRetrieverParent)
qa.set_project_retriever(project="www.mulouny.cz", retriever_weights= (0, 0, 0, 0, 1))

# setup www.multima.cz
#######################
system_msg2 = """Jsi AI asistent na webu Multima a odpovídáš pouze na dotazy, které se týkají Multimy. \
Odpovídáš na základě Vědomostí dodaných uživatelem. \
Pokud není na základě uvedených vědomostí možné jednoznačně odpovědět na otázku, odpověz "Nevím".

Příklad, pokud informace není obsažena ve vědomostech:
Uživatel: kde je v Pardubicích železniční stanice?
Asistent: Nevím"""

self_doc_descr2 = "Informace o produktech, službách a aktivitách společnosti Multima."

self_metadata2 = [
    AttributeInfo(
        name="subject",
        description="Jaké jsou převažující informace v textu. Pouze jeden z ['Produkty', 'Služby', 'Kontaktní informace', 'Kariéra', 'Informace o firmě', 'Jiné'].",
        type="string",
    ),
    AttributeInfo(
        name="price_list",
        description="Zda je v textu obsažen ceník.",
        type="boolean",
    ),
    AttributeInfo(
        name="product",
        description="Název produktu nabízeného Multimou. Pouze jeden z ['Nathan AI', 'Dokladovna', 'Keymate', 'Odtahovka', 'Mentor', 'Řízená dokumentace', 'Multiskills'].",
        type="string",
    ),
    AttributeInfo(
        name="service",
        description="Název služby nabízené Multimou. Pouze jedna z ['Vývoj software', 'Integrace', 'AI - umělá inteligence', 'Cloudifikace', 'Powerapps', 'Sharepoint', 'Správa obsahu v Microsoft 365'].",
        type="string",
    ),
    AttributeInfo(
        name="case_study",
        description="Název případové studie nabízené Multimou. Pouze jedna z ['Pojišťovny', 'Farmacie'].",
        type="string",
    ),
]

qa.set_project_par(project="www.multima.cz", system_msg = system_msg2, api_model="gpt35_1106", citation=False, self_doc_descr = self_doc_descr2,
                   self_metadata = self_metadata2, k=10)
# (EmbeddingRetriever, SelfRetriever, BM25, MultiRetriever, SelfRetrieverParent)
qa.set_project_retriever(project="www.multima.cz", retriever_weights= (0, 1, 0, 0, 0))


# setup PRE
#######################
system_msg3 = """Jsi AI asistent pro normy Pražské energetiky. \
Odpovídáš na základě Vědomostí dodaných uživatelem. \
Pokud není na základě uvedených vědomostí možné jednoznačně odpovědět na otázku, odpověz "Nevím".

Příklad, pokud informace není obsažena ve vědomostech:
Uživatel: kde je v Pardubicích železniční stanice?
Asistent: Nevím"""


self_doc_descr3 = "Informace o normách Pražské energetiky."

self_metadata3 = [
    AttributeInfo(
        name="CN",
        description="Číslo normy ve tvaru AA NNN. AA jsou 2 velká písmena, NNN trojmístné číslo.",
        type="string",
    ),
    AttributeInfo(
        name="Nazev",
        description="Název normy.",
        type="string",
    ),
    AttributeInfo(
        name="Ucel",
        description="Jaké oblasti se norma týká. Popis je uveden v kapitole: A.2",
        type="string",
    ),
    AttributeInfo(
        name="Klice",
        description="Seznam klíčových slov normy. Je uveden v kapitole: A.3 Klíčová slova",
        type="string",
    ),
]
    


qa.set_project_par(project="PRE", system_msg = system_msg3, api_model="gpt35_1106", citation=True,
                   self_doc_descr = self_doc_descr3, self_metadata = self_metadata3, k = 10,
                   routing_field = "CN", citation_field = "CN"
                   )
# (EmbeddingRetriever, SelfRetriever, BM25, MultiRetriever, SelfRetrieverParent)
qa.set_project_retriever(project="PRE", retriever_weights= (0, 1, 0, 0, 0))











