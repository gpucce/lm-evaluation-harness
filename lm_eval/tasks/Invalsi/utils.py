from datasets import disable_caching

disable_caching()

def process_docs(dataset):
    # def _helper(doc):
    #   # modifies the contents of a single
    #   # document in our dataset.
    #   doc["choices"] = [doc["choice1"], doc["choice2"], doc["wrong_answer"]]
    #   doc["gold"] = doc["label"]
    #   return doc
    def _helper(doc):
        prompt = ""
        if len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object


def process_docs_mate_multipla(dataset):

    # INVALSI_MATE_MISTRAL_INSTRUCT = {
    #     'completa frase': "[INST] Dato un testo (TESTO) completalo come richiesto e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    #     'multipla': "[INST] Dato un testo (TESTO) scegli la risposta alla domanda (DOMANDA) seguente tra quelle disponibili e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    #     'numero': "[INST] Dato un testo (TESTO) calcola il numero come richiesto nella domanda (DOMANDA) alla fine dai come risposta finale solo il numero. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    #     'vero/falso': "[INST] Dato un testo (TESTO) indica se la frase (FRASE) che lo segue è vera o falsa e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nFRASE:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    # }

    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    def _helper(doc):
        prompt = ""
        if len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = "ABCD".index(doc["risposta"])
        doc["choice"] = ["A", "B", "C", "D"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_mate_verofalso(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "vero/falso")
    def _helper(doc):
        prompt = ""
        if len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = "VF".index(doc["risposta"])
        doc["choice"] = ["vero", "falso"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_mate_numero(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "numero")
    def _helper(doc):
        prompt = ""
        if len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = doc["risposta"]
        doc["choice"] = [doc["risposta"], doc["alt1"], doc["alt2"], doc["alt3"]]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_ita_multipla(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = int("ABCD".index(doc["risposta"]))
        doc["choices"] = ["A", "B", "C", "D"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_ita_binarie(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = 0
        doc["choices"] = [doc["risposta"], "WRONG"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object