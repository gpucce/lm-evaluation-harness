
def process_docs_abs(ds):

    def _helper(doc):
        prompt = """Assegna un valore di astrazione da 1 a 5 alla parola {parola} nel contesto della frase seguente:
{frase}
Descrizione dei valori:
1 - La parola è estremamente concreta (e.g. un cane specifico)
2 - La parola è lievemente concreta (e.g. un cane di una certa razza)
3 - La parola è neutra (e.g. un cane tra tanti)
4 - La prola è lievemente astratta (e.g. un cane è un animale da compagnia)
5 - La parola è estremamente astratta (e.g. il cane è un mammifero)
"""
        prompt = prompt.format(parola=doc["target_token"], frase=doc["text"])
        doc["prompt"] = prompt
        doc["abs"] = round(5 * doc["abs_mean"])
        if doc["abs"] == 5:
            doc["abs"] = 4
        doc["label"] = doc["abs"]
        doc["choices"] = ["1", "2", "3", "4", "5"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object


def process_docs_inc(ds):

    def _helper(doc):
        prompt = """Assegna un valore di inclusività da 1 a 5 alla parola {parola} nel contesto della frase seguente:
{frase}
Descrizione dei valori:
1 - La parola è estremamente specifica (e.g. un cane specifico)
2 - La prola è lievemente specifica (e.g. un cane di una certa razza)
3 - La parola è neutra (e.g. un cane tra tanti)
4 - La parola è lievemente inclusiva (e.g. un cane è un animale da compagnia)
5 - La parola è estremamente inclusiva (e.g. il cane è un mammifero)
"""
        prompt = prompt.format(parola=doc["target_token"], frase=doc["text"])
        doc["prompt"] = prompt
        doc["inc"] = round(5 * doc["inc_mean"])
        if doc["inc"] == 5:
            doc["inc"] = 4
        doc["label"] = doc["inc"]
        doc["choices"] = ["1", "2", "3", "4", "5"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

# def list_fewshot_samples():
#     return {}