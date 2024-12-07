

def process_answer_mod(dataset):

    def _helper(doc):
        int_ans = int(doc["answer"].split("### ")[-1].replace(",", ""))
        mod_answer = f"{int_ans}*(\\sin^2(x) + \\cos^2(x))"
        doc["answer"] = doc["answer"].replace(f"### {int_ans}", f"### {mod_answer}")
        return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object