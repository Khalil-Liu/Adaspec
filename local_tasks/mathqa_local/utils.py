import re


def doc_to_choice(doc):
    return [
        choice[4:].rstrip(" ,")
        for choice in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
