from datamaestro import prepare_dataset

# Returns Documents object
documents = prepare_dataset("irds.antique.train").documents

for _, document in zip(range(20), documents.iter()):
    print(document.docid, document.text)  # (qid, text) tuple, noqa: T201
