from datamaestro import prepare_dataset

# Returns AdhocDocuments object
documents = prepare_dataset("irds.antique.train").documents

for _, document in zip(range(20), documents.iter()):
    print(document.docid, document.text)  # (qid, text) tuple
