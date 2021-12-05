from datamaestro import prepare_dataset

queries = prepare_dataset("irds.aquaint.trec-robust-2005.queries")
for query in queries.iter():
    print(query.qid, query.text, query.metadata)  # (qid, text, metadata) tuple
