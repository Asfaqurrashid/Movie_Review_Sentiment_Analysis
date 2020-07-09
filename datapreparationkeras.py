from keras.preprocessing.text import Tokenizer

# 5 docs
docs = ['Well done!',
        'Good work',
        'Great effort',
        'Nice work',
        'Excellent!']
t = Tokenizer()
t.fit_on_texts(docs)

print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)
