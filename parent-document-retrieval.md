# Parent Document Retrieval

문서를 크기에 따라 parent chunk와 child chunk로 나누어서 child chunk를 찾은후에 LLM의 context에는 parent chunk를 사용하면, 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다. 

## Parent/Child Chunking

문서 또는 이미지 파일에서 텍스트를 추출시 parent와 child로 chunking을 수행합니다. 상세한 코드는 [lambda-document](./lambda-document-manager/lambda_function.py)을 참조합니다.

먼저 parent/child splitter를 지정합니다. parent의 경우에 전체 문서를 나눠야하므로 separator로 개행이나 마침표와 같은 단어를 기준으로 합니다. 여기서는, chunk_size는 2000으로 하였으나 목적에 맞게 조정할 수 있습니다. chunk_size가 크면 LLM이 충분한 정보를 가질수 있으나, 전체적으로 token 소모량이 증가하고, 문서의 수(top_k)를 많이 넣으면 LLM에 따라서는 context window를 초과할 수 있어 주의가 필요합니다. child splitter의 경우는 여기서는 400을 기준으로 50의 overlap을 설정하였습니다. child의 경우에 관련된 문서를 찾는 기준이 되나 실제 사용은 parent를 사용하므로 검색이 잘되도록 하는것이 중요합니다. child의 경우는 하나의 문장의 일부가 될 수 있어야 하고 제목이 하나의 chunk를 가져가면 안되므로, 아래와 같이 개행문자등을 separator로 등록하지 않았습니다. 

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    # separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
```

분리된 parent/child chunk들을 OpenSearch에 등록합니다. OpenSearch에 등록할때 사용하는 문서의 id는 OpenSearch의 성능에 영향을 줄 수 있으며, child가 가지고 있는 parent의 id를 기반으로 검색할 수 있어야 하므로, OpenSearch가 생성하는 id를 이용합니다. 아래와 같이 먼저 parent가 되는 chunk의 metadata에 doc_level을 parent로 설정한 후에 OpenSearch에 add_documents()로 등록합니다. 이때 child는 생성된 id를 parent의 id로 활용하여 아래와 같이 metadata에 parent_doc_id를 등록합니다. 

```python
parent_docs = parent_splitter.split_documents(docs)
    if len(parent_docs):
        for i, doc in enumerate(parent_docs):
            doc.metadata["doc_level"] = "parent"
            print(f"parent_docs[{i}]: {doc}")
                    
        parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
        print('parent_doc_ids: ', parent_doc_ids)
                
        child_docs = []
        for i, doc in enumerate(parent_docs):
            _id = parent_doc_ids[i]
            sub_docs = child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata["parent_doc_id"] = _id
                _doc.metadata["doc_level"] = "child"
            child_docs.extend(sub_docs)
        # print('child_docs: ', child_docs)
                
        child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)
        print('child_doc_ids: ', child_doc_ids)
                    
        ids = parent_doc_ids+child_doc_ids
```

## Metadata 정보 파일 생성

사용자가 문서 삭제나 업데이트를 할 경우에 OpenSearch에 저장된 파일정보를 같이 삭제하여야 합니다. 이때는 parent_doc_ids와 child_doc_ids를 이용하여야 하므로 파일을 읽을때 아래와 같이 metadata를 저장하는 json 파일을 생성하여 id를 저장합니다.

```python
def create_metadata(bucket, key, meta_prefix, s3_prefix, uri, category, documentId, ids):
    title = key
    timestamp = int(time.time())

    metadata = {
        "Attributes": {
            "_category": category,
            "_source_uri": uri,
            "_version": str(timestamp),
            "_language_code": "ko"
        },
        "Title": title,
        "DocumentId": documentId,      
        "ids": ids  
    }
    print('metadata: ', metadata)
    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])

    client = boto3.client('s3')
    try: 
        client.put_object(
            Body=json.dumps(metadata), 
            Bucket=bucket, 
            Key=meta_prefix+objectName+'.metadata.json' 
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")
```

파일이 삭제되거나 새로운 파일로 업데이트가 되면, 아래처럼 OpenSearch의 id 리스트를 읽은 후에 OpenSearch의 delete()를 이용해 삭제합니다.

```python
def delete_document_if_exist(metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        print('objs: ', objs)
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            print('meta: ', meta)
            
            ids = json.loads(meta)['ids']
            print('ids: ', ids)
            
            result = vectorstore.delete(ids)
            print('result: ', result)        
        else:
            print('no meta file: ', metadata_key)
```

## Parent document의 활용

OpenSearch에 RAG 정보를 요청할 때에 아래와 같이 pre_filter로 doc_level이 child인 문서들을 검색합니다. 상세한 코드는 [lambda-chat](./lambda-chat-ws/lambda_function.py)을 참조합니다.

필터를 적용하여 질문과 가장 가까운 child 문서들을 검색한 후에 metadata에서 parent_doc_id을 추출합니다. 하나의 parent 문서에서 여러개의 child 문서가 나오는데, 검색시 하나의 parent에서 여러개의 child가 가장 가까운 문서가 될수 있습니다. 따라서 아래와 같이 child 검색시 더 많은 문서를 검색하고, relevant_documents를 선정할 때에 같은 parent_doc_id인지 확인하여 제외합니다.

```python
def get_documents_from_opensearch(vectorstore_opensearch, query, top_k):
    result = vectorstore_opensearch.similarity_search_with_score(
        query = query,
        k = top_k*2,  
        pre_filter={"doc_level": {"$eq": "child"}}
    )
    print('result: ', result)
            
    relevant_documents = []
    docList = []
    for re in result:
        if 'parent_doc_id' in re[0].metadata:
            parent_doc_id = re[0].metadata['parent_doc_id']
            doc_level = re[0].metadata['doc_level']
            print(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                    
            if doc_level == 'child':
                if parent_doc_id in docList:
                    print('duplicated!')
                else:
                    relevant_documents.append(re)
                    docList.append(parent_doc_id)
                    
                    if len(relevant_documents)>=top_k:
                        break
                                
    # print('lexical query result: ', json.dumps(response))
    print('relevant_documents: ', relevant_documents)
    
    return relevant_documents
```    

선택된 child 문서의 parent 문서를 찾아서 relevant_documents로 활용합니다. 이를 위해 parent_doc_id를 추출한 후에, boto3로 OpenSearch의 get()을 이용해 문서 정보를 가져옵니다. 가져온 정보중에 문서의 text등을 추출하여 활용합니다.

```python
relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, keyword, top_k)

for i, document in enumerate(relevant_documents):
    print(f'## Document(opensearch-vector) {i+1}: {document}')
    
    parent_doc_id = document[0].metadata['parent_doc_id']
    doc_level = document[0].metadata['doc_level']
    print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
        
    excerpt, name, uri, doc_level = get_parent_document(parent_doc_id) # use pareant document
    print(f"parent: name: {name}, uri: {uri}, doc_level: {doc_level}")
    
    answer = answer + f"{excerpt}, URL: {uri}\n\n"

def get_parent_document(parent_doc_id):
    response = os_client.get(
        index="idx-rag", 
        id = parent_doc_id
    )
    
    source = response['_source']                                
    metadata = source['metadata']    
    
    return source['text'], metadata['name'], metadata['uri'], metadata['doc_level']    
```


## Others 

### LangChain의 Parent Document Retriever

[(Parent Document Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/)에서는 In-memory store를 이용하여 parent-child chunking을 구현하는 방법을 설명하고 있습니다. AWS Lambda와 같은 경우는 event가 일정시간 없으면 초기화되므로 In-memory store를 사용할 수 없습니다. 

### Parent child splitter 활용

[MongoDB Parent Document Retrieval over your data with Amazon Bedrock](https://medium.com/@dminhk/mongodb-parent-document-retrieval-over-your-data-with-amazon-bedrock-0ecf1db9d999)와 같이 parent-child spliteer를 이용합니다.


