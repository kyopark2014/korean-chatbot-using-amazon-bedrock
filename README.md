# RAG를 활용하여 향상된 Korean Chatbot 만들기

RAG(Retrieval-Augmented Generation)를 활용하면, LLM(Large Language Model)의 기능을 강화하여 다양한 어플리케이션을 개발할 수 있습니다. 여기에서는 RAG의 성능을 향상시키는 방법들에 대해 설명하고 이를 이용하여 기업 또는 개인의 데이터를 쉽게 활용할 수 있는 한국어 Chatbot을 만들고자 합니다. 

- Multimodal: 텍스트뿐 아니라 이미지에 대한 분석을 할 수 있습니다.
- Multi-RAG: 다양한 [지식 저장소(Knowledge Store)](https://aws.amazon.com/ko/about-aws/whats-new/2023/09/knowledge-base-amazon-bedrock-models-data-sources/)활용합니다. 
- Multi-Region LLM: 여러 리전에 있는 LLM을 동시에 활용함으로써 질문후 답변까지의 동작시간을 단축하고, On-Demand 방식의 동시 실행 수의 제한을 완화할 수 있습니다.
- Agent: 외부 API를 통해 얻어진 결과를 대화시 활용합니다. 
- 인터넷 검색: RAG의 지식저장소에 관련된 문서가 없는 경우에 인터넷 검색을 통해 활용도를 높입니다.
- 한영 동시 검색: RAG에 한국어와 영어 문서들이 혼재할 경우에 한국어로 영어 문서를 검색할 수 없습니다. 한국어로 한국어, 영어 문서를 모두 검색하여 RAG의 성능을 향상 시킬 수 있습니다.
- Prioroty Search: 검색된 문서를 관련도에 따라 정렬하면 LLM의 결과가 향상됩니다.
- Kendra 성능 향상: LangChain에서 Kendra의 FAQ를 활용합니다.
- Vector/Keyword 검색: Vector 검색(Sementaic) 뿐 아니라, Lexical 검색(Keyword)을 활용하여 관련된 문서를 찾을 확율을 높입니다.
- Code Generation: 기존 코드를 이용하여 Python/Node.js 코드를 생성할 수 있습니다.
  
여기서 구현한 코드들은 [LangChain](https://aws.amazon.com/ko/what-is/langchain/)을 기반으로 합니다. 또한, 아래와 같은 Prompt Engineing 예제를 사용해 볼 수 있습니다.

- 번역 (translation): 입력된 문장을 번역합니다.
- 문법 오류 추출 (Grammatical Error Correction): 영어에 대한 문장 에러를 설명하고, 수정된 문장을 보여줍니다.
- 리뷰 분석 (Extracted Topic and Sentiment): 입력된 리뷰의 주제와 감정(Sentiment)을 추출합니다.
- 정보 추출 (Information Extraction): 입력된 문장에서 email과 같은 정보를 추출합니다.
- 개인 정보 삭제 (Removing PII): 입력된 문장에서 개인정보(PII)를 삭제할 수 있습니다.
- 복잡한 질문 (Complex Question): step-by-step으로 복잡한 문제를 해결합니다.
- 어린이와 대화 (Child Conversation): 대화상대에 맞게 적절한 어휘나 답변을 할 수 있습니다.
- 시간정보 추출 (Timestamp Extraction): 입력된 정보에서 시간정보(timestemp)를 추출합니다.
- 자유로운 대화 (Free Conversation): 친구처럼 반말로 대화합니다.


## 아키텍처 개요

전체적인 아키텍처는 아래와 같습니다. 사용자의 질문은 WebSocket을 이용하여 AWS Lambda에서 RAG와 LLM을 이용하여 답변합니다. 대화 이력(chat history)를 이용하여 사용자의 질문(Question)을 새로운 질문(Revised question)으로 생성합니다. 새로운 질문으로 지식 저장소(Knowledge Store)인 Kendra와 OpenSearch에 활용합니다. 두개의 지식저장소에는 용도에 맞는 데이터가 입력되어 있는데, 만약 같은 데이터가 가지고 있더라도, 두개의 지식저장소의 문서를 검색하는 방법의 차이로 인해, 서로 보완적인 역할을 합니다. 지식저장소에 한국어/한국어로 된 문서들이 있다면, 한국어 질문은 영어로 된 문서를 검색할 수 없습니다. 따라서 질문이 한국어라면 한국어로 한국어 문서를 먼저 검색한 후에, 영어로 번역하여 다시 한번 영어 문서들을 검색합니다. 이렇게 함으로써 한국어로 질문을 하더라도 영어 문서까지 검색하여 더 나은 결과를 얻을 수 있습니다. 만약 두 지식저장소가 관련된 문서(Relevant documents)를 가지고 있지 않다면, Google Search API를 이용하여 인터넷에 관련된 웹페이지들이 있는지 확인하고, 이때 얻어진 결과를 RAG처럼 활용합니다. 

<img src="https://github.com/user-attachments/assets/9cd581dc-00de-4790-aa77-f9f56a3c8f9d" width="800">



상세하게 단계별로 설명하면 아래와 같습니다.

단계 1: 사용자의 질문(question)은 API Gateway를 통해 Lambda에 Web Socket 방식으로 전달됩니다. Lambda는 JSON body에서 질문을 읽어옵니다. 이때 사용자의 이전 대화이력이 필요하므로 [Amazon DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html)에서 읽어옵니다. DynamoDB에서 대화이력을 로딩하는 작업은 처음 1회만 수행합니다.

단계 2: 사용자의 대화이력을 반영하여 사용자와 Chatbot이 interactive한 대화를 할 수 있도록, 대화이력과 사용자의 질문으로 새로운 질문(Revised Question)을 생성하여야 합니다. LLM에 대화이력(chat history)를 Context로 제공하고 적절한 Prompt를 이용하면 새로운 질문을 생성할 수 있습니다.

단계 3: 새로운 질문(Revised question)으로 OpenSearch에 질문을 하여 관련된 문서(Relevant Documents)를 얻습니다. 

단계 4: 질문이 한국어인 경우에 영어 문서도 검색할 수 있도록 새로운 질문(Revised question)을 영어로 번역합니다.

단계 5: 번역된 새로운 질문(translated revised question)을 이용하여 Kendra와 OpenSearch에 질문합니다.

단계 6: 번역된 질문으로 얻은 관련된 문서가 영어 문서일 경우에, LLM을 통해 번역을 수행합니다. 관련된 문서가 여러개이므로 Multi-Region의 LLM들을 활용하여 지연시간을 최소화 합니다.

단계 7: 한국어 질문으로 얻은 N개의 관련된 문서와, 영어로 된 N개의 관련된 문서의 합은 최대 2xN개입니다. 이 문서를 가지고 Context Window 크기에 맞도록 문서를 선택합니다. 이때 관련되가 높은 문서가 Context의 상단에 가도록 배치합니다.

단계 8: 관련도가 일정 이하인 문서는 버리므로, 한개의 RAG의 문서도 선택되지 않을 수 있습니다. 이때에는 Google Seach API를 통해 인터넷 검색을 수행하고, 이때 얻어진 문서들을 Priority Search를 하여 관련도가 일정 이상의 결과를 RAG에서 활용합니다. 

단계 9: 선택된 관련된 문서들(Selected relevant documents)로 Context를 생성한 후에 새로운 질문(Revised question)과 함께 LLM에 전달하여 사용자의 질문에 대한 답변을 생성합니다.

이때의 Sequence diagram은 아래와 같습니다. 만약 RAG에서 관련된 문서를 찾지못할 경우에는 Google Search API를 통해 Query를 수행하여 RAG처럼 활용합니다. 대화이력을 가져오기 위한 DynamoDB는 첫번째 질문에만 해당됩니다. 여기서는 "us-east-1"과 "us-west-2"의 Bedrock을 사용하므로, 아래와 같이 질문마다 다른 Region의 Bedrock Claude LLM을 사용합니다.

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/251d2666-8837-4e8b-8521-534cbd3ced53" width="1000">

대량으로 파일 업로드 또는 삭제시는 아래와 같은 Event driven구조를 활용할 수 있습니다. 이를 통해 S3로 대규모로 문서 또는 코드를 넣을때에 정보의 유출없이 RAG의 지식저장소를 데이터를 주입할 수 있습니다. 

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/eaace9e5-9d4e-4bdc-aa67-bad935dffaa3" width="900">


## 향상된 RAG 구현하기

### Multi-RAG

여러개의 RAG를 활용할 경우에 요청후 응답까지의 지연시간이 증가합니다. 따라서 병렬 프로세싱을 이용하여 동시에 지식 저장소에 대한 질문을 수행하여야 합니다. 상세한 내용은 관련된 Blog인 [Multi-RAG와 Multi-Region LLM로 한국어 Chatbot 만들기](https://aws.amazon.com/ko/blogs/tech/multi-rag-and-multi-region-llm-for-chatbot/)를 참조합니다. 

```python
from multiprocessing import Process, Pipe

processes = []
parent_connections = []
for rag in capabilities:
    parent_conn, child_conn = Pipe()
    parent_connections.append(parent_conn)

    process = Process(target = retrieve_process_from_RAG, args = (child_conn, revised_question, top_k, rag))
    processes.append(process)

for process in processes:
    process.start()

for parent_conn in parent_connections:
    rel_docs = parent_conn.recv()

    if (len(rel_docs) >= 1):
        for doc in rel_docs:
            relevant_docs.append(doc)

for process in processes:
    process.join()

def retrieve_process_from_RAG(conn, query, top_k, rag_type):
    relevant_docs = []
    if rag_type == 'kendra':
        rel_docs = retrieve_from_kendra(query=query, top_k=top_k)      
    else:
        rel_docs = retrieve_from_vectorstore(query=query, top_k=top_k, rag_type=rag_type)
    
    if(len(rel_docs)>=1):
        for doc in rel_docs:
            relevant_docs.append(doc)    
    
    conn.send(relevant_docs)
    conn.close()
```

### Multi-Region LLM

여러 리전의 LLM에 대한 profile을 정의합니다. 상세한 내용은 [cdk-korean-chatbot-stack.ts](./cdk-korean-chatbot/lib/cdk-korean-chatbot-stack.ts)을 참조합니다.

```typescript
const claude3_sonnet = [
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",   
    "maxOutputTokens": "4096"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
    "maxOutputTokens": "4096"
  }
];

const profile_of_LLMs = claude3_sonnet;
```

Bedrock에서 client를 지정할때 bedrock_region을 지정할 수 있습니다. 아래와 같이 LLM을 선택하면 Lambda에 event가 올때마다 다른 리전의 LLM을 활용할 수 있습니다. 

```python
from langchain_aws import ChatBedrock

profile_of_LLMs = json.loads(os.environ.get('profile_of_LLMs'))
selected_LLM = 0

def get_chat(profile_of_LLMs, selected_LLM):
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'LLM: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    maxOutputTokens = int(profile['maxOutputTokens'])
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }            
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )       
    
    return chat
```

lambda(chat)와 같이 문서를 번역할 때에서 병렬로 조회하기 위하여, [Lambda의 Multi thread](https://aws.amazon.com/ko/blogs/compute/parallel-processing-in-python-with-aws-lambda/)를 이용합니다. 이때, 병렬 처리된 데이터를 연동 할 때에는 [Pipe()](https://docs.python.org/3/library/multiprocessing.html)을 이용합니다. 

```python
def translate_relevant_documents_using_parallel_processing(docs):
    selected_LLM = 0
    relevant_docs = []    
    processes = []
    parent_connections = []
    for doc in docs:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        chat = get_chat(profile_of_LLMs, selected_LLM)
        bedrock_region = profile_of_LLMs[selected_LLM]['bedrock_region']

        process = Process(target=translate_process_from_relevent_doc, args=(child_conn, chat, doc, bedrock_region))
        processes.append(process)

        selected_LLM = selected_LLM + 1
        if selected_LLM == len(profile_of_LLMs):
            selected_LLM = 0

    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        doc = parent_conn.recv()
        relevant_docs.append(doc)    

    for process in processes:
        process.join()
    
    #print('relevant_docs: ', relevant_docs)
    return relevant_docs
```

### Embedding

[BedrockEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/bedrock)을 이용하여 Embedding을 합니다. 'amazon.titan-embed-text-v1'은 Titan Embeddings Generation 1 (G1)을 의미하며 8k token을 지원합니다. Titan Embedding v2는 "amazon.titan-embed-text-v2:0"을 사용합니다.

```python
bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    region_name = bedrock_region,
    model_id = 'amazon.titan-embed-text-v1' 
)
```

### 대화 저장 및 관리

lambda-chat-ws는 인입된 메시지의 userId를 이용하여 map_chain에 저장된 대화 이력(memory_chain)가 있는지 확인합니다. 채팅 이력이 없다면 아래와 같이 [ConversationBufferWindowMemory](https://python.langchain.com/docs/modules/memory/types/buffer_window)로 memory_chain을 설정합니다. 여기서, 

```python
map_chain = dict() 

if userId in map_chain:
    print('memory exist. reuse it!')        
    memory_chain = map_chain[userId]
        
else: 
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=10)
    map_chain[userId] = memory_chain
        
    allowTime = getAllowTime()
    load_chat_history(userId, allowTime)

msg = general_conversation(connectionId, requestId, chat, text)

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    
    history = memory_chain.load_memory_variables({})["chat_history"]
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
                            
        msg = stream.content
        print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    return msg
```

새로운 Diaglog는 아래와 같이 chat_memory에 추가합니다.

```python
memory_chain.chat_memory.add_user_message(text) 
memory_chain.chat_memory.add_ai_message(msg)
```

### Stream 처리

여기서 stream은 아래와 같은 방식으로 WebSocket을 사용하는 client에 메시지를 전달할 수 있습니다. 상세한 내용은 관련된 Blog인 [Amazon Bedrock을 이용하여 Stream 방식의 한국어 Chatbot 구현하기](https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/)을 참고합니다.

```python
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg
            }
            sendMessage(connectionId, result)
    print('msg: ', msg)
    return msg
```

여기서 client로 메시지를 보내는 sendMessage()는 아래와 같습니다. 여기서는 boto3의 [post_to_connection](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigatewaymanagementapi/client/post_to_connection.html)를 이용하여 메시지를 WebSocket의 endpoint인 API Gateway로 전송합니다.

```python
def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except: 
        raise Exception ("Not able to send a message")
```

### Priority Search (관련도 기준 문서 선택)

Multi-RAG, 한영 동시 검색, 인터넷 검색등을 활용하여 다수의 관련된 문서가 나오면, 관련도가 높은 순서대로 일부 문서만을 RAG에서 활용합니다. 이를 위해 Faiss의 similarity search를 이용합니다. 이것은 정량된 값의 관련도를 얻을 수 있어서, 관련되지 않은 문서를 Context로 활용하지 않도록 해줍니다. 

```python
selected_relevant_docs = []
if len(relevant_docs)>=1:
    selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embeddings)

def priority_search(query, relevant_docs, bedrock_embeddings):
    excerpts = []
    for i, doc in enumerate(relevant_docs):
        if doc['metadata']['translated_excerpt']:
            content = doc['metadata']['translated_excerpt']
        else:
            content = doc['metadata']['excerpt']
        
        excerpts.append(
            Document(
                page_content=content,
                metadata={
                    'name': doc['metadata']['title'],
                    'order':i,
                }
            )
        )  

    embeddings = bedrock_embeddings
    vectorstore_confidence = FAISS.from_documents(
        excerpts,  # documents
        embeddings  # embeddings
    )            
    rel_documents = vectorstore_confidence.similarity_search_with_score(
        query=query,
        k=top_k
    )

    docs = []
    for i, document in enumerate(rel_documents):

        order = document[0].metadata['order']
        name = document[0].metadata['name']
        assessed_score = document[1]

        relevant_docs[order]['assessed_score'] = int(assessed_score)

        if assessed_score < 200:
            docs.append(relevant_docs[order])    

    return docs
```

### 한영 동시 검색

한영 검색을 위해 먼저 한국어로 RAG를 조회하고, 영어로 번역한 후에 각각의 관련된 문서들(Relevant Documents)를 번역합니다. 관련된 문서들에 대해 질문에 따라 관련성을 비교하여 관련도가 높은 문서순서로 Context를 만들어서 활용합니다. 상세한 내용은 관련된 Blog인 [
한영 동시 검색 및 인터넷 검색을 활용하여 RAG를 편리하게 활용하기](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)을 참조합니다. 

```python
translated_revised_question = traslation_to_english(llm=llm, msg=revised_question)

relevant_docs_using_translated_question = retrieve_from_vectorstore(query=translated_revised_question, top_k=4, rag_type=rag_type)
            
docs_translation_required = []
if len(relevant_docs_using_translated_question)>=1:
    for i, doc in enumerate(relevant_docs_using_translated_question):
        if isKorean(doc)==False:
            docs_translation_required.append(doc)
        else:
            relevant_docs.append(doc)
                                   
    translated_docs = translate_relevant_documents_using_parallel_processing(docs_translation_required)
    for i, doc in enumerate(translated_docs):
        relevant_docs.append(doc)
```

### 인터넷 검색

Multi-RAG를 이용하여 여러개의 지식 저장소에 관련된 문서를 조회하였음에도 문서가 없다면, 구글 인터넷 검색을 통해 얻어진 결과를 활용합니다. 여기서, assessed_score는 priority search시 FAISS의 Score로 업데이트 됩니다. 상세한 내용은 [Google Search API](./GoogleSearchAPI.md) 관련된 Blog인 [
한영 동시 검색 및 인터넷 검색을 활용하여 RAG를 편리하게 활용하기](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)을 참조합니다. 

```python
from googleapiclient.discovery import build

google_api_key = os.environ.get('google_api_key')
google_cse_id = os.environ.get('google_cse_id')

api_key = google_api_key
cse_id = google_cse_id

relevant_docs = []
try:
    service = build("customsearch", "v1", developerKey = api_key)
    result = service.cse().list(q = revised_question, cx = cse_id).execute()
    print('google search result: ', result)

    if "items" in result:
        for item in result['items']:
            api_type = "google api"
            excerpt = item['snippet']
            uri = item['link']
            title = item['title']
            confidence = ""
            assessed_score = ""

            doc_info = {
                "rag_type": 'search',
                "api_type": api_type,
                "confidence": confidence,
                "metadata": {
                    "source": uri,
                    "title": title,
                    "excerpt": excerpt,                                
                },
                "assessed_score": assessed_score,
            }
        relevant_docs.append(doc_info)
```


### Code Generation

RAG에 저장된 기존 코드를 이용하여 새로운 코드를 생성합니다. [rag-code-generation](https://github.com/kyopark2014/rag-code-generation)는 Code를 한국어로 요약하여 RAG에 저장하고 검색하는 방법을 설명했습니다. 여기에서는 일반 문서와 Code reference를 하나의 RAG에 저장하고 활용합니다. 

### Parent Document Retrieval

RAG의 검색정확도를 향상시키기 위한 여러가지 방법중에 Parent/Child Chunking을 이용할 수 있습니다. [Parent Document Retrieval](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/parent-document-retrieval.md)에서는 parent/child로 chunking 전략을 달리하는 방법에 대해 설명합니다. 

### S3 Event

S3에 문서를 업로드할때 발생하는 Event를 이용하여 자동으로 RAG 등록을 할 수 있습니다. 이때 필요한 event에 대해 [RAG-s3-event.md](./RAG-s3-event.md)에서 설명합니다.


## 문서의 정보 추출

### Chunking Strategy

[Chunking Strategy](./chunking_stretegy.md)에서는 문서를 분할하여 chunk를 만드는 방법에 대해 설명합니다. 

### 문서의 이미지 활용

#### 문서를 페이지 단위로 이미지 추출하기

[page-image-extraction.md](./page-image-extraction.md)에서는 문서의 페이지 단위로 저장하는 방법에 대해 설명합니다. 이미지를 단독 저장할때보다 삽입된 이미지 근처의 텍스트를 같이 추출하면 더 많은 설명을 할 수 있습니다. 

#### 문서에 포함된 이미지를 추출하기 

[image-extraction.md](./image-extraction.md)에서는 pdf, docx, pptx에서 이미지를 추출하여 S3에 저장하는 방법을 설명합니다.

또한, 이미지 추출을 enable 하기 위해서는 [cdk-korean-chatbot-stack.ts](./cdk-korean-chatbot/lib/cdk-korean-chatbot-stack.ts)를 참조하여 아래의 enableImageExtraction을 'true'로 변경합니다. 이후 [deployment.md](./deployment.md)를 참조하여, 재배포합니다. 

```python
const enableImageExtraction = 'false';
```

#### PDF에서 정보 추출하기

[PDF에서 정보 추출하기](./pdf-extraction.md)에서는 pdf에서 이미지에 대한 정보를 추출하는 방법에 대해 설명합니다.

[PDF에서 텍스트, 이미지, 테이블 정보를 추출하기](./rag-pdf.md)에서는 S3의 PDF 문서에서 텍스트, 이미지, 테이블을 추출하는 방법에 대해 설명합니다. 


## Agent 정의 및 활용

[LLM Agent](https://github.com/kyopark2014/llm-agent)와 같이, 다양한 API를 이용하기 위하여 Agent를 이용할 수 있습니다. 메뉴에서 ReAct나 ReAct chat을 이용해 기능을 확인할 수 있습니다.


### 결과 읽어주기

Amazon Polly를 이용하여 결과를 한국어로 읽어줍니다. [start_speech_synthesis_task](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/polly/client/start_speech_synthesis_task.html#)을 활용합니다.

```python
def get_text_speech(path, speech_prefix, bucket, msg):
    ext = "mp3"
    polly = boto3.client('polly')
    try:
        response = polly.start_speech_synthesis_task(
            Engine='neural',
            LanguageCode='ko-KR',
            OutputFormat=ext,
            OutputS3BucketName=bucket,
            OutputS3KeyPrefix=speech_prefix,
            Text=msg,
            TextType='text',
            VoiceId='Seoyeon'        
        )
        print('response: ', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create voice")
    
    object = '.'+response['SynthesisTask']['TaskId']+'.'+ext
    print('object: ', object)

    return path+speech_prefix+parse.quote(object)
```

## Kendra

### Kendra의 성능 향상

[Kendra 를 이용한 RAG의 구현](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra.md)에 따라 Kendra의 RAG 성능을 향상 시킬 수 있습니다. [Kendra의 FAQ](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra-faq.md)와 같이 정리된 문서를 활용하고, 관련도 기반으로 관련 문서를 선택하여 Context로 확인 합니다. Kendra에서 문서 등록에 필요한 내용은 [kendra-document.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/kendra-document.md)을 참조합니다. 또한, 상세한 내용은 관련된 Blog인 [Amazon Bedrock의 Claude와 Amazon Kendra로 향상된 RAG 사용하기](https://aws.amazon.com/ko/blogs/tech/bedrock-claude-kendra-rag/)을 참고합니다. 


### S3를 데이터 소스로 하기 위한 퍼미션 (Kendra)

Log에 대한 퍼미션이 필요합니다.

```java
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "cloudwatch:GenerateQuery",
                "logs:*"
            ],
            "Resource": "*",
            "Effect": "Allow"
        }
    ]
}
```

개발 및 테스트를 위해 Kendra에서 추가로 S3를 등록할 수 있도록 모든 S3에 대한 읽기 퍼미션을 부여합니다. 

```java
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Action": [
				"s3:Describe*",
				"s3:Get*",
				"s3:List*"
			],
			"Resource": "*",
			"Effect": "Allow"
		}
	]
}
```

이를 CDK로 구현하면 아래와 같습니다.

```typescript
const kendraLogPolicy = new iam.PolicyStatement({
    resources: ['*'],
    actions: ["logs:*", "cloudwatch:GenerateQuery"],
});
roleKendra.attachInlinePolicy( // add kendra policy
    new iam.Policy(this, `kendra-log-policy-for-${projectName}`, {
        statements: [kendraLogPolicy],
    }),
);
const kendraS3ReadPolicy = new iam.PolicyStatement({
    resources: ['*'],
    actions: ["s3:Get*", "s3:List*", "s3:Describe*"],
});
roleKendra.attachInlinePolicy( // add kendra policy
    new iam.Policy(this, `kendra-s3-read-policy-for-${projectName}`, {
        statements: [kendraS3ReadPolicy],
    }),
);    
```

### Kendra 파일 크기 Quota

[Quota Console - File size](https://ap-northeast-1.console.aws.amazon.com/servicequotas/home/services/kendra/quotas/L-C108EA1B)와 같이 Kendra에 올릴수 있는 파일크기는 50MB로 제한됩니다. 이는 Quota 조정 요청을 위해 적절한 값으로 조정할 수 있습니다. 다만 이 경우에도 파일 한개에서 얻어낼수 있는 Text의 크기는 5MB로 제한됩니다. msg를 한국어 Speech로 변환한 후에 CloudFront URL을 이용하여 S3에 저장된 Speech를 URI로 공유할 수 있습니다.

### 데이터 소스 추가

S3를 데이터 소스르 추가할때 아래와 같이 수행하면 되나, languageCode가 미지원되어서 CLI로 대체합니다.

```typescript
const cfnDataSource = new kendra.CfnDataSource(this, `s3-data-source-${projectName}`, {
    description: 'S3 source',
    indexId: kendraIndex,
    name: 'data-source-for-upload-file',
    type: 'S3',
    // languageCode: 'ko',
    roleArn: roleKendra.roleArn,
    // schedule: 'schedule',

    dataSourceConfiguration: {
        s3Configuration: {
            bucketName: s3Bucket.bucketName,
            documentsMetadataConfiguration: {
                s3Prefix: 'metadata/',
            },
            inclusionPrefixes: ['documents/'],
        },
    },
});
```

CLI 명령어 예제입니다.

```text
aws kendra create-data-source
--index-id azfbd936-4929-45c5-83eb-bb9d458e8348
--name data-source-for-upload-file
--type S3
--role-arn arn:aws:iam::123456789012:role/role-lambda-chat-ws-for-korean-chatbot-us-west-2
--configuration '{"S3Configuration":{"BucketName":"storage-for-korean-chatbot-us-west-2", "DocumentsMetadataConfiguration": {"S3Prefix":"metadata/"},"InclusionPrefixes": ["documents/"]}}'
--language-code ko
--region us-west-2
```

## OpenSearch

### OpenSearch 준비

[Python client](https://opensearch.org/docs/latest/clients/python-low-level/)에 따라 OpenSearch를 활용합니다.

opensearch-py를 설치합니다.

```text
pip install opensearch-py
```

[Index naming restrictions](https://opensearch.org/docs/1.0/opensearch/rest-api/create-index/#index-naming-restrictions)에 따랏 index는 low case여야하고, 공백이나 ','을 가질수 없습니다.

### OpenSearch의 성능 향상 방법

Vector 검색(Sementaic) 뿐 아니라, Lexical 검색(Keyword)을 활용하여 관련된 문서를 찾을 확율을 높입니다. 상세한 내용은 [OpenSearch에서 Lexical 검색](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/opensearch-nori-plugin.md)에 있습니다.


### OpenSearch의 문서 업데이트

문서 생성시 업데이트까지 고려하여 index를 체크하여 지우는 방식을 사용하였으나 shard가 과도하게 증가하여, metadata에 ids를 저장후 지우는 방식으로 변경하였습니다. [lambda-document-manager](./lambda-document-manager/lambda_function.py)을 참조합니다. 동작은 파일 업데이트시 meta에서 이전 document들을 찾아서 지우고 새로운 문서를 삽입니다.

```python
def store_document_for_opensearch(docs, key):    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
    metadata_key = meta_prefix+objectName+'.metadata.json'
    delete_document_if_exist(metadata_key)
    
    try:        
        response = vectorstore.add_documents(docs, bulk_size = 2000)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
        #raise Exception ("Not able to request to LLM")

    print('uploaded into opensearch')
    
    return response

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
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")
```

### OpenSearch Embedding시 bulk_size

아래는 OpenSearch에서 Embedding을 할때 bulk_size 기본값인 500을 사용할때의 에러입니다. 문서를 embedding하기 위해 1840번 embedding을 해야하는데, bulk_size가 500이므로 에러가 발생하였습니다.

```text
RuntimeError: The embeddings count, 1840 is more than the [bulk_size], 500. Increase the value of [bulk_size].
```

bulk_size를 10000으로 변경하여 해결합니다.

```python
new_vectorstore = OpenSearchVectorSearch(
    index_name=index_name,  
    is_aoss = False,
    #engine="faiss",  # default: nmslib
    embedding_function = bedrock_embeddings,
    opensearch_url = opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd),
)
response = new_vectorstore.add_documents(docs, bulk_size = 10000)
```


### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-korean-chatbot/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


## 실행결과

#### Multi modal 및 RAG

"Conversation Type"으로 [General Conversation]을 선택하고, [dice.png](./contents/dice.png) 파일을 다운로드합니다.


<img src="./contents/dice.png" width="300">

이후에 채팅창 아래의 파일 버튼을 선택하여 업로드합니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/3980f1f4-1809-4250-b137-4511ae166f06)



[fsi_faq_ko.csv](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/fsi_faq_ko.csv)을 다운로드한 후에 파일 아이콘을 선택하여 업로드한후, 채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. 이때의 결과는 ＂아니오”입니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c7aeca05-0209-49c3-9df9-7e04026900f2)

채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/56ad9192-6b7c-49c7-9289-b6a3685cb7d4)

채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. "영문뱅킹에서는 간편조회서비스 이용불가"하므로 좀더 자세한 설명을 얻었습니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/3a896488-af0c-42b2-811b-d2c0debf5462)

채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?"라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/2e2b2ae1-7c50-4c14-968a-6c58332d99af)

#### Agent 사용하기

채팅창에서 뒤로가기 한 후에 "1-2 Agent"를 선택합니다. 아래와 같이 "여행 관련 도서 추천해줘."와 같이 입력하면 교보문고의 API를 이용하여 "여행"과 관련된 문서를 조회한 후 결과를 보여줍니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/b3997df2-b6ea-4419-9843-1dfe420fef2d)

"서울의 오늘 날씨 알려줘"라고 입력하면 아래와 같이 날씨 정보를 조회하여 보여줍니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/f805a915-0d8a-4124-8243-a08fb1463919)

LLM에 시간을 물어보면 마지막 Training 시간이나 전혀 관련없는 Hallucination 값을 줍니다. Agent를 사용할 경우에 아래와 같이 현재 시간을 조회하여 보여줍니다. "오늘 날짜 알려줘."와 "현재 시간은?"을 이용하여 동작을 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/69b370c5-882a-4031-8e54-b4840d023b90)


#### 잘못된 응답 유도해보기

"엔씨의 Lex 서비스는 무엇인지 설명해줘."와 같이 잘못된 단어를 조합하여 질문하였습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/78f2f2c0-cecf-43a2-98c7-843276755248)

"Amazon Varco 서비스를 Manufactoring에 적용하는 방법 알려줘."로 질문하고 응답을 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/8c484742-a294-4876-afe9-df0e0f2d96c5)

#### 한영 동시검색

"Amazon의 Athena 서비스에 대해 설명해주세요."로 검색할때 한영 동시 검색을 하면 영어 문서에서 답변에 필요한 관련문서를 추출할 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/4526c9aa-a0aa-4b23-8818-860f5376b898)

한영동시 검색을 하지 않았을때의 결과는 아래와 같습니다. 동일한 질문이지만, OpenSearch의 결과를 많이 참조하여 잘못된 답변을 할 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/b5548594-abc8-4447-8f95-d6d12d36c23e)


## Prompt Engineering 결과 예제

### Translation

"아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다.”로 입력하고 번역 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/818662e1-983f-44c2-bfcf-e2605ba7a1e6)

### Extracted Topic and sentiment

“식사 가성비 좋습니다. 위치가 좋고 스카이라운지 바베큐 / 야경 최곱니다. 아쉬웠던 점 · 지하주차장이 비좁습니다.. 호텔앞 교통이 너무 복잡해서 주변시설을 이용하기 어렵습니다. / 한강나가는 길 / 주변시설에 나가는 방법등.. 필요합니다.”를 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/8c38a58b-08df-4e9e-a162-1cd8f542fb46)

### Information extraction

“John Park. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-5555“로 입력후에 이메일이 추출되는지 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/f613e86e-b08d-45e8-ac0e-71334427f450)

### PII(personally identifiable information) 삭제하기

PII(Personal Identification Information)의 삭제의 예는 아래와 같습니다. "John Park, Ph.D. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-4567"와 같이 입력하여 name, phone number, address를 삭제한 텍스트를 얻습니다. 프롬프트는 [PII](https://docs.anthropic.com/claude/docs/constructing-a-prompt)를 참조합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a77d034c-32fc-4c84-8054-f4e1230292d6)

### 문장 오류 고치기

"To have a smoth conversation with a chatbot, it is better for usabilities to show responsesess in a stream-like, conversational maner rather than waiting until the complete answer."로 오류가 있는 문장을 입력합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/55774e11-58e3-4eb4-b91c-5b09572456bd)

"Chatbot과 원할한 데화를 위해서는 사용자의 질문엥 대한 답변을 완전히 얻을 때까지 기다리기 보다는 Stream 형태로 보여주는 것이 좋습니다."로 입력후에 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7b098a29-9bf5-43bf-a32f-82c94ccd04eb)

### 복잡한 질문 (step-by-step)

"I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?"를 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c1bf6749-1ce8-44ba-81f1-1fb52e04a2e8)


"내 고양이 두 마리가 있다. 그중 한 마리는 다리가 하나 없다. 다른 한 마리는 고양이가 정상적으로 가져야 할 다리 수를 가지고 있다. 전체적으로 보았을 때, 내 고양이들은 다리가 몇 개나 있을까?"로 질문을 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/992c8385-f897-4411-b6cf-b185465e8690)

### 날짜/시간 추출하기

메뉴에서 "Timestamp Extraction"을 선택하고, "지금은 2023년 12월 5일 18시 26분이야"라고 입력하면 prompt를 이용해 아래처럼 시간을 추출합니다.

![noname](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7dd7e659-498c-4898-801c-b72830bf254b)


실제 결과 메시지는 아래와 같습니다. 

```xml
<result>
<year>2023</year>
<month>12</month>
<day>05</day>
<hour>18</hour>
<minute>26</minute>
</result>
```

### 어린이와 대화 (Few shot example)

대화의 상대에 맞추어서 질문에 답변을하여야 합니다. 이를테면 [General Conversation]에서 "산타가 크리스마스에 선물을 가져다 줄까?"로 질문을 하면 아래와 같이 답변합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/4624727b-addc-4f5d-8f3d-94f358572326)

[9. Child Conversation (few shot)]으로 전환합니다. 동일한 질문을 합니다. 상대에 맞추어서 적절한 답변을 할 수 있었습니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/cbbece5c-5476-4f3b-89f7-c7fcf90ca796)


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2)로 접속하여 "rest-api-for-stream-chatbot", "ws-api-for-stream-chatbot"을 삭제합니다.

2) [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cdk destroy --all
```

## 결론

LLM을 사용한 Enterprise용 application을 개발하기 위해서는 기업이 가진 다양한 정보를 활용하여야 합니다. 이를 위해 Fine-tuning이나 RAG를 활용할 수 있습니다. Fine-tuning은 일반적으로 RAG보다 우수한 성능을 기대할 수 있으나, 다양한 application에서 활용하기 위해서 많은 비용과 시행착오가 있을 수 있습니다. RAG는 데이터의 빠른 업데이트 및 비용면에서 활용도가 높아서, Fine-tuning과 RAG를 병행하여 활용하는 방법을 생각해 볼 수 있습니다. 여기에서는 RAG의 성능을 향상시키리 위해 다양한 기술을 통합하고, 이를 활용할 수 있는 Korean Chatbot을 만들었습니다. 이를 통해 다양한 RAG 기술들을 테스트하고 사용하는 용도에 맞게 RAG 기술을 활용할 수 있습니다.

## Reference

[Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)

[Advanced RAG Techniques: An Overview](https://www.linkedin.com/pulse/advanced-rag-techniques-overview-yugank-aman-t4kkf/)

