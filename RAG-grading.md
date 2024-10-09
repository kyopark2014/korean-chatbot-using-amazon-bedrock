# LLM으로 RAG Grading 활용하기 

RAG는 지식저장소에서 관련된 문서(Relevant doc)을 가져옵니다. 이때 RAG의 성능을 향상시키기 위해 관련된 문서가 실제로 관련되었는지 확인하는 절차가 필요합니다. 이러한 방법에서 Rerank나 [priority search](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/tree/main?tab=readme-ov-file#priority-search-%EA%B4%80%EB%A0%A8%EB%8F%84-%EA%B8%B0%EC%A4%80-%EB%AC%B8%EC%84%9C-%EC%84%A0%ED%83%9D)를 활용할 수 있습니다. 여기에서는 LLM으로 관련도를 확인하는 방법을 설명합니다. 상세한 코드는 [lambda_function.py](https://github.com/kyopark2014/rag-with-reflection/blob/main/lambda-chat-ws/lambda_function.py)을 참조합니다.

여기서는 [structued output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 활용하여 grading을 수행합니다. Grading은 "yes", "no"로 판정하고 있으나 좀더 분류를 하고자 할때에는 "5점 이하의 점수로 표현해줘"와 같이 prompt를 변경하여 활용할 수 있습니다. 

```python
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader
```

관련된 문서은 top_k만큼이 얻어지므로 아래와 같이 병렬처리를 통해 지연시간은 단축할 수 있습니다.

```python
from multiprocessing import Process, Pipe

def parallel_grader(query, relevant_docs):
    print("###### parallel_grader ######")
    
    global selected_chat    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(relevant_docs):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, query, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == len(multi_region_models):
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        doc = parent_conn.recv()

        if doc is not None:
            filtered_docs.append(doc)

    for process in processes:
        process.join()    
    
    global reference_docs 
    reference_docs += filtered_docs    
    
    # duplication checker
    reference_docs = check_duplication(reference_docs)
    
    return {
        "filtered_docs": filtered_docs
    }    

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
```

여기서 chat 모델은 아래와 같이 multi region을 활용합니다.

```python
def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
                          
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

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    return chat
```






