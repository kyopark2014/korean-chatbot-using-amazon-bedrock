# Knowledge Base를 이용한 RAG의 구현

## Knowledge Base 구성하기

## Knowledge Base로 Retriever 구성하기

```python
def get_answer_using_knowledge_base(chat, text, connectionId, requestId):    
    revised_question = text # use original question for test
    
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="CFVYNN0NQN", 
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )
    
    relevant_docs = retriever.invoke(revised_question)
    print(relevant_docs)
    
    relevant_context = ""
    for i, document in enumerate(relevant_docs):
        print(f"{i}: {document}")
        if document.page_content:
            content = document.page_content
        print('score:', document.metadata["score"])
        
        uri = document.metadata["location"]["s3Location"]["uri"] if document.metadata["location"]["s3Location"]["uri"] is not None else ""
        print('uri:', uri)
        
        relevant_context = relevant_context + content + "\n\n"
    
    print('relevant_context: ', relevant_context)

    msg = query_using_RAG_context(connectionId, requestId, chat, relevant_context, revised_question)

    reference = get_reference_of_knoweledge_base(relevant_docs, path, doc_prefix)  
    
    return msg, reference

def query_using_RAG_context(connectionId, requestId, chat, context, revised_question):    
    if isKorean(revised_question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
                   
    chain = prompt | chat
    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "context": context,
                "input": revised_question,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        print('msg: ', msg)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    return msg

def get_reference_of_knoweledge_base(docs, path, doc_prefix):
    reference = "\n\nFrom\n"
    
    for i, document in enumerate(docs):
        if document.page_content:
            excerpt = document.page_content
        
        score = document.metadata["score"]            
        uri = document.metadata["location"]["s3Location"]["uri"] if document.metadata["location"]["s3Location"]["uri"] is not None else ""
        
        pos = uri.find(f"/{doc_prefix}")
        name = uri[pos+len(doc_prefix)+1:]
        encoded_name = parse.quote(name)
        
        uri = f"{path}{doc_prefix}{encoded_name}"        
        reference = reference + f"{i+1}. <a href={uri} target=_blank>{name}</a>, <a href=\"#\" onClick=\"alert(`{excerpt}`)\">관련문서</a>\n"
                    
    return reference
```


## Knowledge Base에서 Hybrid Search



```python
import boto3

bedrock_agent_runtime = boto3.client(
    service_name = "bedrock-agent-runtime"
)

def retrieve_and_generate(query, kbId):
    modelArn = 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-v2:1'
    
    return bedrock_agent_runtime.retrieve_and_generate(
        input={
            'text': query,
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kbId,
                'modelArn': modelArn,
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'overrideSearchType': 'HYBRID' # Default는 SEMANTIC
                    }
                }
            }
            }
        }
    )

def lambda_handler(event, context):
    response = retrieve_and_generate("생애최초 특별공급은 어떻게 신청하나요?", "<KnowledgeBaseID>")
    output = response["output"]
    citations = response["citations"]
    
    return output
```
