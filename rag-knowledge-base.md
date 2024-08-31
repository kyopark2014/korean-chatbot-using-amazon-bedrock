# Knowledge Base를 이용한 RAG의 구현

## Knowledge Base 구성하기

[knowledge-base.md](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/knowledge-base.md)을 참조하여 Knowledge Base를 생성합니다. 생성후 Knowledge Base ID를 복사합니다. 

## Knowledge Base로 Retriever 구성하기

애플리케이션에서 Knowledge Base를 통해 질문과 관련된 문서를 추출할 때에는 아래와 같이 Knowledge Base ID를 활용합니다. 또한 여기에서는 관련된 문서의 숫자(numberOfResults)는 4로 지정하였습니다. Knowledge Base를 활용하기 위하여 여기서는 LangChain의 [AmazonKnowledgeBasesRetriever](https://python.langchain.com/v0.2/docs/integrations/retrievers/bedrock/)을 이용하고 있습니다. Knowledge Base에서 관련된 문서뿐 아니라 참고문헌(reference)도 추출하여 활용할 수 있습니다.

Knowledge base의 생성후에 knowledge_base_id를 알수 있습니다. 여기서는 편의상 knowledge_base_name을 이용하여 knowledge_base_id를 검색해서 사용하고 있습니다.

```python
from langchain_aws import AmazonKnowledgeBasesRetriever

def get_answer_using_knowledge_base(chat, text, connectionId, requestId):    
    revised_question = text # use original question for test

    if not knowledge_base_id:        
        client = boto3.client('bedrock-agent')         
        response = client.list_knowledge_bases(
            maxResults=10
        )
                
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    break
    
    msg = reference = ""
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
        )
        
        relevant_docs = retriever.invoke(revised_question)
        
        relevant_context = ""
        for i, document in enumerate(relevant_docs):
            print(f"{i}: {document}")
            if document.page_content:
                content = document.page_content
            
            relevant_context = relevant_context + content + "\n\n"
        
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

[하이브리드 검색 구현하기 (feat. EnsembleRetriever, Knowledge Bases for Amazon Bedrock)](https://medium.com/@nuatmochoi/%ED%95%98%EC%9D%B4%EB%B8%8C%EB%A6%AC%EB%93%9C-%EA%B2%80%EC%83%89-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-feat-ensembleretriever-knowledge-bases-for-amazon-bedrock-d6ef1a0daaf1)와 같이 Boto3를 이용하여 Knowledge Base에서 Hybrid Search를 구현할 수 있습니다.


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

retriever만 이용시 [실제 서비스에서 Knowledge Bases for Amazon Bedrock 활용 (with API, LangChain)](https://medium.com/@nuatmochoi/%EC%8B%A4%EC%A0%9C-%EC%84%9C%EB%B9%84%EC%8A%A4%EC%97%90%EC%84%9C-knowledge-bases-for-amazon-bedrock-%ED%99%9C%EC%9A%A9-with-api-langchain-dc9b00ecc44d)을 활용할 수 있습니다.

```python
import boto3

bedrock_agent_runtime = boto3.client(
    service_name = "bedrock-agent-runtime"
)

def retrieve(query, kbId, numberOfResults=5):
    return bedrock_agent_runtime.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults
            }
        }
    )

def lambda_handler(event, context):
    response = retrieve("생애최초 특별공급은 어떻게 신청하나요?", "{KnowledgeBaseID}")
    results = response["retrievalResults"]
    return results
```

### LangChain의 AmazonKnowledgeBasesRetriever

[AmazonKnowledgeBasesRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html)는 2024년 8월 현재에 아직 hybrid를 제공하지 않으므로, Hybrid 검색시는 Boto3의 bedrock_agent_runtime.retrieve을 이용하여야 합니다.



