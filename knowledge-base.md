# Amazon Knowledge Base

Amazon Bedrock의 [Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)을 이용하면 쉽고 편리하게 RAG를 구성할 수 있습니다.

## Knowledge Base의 구성

1) [Knowledge base console)(https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/knowledge-bases)에 접속하여 [Create knowledge base]을 선택합니다. 이후 아래와 같이 name을 지정후에 Data Souce는 기본값인 Amazon S3를 유지한 상태에서 [Next]를 선택합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/8dd3f6a4-5c82-4ceb-a42b-4e0d1e8db843">

2) data source의 이름을 지정하고 [Browse S3]를 선택해 적절한 S3를 선택합니다.
   
<img width="700" alt="image" src="https://github.com/user-attachments/assets/08d3f294-b8ff-4216-82af-ece338debf26">

3) 아래로 스크롤하여 [Chunking and parsing configurations]에서 "Custom"을 선택하고, [Chunking strategy]으로 "Hierarchical chunking"을 선택합니다. [Max parent token size]와 [Max child token size]의 기본값은 각각 1500과 300입니다. Hierarchical chunking을 사용하면 검색은 child chunk를 활용하고, context는 parent chunk을 사용하게 됩니다. 따라서, 검색의 정확도를 높이면서 관련된 문서(relevant document)에 대한 충분한 context를 제공할 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/91bea97e-6a04-4ffa-be07-01379c9c3366">

4) Embedding model로 "Tital Text Embeddings v2"를 선택합니다. Vector dimesions으로는 기본값인 1024를 선택합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/e86f0278-c599-4b52-9df9-f76c436dfc2e">

5) Vector database로는 기본값인 "Quick create a new vector store - Recommended"을 선택하고, [Next]를 선택합니다.
   
<img width="700" alt="image" src="https://github.com/user-attachments/assets/a6376701-a342-455c-a32d-65f4ef6e2d79">

6) 설정된 값을 확인후에 [Create knowledge base]을 선택하여 Knowledge Base를 생성합니다.

## Knowledge Base의 S3 동기화

아래와 같이 Knowledge Base ID를 확인할 수 있습니다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/8633f3e9-4f4a-4971-8a51-979ba15c97cd">

아래로 스크롤하여 [Data source]에서 생성한 Data souce를 지정한 후에 [Sync]를 선택합니다. 

<img width="800" alt="image" src="https://github.com/user-attachments/assets/9a238419-599e-41d0-9753-500905627b92">

## Knowledge Base의 동작 확인

Knowledge Base는 RAG를 테스트할 수 있는 환경을 제공합니다. 아래와 같이 질문을 입력하면 검색된 문서를 기반으로 답변을 확인할 수 있습니다.

<img width="350" alt="image" src="https://github.com/user-attachments/assets/906956a8-d9f1-4d3c-a147-16d8e007a8e9">