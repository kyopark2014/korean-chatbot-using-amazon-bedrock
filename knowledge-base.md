# Amazon Knowledge Base

Amazon Bedrock의 [Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)을 이용하면 쉽고 편리하게 RAG를 구성할 수 있습니다.

## Knowledge Base의 구성

1) [Knowledge base console)(https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/knowledge-bases)에 접속하여 [Create knowledge base]을 선택합니다. 이후 아래와 같이 name을 지정후에 Data Souce는 기본값인 Amazon S3를 유지한 상태에서 [Next]를 선택합니다.

![noname](https://github.com/user-attachments/assets/8dd3f6a4-5c82-4ceb-a42b-4e0d1e8db843)

2) data source의 이름을 지정합니다.

![noname](https://github.com/user-attachments/assets/8cc00f9b-a8fa-4c4b-a8f4-18c2fb5831c2)

3) 아래로 스크롤하여 [Chunking and parsing configurations]에서 "Custom"을 선택하고, [Chunking strategy]으로 "Hierarchical chunking"을 선택합니다. 

![noname](https://github.com/user-attachments/assets/91bea97e-6a04-4ffa-be07-01379c9c3366)


4) 
