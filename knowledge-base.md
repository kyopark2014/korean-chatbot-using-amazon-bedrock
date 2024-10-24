# Knowledge Base로 Advanced RAG 구현

Amazon Bedrock의 [Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)을 이용하면 쉽고 편리하게 RAG를 구성할 수 있습니다.

## Knowledge Base의 구성

1) [S3 console](https://ap-northeast-2.console.aws.amazon.com/s3/home?region=ap-northeast-2#)에 접속해서 [Create bucket]을 선택합니다. Bucket name을 지정하고 [Create bucket]을 선택합니다. 기존에 사용하던 Bucket이 있다면 아래는 skip 합니다.

![noname](https://github.com/user-attachments/assets/414f7d32-9aea-4350-aa52-fcc64cde1260)

2) [Knowledge base console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/knowledge-bases)에 접속하여 [Create knowledge base]을 선택합니다. 이후 아래와 같이 name을 "aws-rag"로 지정힌 후에 Data Souce는 기본값인 Amazon S3를 유지한 상태에서 [Next]를 선택합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/8dd3f6a4-5c82-4ceb-a42b-4e0d1e8db843">

3) data source의 이름을 지정하고 [Browse S3]를 선택해 적절한 S3를 선택합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/9dfc106c-217d-497f-8d6d-9eeb6c6ec81b">

4) 아래와 같이 [Parsing strategy]에서 [Use foundation model for parsing]을 선택하고 "Claude 3 Sonnet v1"을 선택합니다. 이를 선택하면 PDF 파일에서 이미지와 텍스트의 정보를 추출하여 RAG에서 활용합니다. 추출에 사용되는 prompt는 [Instructions for the parser - optional]에서 확인할 수 있습니다. 여기서는 기본 prompt를 그대로 활용합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/a156fd9f-786b-4b22-9a7b-2a634b3bb88d">

참고로 기본 Prompt의 내용중 이미지와 표에 대한 내용은 아래와 같습니다.

```text
4. If the element is a visualization
    - Provide a detailed description in natural language
    - Do not transcribe text in the visualization after providing the description

5. If the element is a table
    - Create a markdown table, ensuring every row has the same number of columns
    - Maintain cell alignment as closely as possible
    - Do not split a table into multiple tables
    - If a merged cell spans multiple rows or columns, place the text in the top-left cell and output ' ' for other
    - Use | for column separators, |-|-| for header row separators
    - If a cell has multiple items, list them in separate rows
    - If the table contains sub-headers, separate the sub-headers from the headers in another row
```
   
5) [Chunking strategy]으로 "Hierarchical chunking"을 선택하고 [Next]를 선택합니다. 여기서, [Max parent token size]와 [Max child token size]의 기본값은 각각 1500과 300입니다. Hierarchical chunking을 사용하면 검색은 child chunk를 활용하고, context는 parent chunk을 사용하게 됩니다. 따라서, 검색의 정확도를 높이면서 관련된 문서(relevant document)에 대한 충분한 context를 제공할 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/28b97138-87cd-4cb8-9c4d-a0aa5a127262">


5) Embedding model로 "Titan Text Embeddings v2"를 선택합니다. Vector dimensions로는 기본값인 1024를 선택합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/e86f0278-c599-4b52-9df9-f76c436dfc2e">

6) Vector database로는 기본값인 "Quick create a new vector store - Recommended"을 선택하고, [Next]를 선택합니다.
   
<img width="700" alt="image" src="https://github.com/user-attachments/assets/a6376701-a342-455c-a32d-65f4ef6e2d79">

7) 설정된 값을 확인후에 [Create knowledge base]을 선택하여 Knowledge Base를 생성합니다.

   

## Knowledge Base의 S3 동기화

[Amazon S3 console](https://us-west-2.console.aws.amazon.com/s3/buckets?region=us-west-2)에 접속해서, Knowledge base에서 지정한 bucket에 사용할 문서들을 등록합니다. 

[Knowledge base console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/knowledge-bases)에 접속하여 생성한 "aws-rag"에 접속합니다. 아래와 같이 Knowledge Base ID를 확인할 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/cb4ad217-f587-44d3-a707-edc039668860">

아래로 스크롤하여 [Data source]에서 생성한 Data source를 지정한 후에 [Sync]를 선택합니다. 

<img width="800" alt="image" src="https://github.com/user-attachments/assets/9a238419-599e-41d0-9753-500905627b92">

## Knowledge Base에 Web Crawler 추가하기

[Data source]에서 [Add]를 선택한 후에 "Web Crawler"를 선택한 후에 [Next]를 선택합니다. 


<img width="700" alt="image" src="https://github.com/user-attachments/assets/0f097867-c4c1-4455-9e64-33cbfee3a68d">

[Configure data source]의 [Name and description]을 입력하고 [Source URLs]에 "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html"을 입력합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/9b26bb16-e61d-4957-b9d6-1fb6815a7b25">

아래의 메뉴에서 [Sync scope]는 기본값인 "Default"를 그대로 유지합니다. 이때, 동일한 호스트와 동일한 초기 URL 경로에 속하는 웹 페이지를 크롤링하게 됩니다. 

아래로 스크롤하여 [Content chunking and parsing]을 S3 data source와 동일하게 아래처럼 설정합니다. 이후 [Next]를 선택하여 Review 한 다음에 [Add data sources]로 선택합니다.  

<img width="700" alt="image" src="https://github.com/user-attachments/assets/dd189162-053e-4e2f-93ff-6e679f5e69ad">

"Web Crawler"가 데이터를 가져올 수 있도록 아래와 같이 Web Crawler"를 선택한 후에 [Sync]를 선택합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/31ad777c-d445-40ea-860c-222164976a55">


## Knowledge Base의 동작 확인


Knowledge Base는 RAG를 테스트할 수 있는 환경을 제공합니다. 아래와 같이 질문을 입력하면 검색된 문서를 기반으로 답변을 확인할 수 있습니다.

<img width="350" alt="image" src="https://github.com/user-attachments/assets/906956a8-d9f1-4d3c-a147-16d8e007a8e9">


아래와 같이 Knowledge Base에서 질문하여 이미지 정보를 활용하였는지 확인합니다. 

<img width="518" alt="image" src="https://github.com/user-attachments/assets/4db9fe1f-35b0-4678-ada9-9c5b9049ae14">

아래와 같이 참조된 데이터에는 그림이 포함되어 있는것을 알 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/a3353f5a-c73f-4f76-bf54-ee3f9f19f36b">

애플리케이션에 knowledge base를 호출하는 것을 시험하기 위하여 메뉴에서 "Knowledge Base (RAG)"을 선택합니다.

이후 아래와 같이 입력하면 애플리케이션에 knowledge base를 통해 얻은 결과를 확인할 수 있습니다.

![image](https://github.com/user-attachments/assets/dfc5bd14-8b3c-44bc-a360-7559c50af3ba)


### Advanced parsing의 제한 사항

Data source의 크기는 104857600 byte(100MB)까지 허용하고 있습니다. 관련된 Quota는 [Advanced parsing total data size](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html)입니다. 그밖에 파일의 최대 크기는 50MB이고, 최대 100개까지만 허용합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/3e552153-19d2-4a3a-8d0b-646395930b33">
