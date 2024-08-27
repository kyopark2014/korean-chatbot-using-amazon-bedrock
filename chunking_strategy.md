# Chunking

LLM의 context window를 고려하여 문서는 적절한 크기의 chunk로 잘라서 사용합니다.

## Chunking Method

- Fixed size chunking
  - 가장 일반적이고 직접적인 접근 방식으로 token 수를 이용하여 chunking을 수행합니다.
  - Chunk 간의 의미 맥락(semantic context)을 유지 하기 위하여 Chunk 간에 겹치도록 합니다.
  - 다른 형태의 chunk method와 비교할때, 고정 크기 청킹은 계산 비용이 저렴하고 NLP 라이브러리를 사용할 필요가 없기 때문에 사용하기 간단합니다.
    
- Recursive Chunking
  - 재귀적 chunking은 일련의 구분 기호를 사용하여 입력 텍스트를 계층적이고 반복적인 방식으로 더 작은 chunk로 나눕니다.
  - 초기 텍스트 분할 시도에서 원하는 크기나 구조의 chunk가 생성되지 않으면, 원하는 chunk 크기나 구조가 달성될 때까지 다른 구분 기호나 기준으로 재귀적으로 반복 수행합니다.
  - Chunk의 크기가 정확히 동일하지는 않지만 비슷한 크기를 "지향(aspire)"한다는 것을 의미합니다. 고정 크기 청크와 중첩의 장점을 활용합니다. 

- Document Specific Chunking
  - 문서의 구조를 고려하여 고정된 문자 수나 재귀 프로세스를 사용하는 대신 문단이나 하위 섹션과 같은 문서의 논리적 섹션에 맞춰 청크를 생성합니다.
  - 작성자가 콘텐츠를 구성한 방식을 유지하여 텍스트의 일관성을 유지합니다. 이를 통해 검색된 정보의 관련성과 유용성이 높아지며, 특히 명확하게 정의된 섹션이 있는 구조화된 문서에 유용합니다.
  - Markdown, Html 등의 형식을 처리할 수 있습니다. 

- Semantic Chunking
  - 텍스트 내의 관계를 고려하여, 텍스트를 의미 있고 의미적으로 완전한 덩어리로 나눕니다.
  - 검색 중 정보의 무결성을 보장하여 보다 정확하고 상황에 맞는 결과를 얻을 수 있습니다.
  - 다른 청킹 전략에 비해 속도가 느립니다.
    
- Agentic Chunking
  - 다중 LLM 호출을 처리하는 데 걸리는 시간과 그 호출 비용 때문에 아직 대규모 활용에는 적합하지 않습니다.
  - 아직 공개 라이브러리에서 구현된 것은 없습니다. 
  - Human이 문서를 처리하는 절차와 유사한 방식으로 처리합니다.
    1) 문서의 맨 위에서 시작하여 첫 번째 부분을 하나의 Chunk로 취급합니다.
    2) 문서를 따라 내려가면서 새로운 문장이나 정보가 첫 번째 chunk에 속하는지 아니면 새로운 chunk를 시작해야 하는지 결정합니다.
    3) 문서의 끝에 도달할 때까지 이 과정을 계속합니다. 

- Hierarchical chunking (parent/child chunking)
  - 문서를 크기에 따라 parent chunk와 child chunk로 나누어서 child chunk를 찾은후에 LLM의 context에는 parent chunk를 사용하면, 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다.
  - 상세한 내용은 [Parent Document Retrieval](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/parent-document-retrieval.md)을 참조합니다.


## Semantic Chunking

LangChain의 [SemanticChunker](https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)를 이용할 수 있습니다. 

breakpoint_threshold_type으로 'percentile', 'standard_deviation', 'interquartile', 'gradient'를 이용할 수 있습니다.

- percentile: 문장 간의 차이를 계산한 다음 백분위 수보다 큰 차이는 분할됩니다.
- standard_deviation: 표준 편차보다 크면 분할 합니다.
- interquartile: 사분위(interquartile) 거리로 분할합니다.
  

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
  embed_model,
  breakpoint_threshold_type="percentile")

semantic_chunks = semantic_chunker.create_documents(
  [d.page_content for d in documents])
```

## Agentic Chunking


## Reference

[Semantic Chunking for RAG](https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5)
