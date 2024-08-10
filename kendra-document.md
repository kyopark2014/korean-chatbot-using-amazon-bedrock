# Kendra의 Document 설정

Kendra의 검색 성능을 향상시키기 위한 방법에 대해 설명합니다.

## Attribute

[Document Attribute](https://docs.aws.amazon.com/kendra/latest/dg/hiw-document-attributes.html)와 같이 주요한 document field는 아래와 같습니다.

- _authors: 저자 리스트
- _category: Document group의 category
- _data_source_id: data source의 id
- _document_body: document body
- _document_id: document의 unique id
- _document_title: document의 제목
- _excerpt_page_number: 페이지 번호
- _faq_id: FAQ의 id
- _file_type: document type
- _source_uri: document의 URI
- _language_code: 언어코드, 영어(en), 한국어(ko)

## [BatchPutDocument](https://docs.aws.amazon.com/kendra/latest/APIReference/API_BatchPutDocument.html) API에서 Attribute 추가하기

[batch_put_document](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/kendra/client/batch_put_document.html)에 따라 아래처럼 attribute를 지정할 수 있습니다.

```java
response = client.batch_put_document(
    IndexId='string',
    RoleArn='string',
    Documents=[
        {
            'Id': 'string',
            'Title': 'string',
            'Blob': b'bytes',
            'S3Path': {
                'Bucket': 'string',
                'Key': 'string'
            },
            'Attributes': [
                {
                    'Key': 'string',
                    'Value': {
                        'StringValue': 'string',
                        'StringListValue': [
                            'string',
                        ],
                        'LongValue': 123,
                        'DateValue': datetime(2015, 1, 1)
                    }
                },
            ],
            'AccessControlList': [
                {
                    'Name': 'string',
                    'Type': 'USER'|'GROUP',
                    'Access': 'ALLOW'|'DENY',
                    'DataSourceId': 'string'
                },
            ],
            'HierarchicalAccessControlList': [
                {
                    'PrincipalList': [
                        {
                            'Name': 'string',
                            'Type': 'USER'|'GROUP',
                            'Access': 'ALLOW'|'DENY',
                            'DataSourceId': 'string'
                        },
                    ]
                },
            ],
            'ContentType': 'PDF'|'HTML'|'MS_WORD'|'PLAIN_TEXT'|'PPT'|'RTF'|'XML'|'XSLT'|'MS_EXCEL'|'CSV'|'JSON'|'MD',
            'AccessControlConfigurationId': 'string'
        },
    ]
)
```

여기서 ContentType으로 아래와 같은 파일 확장자를 제공하므로 TXT는 PLAIN_TEXT로, PPTX는 "PPT"로 등록하여야 합니다. 

```text
PLAIN_TEXT, XSLT, MS_WORD, RTF, CSV, JSON, HTML, PDF, PPT, MD, XML, MS_EXCEL
```
