# Document를 PDF 파일로 변환하기

RAG에 등록하는 문서로 DOC이나 PDF에서 이미지를 추출할때 문서와 함께 이미지를 추출하는것이 어렵습니다. Multimodal LLM에서 DOC나 PPT의 내용을 같이 분석할 수 있도록 DOC과 PPT문서를 PDF로 변환하고자 합니다. 


## LibreOffice를 이용하여 문서를 변환

[LibreOffice](https://github.com/vladholubiev/serverless-libreoffice)를 이용하여 /tmp에서 문서를 변환합니다. [Lambda layer](https://github.com/shelfio/libreoffice-lambda-layer) 또는 [docker image](https://github.com/vladholubiev/serverless-libreoffice?tab=readme-ov-file)로 활용할 수 있습니다.


[Convert Doc or Docx to pdf using AWS Lambda](https://medium.com/analytics-vidhya/convert-word-to-pdf-using-aws-lambda-cb111be0d685)

[Converting Office Docs to PDF with AWS Lambda](https://madhavpalshikar.medium.com/converting-office-docs-to-pdf-with-aws-lambda-372c5ac918f1)
