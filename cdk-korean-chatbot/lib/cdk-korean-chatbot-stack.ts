import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as path from "path";
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudFront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as apiGateway from 'aws-cdk-lib/aws-apigateway';
import * as s3Deploy from "aws-cdk-lib/aws-s3-deployment";
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as apigatewayv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as opensearch from 'aws-cdk-lib/aws-opensearchservice';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kendra from 'aws-cdk-lib/aws-kendra';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { SqsEventSource } from 'aws-cdk-lib/aws-lambda-event-sources';

const region = process.env.CDK_DEFAULT_REGION;    
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const debug = false;
const stage = 'dev';
const s3_prefix = 'docs';
const projectName = `korean-chatbot-with-rag`; 
const bucketName = `storage-for-${projectName}-${accountId}-${region}`; 
let kendra_region = process.env.CDK_DEFAULT_REGION;   //  "us-west-2"
const rag_method = 'RetrievalPrompt'; // RetrievalPrompt, RetrievalQA, ConversationalRetrievalChain

const opensearch_account = "admin";
const opensearch_passwd = "Wifi1234!";
const enableReference = 'true';
let opensearch_url = "";
const debugMessageMode = 'false'; // if true, debug messages will be delivered to the client.
const useParallelRAG = 'true';
const numberOfRelevantDocs = '6';
const kendraMethod = "custom_retriever"; // custom_retriever or kendra_retriever
const allowDualSearch = 'false';
const capabilities = JSON.stringify(["kendra", "opensearch"]); 
const supportedFormat = JSON.stringify(["pdf", "txt", "csv", "pptx", "ppt", "docx", "doc", "xlsx", "py", "js", "md", 'png', 'jpeg', 'jpg']);  
const separated_chat_history = 'true';

const max_object_size = 102400000; // 100 MB max size of an object, 50MB(default)
const enableHybridSearch = 'true';
const enableParallelSummary = 'true';
const enalbeParentDocumentRetrival = 'true';
const speech_generation = 'false';
// const flow_id = 'arn:aws:bedrock:us-west-2:677146750822:flow/TQE3MT9IQO'
// const flow_alias = 'BasicPromptFlow'
const flow_id = 'arn:aws:bedrock:us-west-2:677146750822:flow/72JCN8U0J4'
const flow_alias = 'aws_bot'

const claude3_5_sonnet = [
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "claude3.5",
    "max_tokens": 4096,
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "claude3.5",
    "max_tokens": 4096,
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
  },
  {
    "bedrock_region": "eu-central-1", // Frankfurt
    "model_type": "claude3.5",
    "max_tokens": 4096,
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
  },
  {
    "bedrock_region": "ap-northeast-1", // Tokyo
    "model_type": "claude3.5",
    "max_tokens": 4096,
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
  }
];

const claude3_sonnet = [
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  {
    "bedrock_region": "ca-central-1", // Canada
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  {
    "bedrock_region": "eu-west-2", // London
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  {
    "bedrock_region": "sa-east-1", // Sao Paulo
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
  }
];

const claude3_haiku = [
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0"
  },
  {
    "bedrock_region": "ca-central-1", // Canada
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0"
  },
  {
    "bedrock_region": "eu-west-2", // London
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0"
  },
  {
    "bedrock_region": "sa-east-1", // Sao Paulo
    "model_type": "claude3",
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0"
  }
];

const titan_embedding_v1 = [  // dimension = 1536
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v1"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v1"
  }
];

const titan_embedding_v2 = [  // dimension = 1024
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "ca-central-1", // Canada
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "eu-west-2", // London
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "sa-east-1", // Sao Paulo
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  }
];

const LLM_for_chat = claude3_sonnet;
const LLM_for_multimodal = claude3_sonnet;
const LLM_embedding = titan_embedding_v2;

export class CdkKoreanChatbotStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // s3 
    const s3Bucket = new s3.Bucket(this, `storage-${projectName}`,{
      bucketName: bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      publicReadAccess: false,
      versioned: false,
      cors: [
        {
          allowedHeaders: ['*'],
          allowedMethods: [
            s3.HttpMethods.POST,
            s3.HttpMethods.PUT,
          ],
          allowedOrigins: ['*'],
        },
      ],
    });
    if(debug) {
      new cdk.CfnOutput(this, 'bucketName', {
        value: s3Bucket.bucketName,
        description: 'The nmae of bucket',
      });
      new cdk.CfnOutput(this, 's3Arn', {
        value: s3Bucket.bucketArn,
        description: 'The arn of s3',
      });
      new cdk.CfnOutput(this, 's3Path', {
        value: 's3://'+s3Bucket.bucketName,
        description: 'The path of s3',
      });
    }

    // copy web application files into s3 bucket
    //new s3Deploy.BucketDeployment(this, `upload-HTML-for-${projectName}`, {
    //  sources: [s3Deploy.Source.asset("../html/")],
    //  destinationBucket: s3Bucket,
    //});    
    new s3Deploy.BucketDeployment(this, `upload-contents-for-${projectName}`, {
      sources: [
        s3Deploy.Source.asset("../contents/faq/")
      ],
      destinationBucket: s3Bucket,
      destinationKeyPrefix: 'faq/' 
    });   
    
    new cdk.CfnOutput(this, 'HtmlUpdateCommend', {
      value: 'aws s3 cp ../html/ ' + 's3://' + s3Bucket.bucketName + '/ --recursive',
      description: 'copy commend for web pages',
    });

    // cloudfront
    const distribution = new cloudFront.Distribution(this, `cloudfront-for-${projectName}`, {
      defaultBehavior: {
        origin: new origins.S3Origin(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200,  
    });
    new cdk.CfnOutput(this, `distributionDomainName-for-${projectName}`, {
      value: distribution.domainName,
      description: 'The domain name of the Distribution',
    });

    // DynamoDB for call log
    const callLogTableName = `db-call-log-for-${projectName}`;
    const callLogDataTable = new dynamodb.Table(this, `db-call-log-for-${projectName}`, {
      tableName: callLogTableName,
      partitionKey: { name: 'user_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'request_time', type: dynamodb.AttributeType.STRING }, 
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const callLogIndexName = `index-type-for-${projectName}`;
    callLogDataTable.addGlobalSecondaryIndex({ // GSI
      indexName: callLogIndexName,
      partitionKey: { name: 'request_id', type: dynamodb.AttributeType.STRING },
    });
    
    // Lambda - chat (websocket)
    const roleLambdaWebsocket = new iam.Role(this, `role-lambda-chat-ws-for-${projectName}`, {
      roleName: `role-lambda-chat-ws-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
        new iam.ServicePrincipal("kendra.amazonaws.com")
      )
    });
    roleLambdaWebsocket.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
    });
    const BedrockPolicy = new iam.PolicyStatement({  // policy statement for sagemaker
      resources: ['*'],
      actions: ['bedrock:*'],
    });
    roleLambdaWebsocket.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `bedrock-policy-lambda-chat-ws-for-${projectName}`, {
        statements: [BedrockPolicy],
      }),
    );        
    const apiInvokePolicy = new iam.PolicyStatement({ 
      // resources: ['arn:aws:execute-api:*:*:*'],
      resources: ['*'],
      actions: [
        'execute-api:Invoke',
        'execute-api:ManageConnections'
      ],
    });        
    roleLambdaWebsocket.attachInlinePolicy( 
      new iam.Policy(this, `api-invoke-policy-for-${projectName}`, {
        statements: [apiInvokePolicy],
      }),
    );  

    // Kendra  
    let kendraIndex = "";
    const roleKendra = new iam.Role(this, `role-kendra-for-${projectName}`, {
      roleName: `role-kendra-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("kendra.amazonaws.com")
      )
    });
    const cfnIndex = new kendra.CfnIndex(this, 'MyCfnIndex', {
      edition: 'DEVELOPER_EDITION',  // ENTERPRISE_EDITION, 
      name: `reg-kendra-${projectName}`,
      roleArn: roleKendra.roleArn,
    }); 
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
      actions: ["s3:Get*","s3:List*","s3:Describe*"],
    });
    roleKendra.attachInlinePolicy( // add kendra policy
      new iam.Policy(this, `kendra-s3-read-policy-for-${projectName}`, {
        statements: [kendraS3ReadPolicy],
      }),
    );    
    new cdk.CfnOutput(this, `index-of-kendra-for-${projectName}`, {
      value: cfnIndex.attrId,
      description: 'The index of kendra',
    }); 

    const accountId = process.env.CDK_DEFAULT_ACCOUNT;
    const kendraResourceArn = `arn:aws:kendra:${kendra_region}:${accountId}:index/${cfnIndex.attrId}`
    if(debug) {
      new cdk.CfnOutput(this, `resource-arn-of-kendra-for-${projectName}`, {
        value: kendraResourceArn,
        description: 'The arn of resource',
      }); 
    }           
      
    const kendraPolicy = new iam.PolicyStatement({  
      resources: [kendraResourceArn],      
      actions: ['kendra:*'],
    });      
    roleKendra.attachInlinePolicy( // add kendra policy
      new iam.Policy(this, `kendra-inline-policy-for-${projectName}`, {
        statements: [kendraPolicy],
      }),
    );      
    kendraIndex = cfnIndex.attrId;

    roleLambdaWebsocket.attachInlinePolicy( 
      new iam.Policy(this, `lambda-inline-policy-for-kendra-in-${projectName}`, {
        statements: [kendraPolicy],
      }),
    ); 

    const passRoleResourceArn = roleLambdaWebsocket.roleArn;
    const passRolePolicy = new iam.PolicyStatement({  
      resources: [passRoleResourceArn],      
      actions: ['iam:PassRole'],
    });
      
    roleLambdaWebsocket.attachInlinePolicy( // add pass role policy
      new iam.Policy(this, `pass-role-of-kendra-for-${projectName}`, {
      statements: [passRolePolicy],
      }), 
    );  

    // Poly Role
    const PollyPolicy = new iam.PolicyStatement({  
      actions: ['polly:*'],
      resources: ['*'],
    });
    roleLambdaWebsocket.attachInlinePolicy(
      new iam.Policy(this, 'polly-policy', {
        statements: [PollyPolicy],
      }),
    );

      // data source
    /*  const cfnDataSource = new kendra.CfnDataSource(this, `s3-data-source-${projectName}`, {
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
              s3Prefix: 'metadata',
            },
            inclusionPrefixes: ['documents'],
          },
        },        
      });  */
    new cdk.CfnOutput(this, `create-S3-data-source-for-${projectName}`, {
      value: 'aws kendra create-data-source --index-id '+kendraIndex+' --name data-source-for-upload-file --type S3 --role-arn '+roleLambdaWebsocket.roleArn+' --configuration \'{\"S3Configuration\":{\"BucketName\":\"'+s3Bucket.bucketName+'\", \"DocumentsMetadataConfiguration\": {\"S3Prefix\":\"metadata/\"},\"InclusionPrefixes\": [\"'+s3_prefix+'\"]}}\' --language-code ko --region '+kendra_region,
      description: 'The commend to create data source using S3',
    });

    // opensearch
    // Permission for OpenSearch
    const domainName = projectName
    const resourceArn = `arn:aws:es:${region}:${accountId}:domain/${domainName}/*`
    if(debug) {
      new cdk.CfnOutput(this, `resource-arn-for-${projectName}`, {
        value: resourceArn,
        description: 'The arn of resource',
      }); 
    }

    const OpenSearchAccessPolicy = new iam.PolicyStatement({        
      resources: [resourceArn],      
      actions: ['es:*'],
      effect: iam.Effect.ALLOW,
      principals: [new iam.AnyPrincipal()],      
    });  

    const domain = new opensearch.Domain(this, 'Domain', {
      version: opensearch.EngineVersion.OPENSEARCH_2_3,
      
      domainName: domainName,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      enforceHttps: true,
      fineGrainedAccessControl: {
        masterUserName: opensearch_account,
        // masterUserPassword: cdk.SecretValue.secretsManager('opensearch-private-key'),
        masterUserPassword:cdk.SecretValue.unsafePlainText(opensearch_passwd)
      },
      capacity: {
        masterNodes: 3,
        masterNodeInstanceType: 'r6g.large.search',
        // multiAzWithStandbyEnabled: false,
        dataNodes: 3,
        dataNodeInstanceType: 'r6g.large.search',        
        // warmNodes: 2,
        // warmInstanceType: 'ultrawarm1.medium.search',
      },
      accessPolicies: [OpenSearchAccessPolicy],      
      ebs: {
        volumeSize: 100,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
      },
      nodeToNodeEncryption: true,
      encryptionAtRest: {
        enabled: true,
      },
      zoneAwareness: {
        enabled: true,
        availabilityZoneCount: 3,        
      }
    });
    new cdk.CfnOutput(this, `Domain-of-OpenSearch-for-${projectName}`, {
      value: domain.domainArn,
      description: 'The arm of OpenSearch Domain',
    });
    new cdk.CfnOutput(this, `Endpoint-of-OpenSearch-for-${projectName}`, {
      value: 'https://'+domain.domainEndpoint,
      description: 'The endpoint of OpenSearch Domain',
    });
    opensearch_url = 'https://'+domain.domainEndpoint;

    // api role
    const role = new iam.Role(this, `api-role-for-${projectName}`, {
      roleName: `api-role-for-${projectName}-${region}`,
      assumedBy: new iam.ServicePrincipal("apigateway.amazonaws.com")
    });
    role.addToPolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: [
        'lambda:InvokeFunction',
        'cloudwatch:*'
      ]
    }));
    role.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/AWSLambdaExecute',
    }); 

    // API Gateway
    const api = new apiGateway.RestApi(this, `api-chatbot-for-${projectName}`, {
      description: 'API Gateway for chatbot',
      endpointTypes: [apiGateway.EndpointType.REGIONAL],
      binaryMediaTypes: ['application/pdf', 'text/plain', 'text/csv', 'application/vnd.ms-powerpoint', 'application/vnd.ms-excel', 'application/msword'], 
      deployOptions: {
        stageName: stage,

        // logging for debug
        // loggingLevel: apiGateway.MethodLoggingLevel.INFO, 
        // dataTraceEnabled: true,
      },
    });  
   
    new cdk.CfnOutput(this, `WebUrl-for-${projectName}`, {
      value: 'https://'+distribution.domainName+'/index.html',      
      description: 'The web url of request for chat',
    });        

    // Lambda - Upload
    const lambdaUpload = new lambda.Function(this, `lambda-upload-for-${projectName}`, {
      runtime: lambda.Runtime.NODEJS_16_X, 
      functionName: `lambda-upload-for-${projectName}`,
      code: lambda.Code.fromAsset("../lambda-upload"), 
      handler: "index.handler", 
      timeout: cdk.Duration.seconds(10),
      environment: {
        bucketName: s3Bucket.bucketName,
        s3_prefix:  s3_prefix
      }      
    });
    s3Bucket.grantReadWrite(lambdaUpload);
    
    // POST method - upload
    const resourceName = "upload";
    const upload = api.root.addResource(resourceName);
    upload.addMethod('POST', new apiGateway.LambdaIntegration(lambdaUpload, {
      passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
      credentialsRole: role,
      integrationResponses: [{
        statusCode: '200',
      }], 
      proxy:false, 
    }), {
      methodResponses: [  
        {
          statusCode: '200',
          responseModels: {
            'application/json': apiGateway.Model.EMPTY_MODEL,
          }, 
        }
      ]
    }); 
    if(debug) {
      new cdk.CfnOutput(this, `ApiGatewayUrl-for-${projectName}`, {
        value: api.url+'upload',
        description: 'The url of API Gateway',
      }); 
    }

    // cloudfront setting  
    distribution.addBehavior("/upload", new origins.RestApiOrigin(api), {
      cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
      allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
      viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    });    

    // Lambda - queryResult
    const lambdaQueryResult = new lambda.Function(this, `lambda-query-for-${projectName}`, {
      runtime: lambda.Runtime.NODEJS_16_X, 
      functionName: `lambda-query-for-${projectName}`,
      code: lambda.Code.fromAsset("../lambda-query"), 
      handler: "index.handler", 
      timeout: cdk.Duration.seconds(60),
      environment: {
        tableName: callLogTableName,
        indexName: callLogIndexName
      }      
    });
    callLogDataTable.grantReadWriteData(lambdaQueryResult); // permission for dynamo
    
    // POST method - query
    const query = api.root.addResource("query");
    query.addMethod('POST', new apiGateway.LambdaIntegration(lambdaQueryResult, {
      passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
      credentialsRole: role,
      integrationResponses: [{
        statusCode: '200',
      }], 
      proxy:false, 
    }), {
      methodResponses: [  
        {
          statusCode: '200',
          responseModels: {
            'application/json': apiGateway.Model.EMPTY_MODEL,
          }, 
        }
      ]
    }); 

    // cloudfront setting for api gateway    
    distribution.addBehavior("/query", new origins.RestApiOrigin(api), {
      cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
      allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
      viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    });

    // Lambda - getHistory
    const lambdaGetHistory = new lambda.Function(this, `lambda-gethistory-for-${projectName}`, {
      runtime: lambda.Runtime.NODEJS_16_X, 
      functionName: `lambda-gethistory-for-${projectName}`,
      code: lambda.Code.fromAsset("../lambda-gethistory"), 
      handler: "index.handler", 
      timeout: cdk.Duration.seconds(60),
      environment: {
        tableName: callLogTableName
      }      
    });
    callLogDataTable.grantReadWriteData(lambdaGetHistory); // permission for dynamo
    
    // POST method - history
    const history = api.root.addResource("history");
    history.addMethod('POST', new apiGateway.LambdaIntegration(lambdaGetHistory, {
      passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
      credentialsRole: role,
      integrationResponses: [{
        statusCode: '200',
      }], 
      proxy:false, 
    }), {
      methodResponses: [  
        {
          statusCode: '200',
          responseModels: {
            'application/json': apiGateway.Model.EMPTY_MODEL,
          }, 
        }
      ]
    }); 

    // cloudfront setting for api gateway    
    distribution.addBehavior("/history", new origins.RestApiOrigin(api), {
      cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
      allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
      viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    });

    // Lambda - deleteItems
    const lambdaDeleteItems = new lambda.Function(this, `lambda-deleteItems-for-${projectName}`, {
      runtime: lambda.Runtime.NODEJS_16_X, 
      functionName: `lambda-deleteItems-for-${projectName}`,
      code: lambda.Code.fromAsset("../lambda-delete-items"), 
      handler: "index.handler", 
      timeout: cdk.Duration.seconds(60),
      environment: {
        tableName: callLogTableName
      }      
    });
    callLogDataTable.grantReadWriteData(lambdaDeleteItems); // permission for dynamo
    
    // POST method - delete items
    const deleteItem = api.root.addResource("delete");
    deleteItem.addMethod('POST', new apiGateway.LambdaIntegration(lambdaDeleteItems, {
      passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
      credentialsRole: role,
      integrationResponses: [{
        statusCode: '200',
      }], 
      proxy:false, 
    }), {
      methodResponses: [  
        {
          statusCode: '200',
          responseModels: {
            'application/json': apiGateway.Model.EMPTY_MODEL,
          }, 
        }
      ]
    }); 

    // cloudfront setting for api gateway    
    distribution.addBehavior("/delete", new origins.RestApiOrigin(api), {
      cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
      allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
      viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    });

    // stream api gateway
    // API Gateway
    const websocketapi = new apigatewayv2.CfnApi(this, `ws-api-for-${projectName}`, {
      description: 'API Gateway for chatbot using websocket',
      apiKeySelectionExpression: "$request.header.x-api-key",
      name: 'api-'+projectName,
      protocolType: "WEBSOCKET", // WEBSOCKET or HTTP
      routeSelectionExpression: "$request.body.action",     
    });  
    websocketapi.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); // DESTROY, RETAIN

    const wss_url = `wss://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
    new cdk.CfnOutput(this, 'web-socket-url', {
      value: wss_url,      
      description: 'The URL of Web Socket',
    });

    const connection_url = `https://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
    if(debug) {
      new cdk.CfnOutput(this, 'api-identifier', {
        value: websocketapi.attrApiId,
        description: 'The API identifier.',
      });

      new cdk.CfnOutput(this, 'connection-url', {
        value: connection_url,        
        description: 'The URL of connection',
      });
    }

    const googleApiSecret = new secretsmanager.Secret(this, `google-api-secret-for-${projectName}`, {
      description: 'secret for google api key',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `googl_api_key-${projectName}`,
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ 
          google_cse_id: 'cse_id'
        }),
        generateStringKey: 'google_api_key',
        excludeCharacters: '/@"',
      },

    });
    googleApiSecret.grantRead(roleLambdaWebsocket) 
    
    const weatherApiSecret = new secretsmanager.Secret(this, `weather-api-secret-for-${projectName}`, {
      description: 'secret for weather api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `openweathermap-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        weather_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    weatherApiSecret.grantRead(roleLambdaWebsocket) 

    const langsmithApiSecret = new secretsmanager.Secret(this, `weather-langsmith-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `langsmithapikey-${projectName}`,
      secretObjectValue: {
        langchain_project: cdk.SecretValue.unsafePlainText(projectName),
        langsmith_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    langsmithApiSecret.grantRead(roleLambdaWebsocket) 

    const tavilyApiSecret = new secretsmanager.Secret(this, `weather-tavily-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `tavilyapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    tavilyApiSecret.grantRead(roleLambdaWebsocket) 

    // lambda-chat using websocket    
    const lambdaChatWebsocket = new lambda.DockerImageFunction(this, `lambda-chat-ws-for-${projectName}`, {
      description: 'lambda for chat using websocket',
      functionName: `lambda-chat-ws-for-${projectName}`,
      code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-chat-ws')),
      timeout: cdk.Duration.seconds(600),
      memorySize: 2048,
      role: roleLambdaWebsocket,
      environment: {
        // bedrock_region: bedrock_region,
        kendra_region: String(kendra_region),
        // model_id: model_id,
        s3_bucket: s3Bucket.bucketName,
        s3_prefix: s3_prefix,
        callLogTableName: callLogTableName,
        connection_url: connection_url,
        enableReference: enableReference,
        opensearch_account: opensearch_account,
        opensearch_passwd: opensearch_passwd,
        opensearch_url: opensearch_url,
        path: 'https://'+distribution.domainName+'/',   
        kendraIndex: kendraIndex,
        roleArn: roleLambdaWebsocket.roleArn,
        debugMessageMode: debugMessageMode,
        rag_method: rag_method,
        useParallelRAG: useParallelRAG,
        numberOfRelevantDocs: numberOfRelevantDocs,
        kendraMethod: kendraMethod,
        LLM_for_chat:JSON.stringify(LLM_for_chat),
        LLM_for_multimodal:JSON.stringify(LLM_for_multimodal),
        LLM_embedding: JSON.stringify(titan_embedding_v2),
        priority_search_embedding: JSON.stringify(titan_embedding_v1),
        capabilities: capabilities,
        googleApiSecret: googleApiSecret.secretName,
        allowDualSearch: allowDualSearch,
        enableHybridSearch: enableHybridSearch,
        projectName: projectName,
        separated_chat_history: separated_chat_history,
        enalbeParentDocumentRetrival: enalbeParentDocumentRetrival,
        speech_generation: speech_generation,
        flow_id: flow_id,
        flow_alias: flow_alias
      }
    });     
    lambdaChatWebsocket.grantInvoke(new iam.ServicePrincipal('apigateway.amazonaws.com'));  
    s3Bucket.grantReadWrite(lambdaChatWebsocket); // permission for s3
    callLogDataTable.grantReadWriteData(lambdaChatWebsocket); // permission for dynamo 
    
    if(debug) {
      new cdk.CfnOutput(this, 'function-chat-ws-arn', {
        value: lambdaChatWebsocket.functionArn,
        description: 'The arn of lambda webchat.',
      }); 
    }

    const integrationUri = `arn:aws:apigateway:${region}:lambda:path/2015-03-31/functions/${lambdaChatWebsocket.functionArn}/invocations`;    
    const cfnIntegration = new apigatewayv2.CfnIntegration(this, `api-integration-for-${projectName}`, {
      apiId: websocketapi.attrApiId,
      integrationType: 'AWS_PROXY',
      credentialsArn: role.roleArn,
      connectionType: 'INTERNET',
      description: 'Integration for connect',
      integrationUri: integrationUri,
    });  

    new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-connect`, {
      apiId: websocketapi.attrApiId,
      routeKey: "$connect", 
      apiKeyRequired: false,
      authorizationType: "NONE",
      operationName: 'connect',
      target: `integrations/${cfnIntegration.ref}`,      
    }); 

    new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-disconnect`, {
      apiId: websocketapi.attrApiId,
      routeKey: "$disconnect", 
      apiKeyRequired: false,
      authorizationType: "NONE",
      operationName: 'disconnect',
      target: `integrations/${cfnIntegration.ref}`,      
    }); 

    new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-default`, {
      apiId: websocketapi.attrApiId,
      routeKey: "$default", 
      apiKeyRequired: false,
      authorizationType: "NONE",
      operationName: 'default',
      target: `integrations/${cfnIntegration.ref}`,      
    }); 

    new apigatewayv2.CfnStage(this, `api-stage-for-${projectName}`, {
      apiId: websocketapi.attrApiId,
      stageName: stage
    }); 

    new cdk.CfnOutput(this, `FAQ-Update-for-${projectName}`, {
      value: 'aws kendra create-faq --index-id '+kendraIndex+' --name faq-banking --s3-path \'{\"Bucket\":\"'+s3Bucket.bucketName+'\", \"Key\":\"faq/faq-banking.csv\"}\' --role-arn '+roleLambdaWebsocket.roleArn+' --language-code ko --region '+kendra_region+' --file-format CSV',
      description: 'The commend for uploading contents of FAQ',
    });

    // S3 - Lambda(S3 event) - SQS(Satandard) - Lambda(S3 event manager) - SQS(fifo) - Lambda(document)
    /*
    // SQS for S3 event
    const queueS3event = new sqs.Queue(this, `queue-s3-event-for-${projectName}`, {
      visibilityTimeout: cdk.Duration.seconds(600),
      //queueName: `queue-s3-event-for-${projectName}.fifo`,  # fifo
      //fifo: true,
      //contentBasedDeduplication: false,
      queueName: `queue-s3-event-for-${projectName}`,      
      deliveryDelay: cdk.Duration.millis(0),
      retentionPeriod: cdk.Duration.days(2),
    });

    // Lambda for s3 event
    const lambdaS3event = new lambda.Function(this, `lambda-s3-event-for-${projectName}`, {
      description: 'lambda for s3 event',
      functionName: `lambda-s3-event-for-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-s3-event')),
      timeout: cdk.Duration.seconds(120),      
      environment: {
        queueS3event: queueS3event.queueUrl
      }
    });
    queueS3event.grantSendMessages(lambdaS3event); // permision for SQS putItem

    // SQS for S3 event (fifo) 
    let queueUrl:string[] = [];
    let queue:any[] = [];
    for(let i=0;i<LLM_for_chat.length;i++) {
      queue[i] = new sqs.Queue(this, 'QueueS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(600),
        queueName: `queue-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(2),
      });
      queueUrl.push(queue[i].queueUrl);
    }

    // Lambda for s3 event manager
    const lambdaS3eventManager = new lambda.Function(this, `lambda-s3-event-manager-for-${projectName}`, {
      description: 'lambda for s3 event manager',
      functionName: `lambda-s3-event-manager-for-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-s3-event-manager')),
      timeout: cdk.Duration.seconds(60),      
      environment: {
        sqsUrl: queueS3event.queueUrl,
        sqsFifoUrl: JSON.stringify(queueUrl),
        nqueue: String(LLM_for_chat.length)
      }
    });
    lambdaS3eventManager.addEventSource(new SqsEventSource(queueS3event)); // permission for SQS
    for(let i=0;i<LLM_for_chat.length;i++) {
      queue[i].grantSendMessages(lambdaS3eventManager); // permision for SQS putItem
    }

    // Lambda for document manager
    let lambdDocumentManager:any[] = [];
    for(let i=0;i<LLM_for_chat.length;i++) {
      lambdDocumentManager[i] = new lambda.DockerImageFunction(this, `lambda-document-manager-for-${projectName}-${i}`, {
        description: 'S3 document manager',
        functionName: `lambda-document-manager-for-${projectName}-${i}`,
        role: roleLambdaWebsocket,
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-document-manager')),
        timeout: cdk.Duration.seconds(600),
        memorySize: 2048,
        environment: {
          bedrock_region: LLM_for_chat[i].bedrock_region,
          s3_bucket: s3Bucket.bucketName,
          s3_prefix: s3_prefix,
          kendra_region: String(kendra_region),
          opensearch_account: opensearch_account,
          opensearch_passwd: opensearch_passwd,
          opensearch_url: opensearch_url,
          kendraIndex: kendraIndex,
          roleArn: roleLambdaWebsocket.roleArn,
          path: 'https://'+distribution.domainName+'/', 
          capabilities: capabilities,
          sqsUrl: queueUrl[i],
          max_object_size: String(max_object_size),
          enableHybridSearch: enableHybridSearch,
          supportedFormat: supportedFormat,
          LLM_for_chat: JSON.stringify(LLM_for_chat),
          enableParallelSummary: enableParallelSummary
        }
      });         
      s3Bucket.grantReadWrite(lambdDocumentManager[i]); // permission for s3
      lambdDocumentManager[i].addEventSource(new SqsEventSource(queue[i])); // permission for SQS
    } 

    // s3 event source
    const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
      events: [
        s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE,
        s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
      ],
      filters: [
        { prefix: s3_prefix+'/' },
      ]
    });
    lambdaS3event.addEventSource(s3PutEventSource); */

    // S3 - Lambda(S3 event) - SQS(fifo) - Lambda(document)
    // DLQ
    let dlq:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      dlq[i] = new sqs.Queue(this, 'DlqS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(600),
        queueName: `dlq-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(14),
      });
    }

    // SQS for S3 event (fifo) 
    let queueUrl:string[] = [];
    let queue:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      queue[i] = new sqs.Queue(this, 'QueueS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(600),
        queueName: `queue-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(2),
        deadLetterQueue: {
          maxReceiveCount: 4,
          queue: dlq[i]
        }
      });
      queueUrl.push(queue[i].queueUrl);
    }

    // Lambda for s3 event manager
    const lambdaS3eventManager = new lambda.Function(this, `lambda-s3-event-manager-for-${projectName}`, {
      description: 'lambda for s3 event manager',
      functionName: `lambda-s3-event-manager-for-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-s3-event-manager')),
      timeout: cdk.Duration.seconds(60),      
      environment: {
        sqsFifoUrl: JSON.stringify(queueUrl),
        nqueue: String(LLM_embedding.length)
      }
    });
    for(let i=0;i<LLM_embedding.length;i++) {
      queue[i].grantSendMessages(lambdaS3eventManager); // permision for SQS putItem
    }

    // Lambda for document manager
    let lambdDocumentManager:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      lambdDocumentManager[i] = new lambda.DockerImageFunction(this, `lambda-document-manager-for-${projectName}-${i}`, {
        description: 'S3 document manager',
        functionName: `lambda-document-manager-for-${projectName}-${i}`,
        role: roleLambdaWebsocket,
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-document-manager')),
        timeout: cdk.Duration.seconds(600),
        memorySize: 2048,
        environment: {
          s3_bucket: s3Bucket.bucketName,
          s3_prefix: s3_prefix,
          kendra_region: String(kendra_region),
          opensearch_account: opensearch_account,
          opensearch_passwd: opensearch_passwd,
          opensearch_url: opensearch_url,
          kendraIndex: kendraIndex,
          roleArn: roleLambdaWebsocket.roleArn,
          path: 'https://'+distribution.domainName+'/', 
          capabilities: capabilities,
          sqsUrl: queueUrl[i],
          max_object_size: String(max_object_size),
          enableHybridSearch: enableHybridSearch,
          supportedFormat: supportedFormat,
          LLM_for_chat:JSON.stringify(LLM_for_chat),
          LLM_for_multimodal:JSON.stringify(LLM_for_multimodal),
          LLM_embedding: JSON.stringify(titan_embedding_v2),
          enableParallelSummary: enableParallelSummary,
          enalbeParentDocumentRetrival: enalbeParentDocumentRetrival
        }
      });         
      s3Bucket.grantReadWrite(lambdDocumentManager[i]); // permission for s3
      lambdDocumentManager[i].addEventSource(new SqsEventSource(queue[i])); // permission for SQS
    }
    
    // s3 event source
    const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
      events: [
        s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE,
        s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
      ],
      filters: [
        { prefix: s3_prefix+'/' },
      ]
    });
    lambdaS3eventManager.addEventSource(s3PutEventSource); 

    // lambda - provisioning
    const lambdaProvisioning = new lambda.Function(this, `lambda-provisioning-for-${projectName}`, {
      description: 'lambda to earn provisioning info',
      functionName: `lambda-provisioning-api-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-provisioning')),
      timeout: cdk.Duration.seconds(30),
      environment: {
        wss_url: wss_url,
      }
    });

    // POST method - provisioning
    const provisioning_info = api.root.addResource("provisioning");
    provisioning_info.addMethod('POST', new apiGateway.LambdaIntegration(lambdaProvisioning, {
      passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
      credentialsRole: role,
      integrationResponses: [{
        statusCode: '200',
      }], 
      proxy:false, 
    }), {
      methodResponses: [  
        {
          statusCode: '200',
          responseModels: {
            'application/json': apiGateway.Model.EMPTY_MODEL,
          }, 
        }
      ]
    }); 

    // cloudfront setting for provisioning api
    distribution.addBehavior("/provisioning", new origins.RestApiOrigin(api), {
      cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
      allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
      viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    });

    // Poly Role
    const roleLambdaPolly = new iam.Role(this, `role-lambda-polly-for-${projectName}`, {
      roleName: `role-lambda-polly-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
      )
    });
    roleLambdaPolly.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
    });
    roleLambdaPolly.attachInlinePolicy( 
      new iam.Policy(this, `polly-api-invoke-policy-for-${projectName}`, {
        statements: [apiInvokePolicy],
      }),
    );  

  /*  const PollyPolicy = new iam.PolicyStatement({  
      actions: ['polly:*'],
      resources: ['*'],
    }); */
    roleLambdaPolly.attachInlinePolicy(
      new iam.Policy(this, `polly-policy-${projectName}`, {
        statements: [PollyPolicy],
      }),
    ); 

    // lambda - polly
    const lambdaPolly = new lambda.Function(this, `lambda-polly-for-${projectName}`, {
      description: 'lambda polly for speech translation',
      functionName: `lambda-polly-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      role: roleLambdaPolly,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-polly')),
      timeout: cdk.Duration.seconds(30),
      environment: {
        s3_bucket: s3Bucket.bucketName,
      }
    });
    s3Bucket.grantReadWrite(lambdaPolly); // permission for s3

    const polySpeech = api.root.addResource("speech");
    polySpeech.addMethod('POST', new apiGateway.LambdaIntegration(lambdaPolly, {
      passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
      credentialsRole: role,
      integrationResponses: [{
        statusCode: '200',
      }], 
      proxy:false, 
    }), {
      methodResponses: [  
        {
          statusCode: '200',
          responseModels: {
            'application/json': apiGateway.Model.EMPTY_MODEL,
          }, 
        }
      ]
    }); 

    // cloudfront setting for api gateway    
    distribution.addBehavior("/speech", new origins.RestApiOrigin(api), {
      cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
      allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
      viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    });

    
    // deploy components
    new componentDeployment(scope, `component-deployment-of-${projectName}`, websocketapi.attrApiId)     

    //const wsOriginRequestPolicy = new cloudFront.OriginRequestPolicy(this, `webSocketPolicy`, {
    //  originRequestPolicyName: "webSocketPolicy",
    //  comment: `A default WebSocket policy`,
    //  cookieBehavior: cloudFront.OriginRequestCookieBehavior.none(),
    //  headerBehavior: cloudFront.OriginRequestHeaderBehavior.allowList(`Sec-WebSocket-Key`, `Sec-WebSocket-Version`, `Sec-WebSocket-Protocol`, `Sec-WebSocket-Accept`),
    //  queryStringBehavior: cloudFront.OriginRequestQueryStringBehavior.none(),
    //});
    
    // cloudfront setting for api gateway    
    // distribution.addBehavior("/ws", new origins.HttpOrigin(websocketapi), {
    //  cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    //  allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,  
    //  viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    //});        
  }
}

export class componentDeployment extends cdk.Stack {
  constructor(scope: Construct, id: string, appId: string, props?: cdk.StackProps) {    
    super(scope, id, props);

    new apigatewayv2.CfnDeployment(this, `api-deployment-of-${projectName}`, {
      apiId: appId,
      description: "deploy api gateway using websocker",  // $default
      stageName: stage
    });   
  }
} 
