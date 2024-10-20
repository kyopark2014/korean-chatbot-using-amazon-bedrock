const langstate = 'korean'; // korean or english
// earn endpoint 
let endpoint = localStorage.getItem('wss_url');
if(endpoint=="") {
    console.log('provisioning is required!');
}
console.log('endpoint: ', endpoint);

console.log('feedback...');
const feedback = document.getElementById('feedback');
feedback.style.display = 'none';    

let webSocket
let isConnected = false;
webSocket = connect(endpoint, 'initial');

// Documents
const title = document.querySelector('#title');
const sendBtn = document.querySelector('#sendBtn');
const message = document.querySelector('#chatInput')
const chatPanel = document.querySelector('#chatPanel');

HashMap = function() {
    this.map = new Array();
};

HashMap.prototype = {
    put: function(key, value) {
        this.map[key] = value;
    },
    get: function(key) {
        return this.map[key];
    },
    size: function() {
        var keys = new Array();
        for(i in this.map) {
            keys.push(i);
        }
        return keys.length;
    },
    remove: function(key) {
        delete this.map[key];
    },
    getKeys: function() {
        var keys = new Array();
        for(i in this.map) {
            keys.push(i);
        }
        return keys;
    }
};

let isResponsed = new HashMap();
let indexList = new HashMap();
let retryNum = new HashMap();

// message log list
let msglist = [];
let maxMsgItems = 200;
let msgHistory = new HashMap();
let sentTime = new HashMap();

let undelivered = new HashMap();
let retry_count = 0;
function sendMessage(message) {
    if(!isConnected) {
        console.log('reconnect...'); 
        webSocket = connect(endpoint, 'reconnect');
        
        if(langstate=='korean') {
            addNotifyMessage("재연결중입니다. 연결후 자동 재전송합니다.");
        }
        else {
            addNotifyMessage("We are connecting again. Your message will be retried after connection.");                        
        }

        undelivered.put(message.request_id, message);
        console.log('undelivered message: ', message);
        
        return false
    }
    else {
        webSocket.send(JSON.stringify(message));     
        console.log('message: ', message);   

        return true;
    }     
}

// TTS
let tts = localStorage.getItem('tts'); // set userID if exists 
if(tts=="" || tts==null) {
    ttsMode = 'disable';    
}
else {
    ttsMode = tts;
}
console.log('ttsMode: ', ttsMode);

if(ttsMode=='enable') {
    var AudioContext;
    var audioContext;

    window.onload = function() {
        navigator.mediaDevices.getUserMedia({ audio: true }).then(() => {
            AudioContext = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioContext();
        }).catch(e => {
            console.error(`Audio permissions denied: ${e}`);
        });
    }
}

let sentance = new HashMap();
let lineText = "";
let playList = [];
let playId = 0;
let requestId = ""
let next = true;
let isPlayedTTS = new HashMap();

let audioData = new HashMap();
function loadAudio(requestId, text) {
    const uri = "speech";
    const xhr = new XMLHttpRequest();

    let speed = 120;
    let voiceId;
    let langCode;
    if(conversationType=='english') {
        langCode = 'en-US';
        voiceId = 'Ivy';
    }
    else {
        langCode = 'ko-KR';  // ko-KR en-US(영어)) ja-JP(일본어)) cmn-CN(중국어)) sv-SE(스페인어))
        voiceId = 'Seoyeon';
    }
    
    // voiceId: 'Aditi'|'Amy'|'Astrid'|'Bianca'|'Brian'|'Camila'|'Carla'|'Carmen'|'Celine'|'Chantal'|'Conchita'|'Cristiano'|'Dora'|'Emma'|'Enrique'|'Ewa'|'Filiz'|'Gabrielle'|'Geraint'|'Giorgio'|'Gwyneth'|'Hans'|'Ines'|'Ivy'|'Jacek'|'Jan'|'Joanna'|'Joey'|'Justin'|'Karl'|'Kendra'|'Kevin'|'Kimberly'|'Lea'|'Liv'|'Lotte'|'Lucia'|'Lupe'|'Mads'|'Maja'|'Marlene'|'Mathieu'|'Matthew'|'Maxim'|'Mia'|'Miguel'|'Mizuki'|'Naja'|'Nicole'|'Olivia'|'Penelope'|'Raveena'|'Ricardo'|'Ruben'|'Russell'|'Salli'|'Seoyeon'|'Takumi'|'Tatyana'|'Vicki'|'Vitoria'|'Zeina'|'Zhiyu'|'Aria'|'Ayanda'|'Arlet'|'Hannah'|'Arthur'|'Daniel'|'Liam'|'Pedro'|'Kajal'|'Hiujin'|'Laura'|'Elin'|'Ida'|'Suvi'|'Ola'|'Hala'|'Andres'|'Sergio'|'Remi'|'Adriano'|'Thiago'|'Ruth'|'Stephen'|'Kazuha'|'Tomoko'

    // Aditi: neural is not support
    // Amy: good
    // Astrid: neural is not support
    // Bianca: 스페인어? (x)
    // Brian: 
    // Camila (o)
   
    if(conversationType == 'translation') {
        langCode = langCode;
        voiceId = voiceId; // child Ivy, adult Joanna
        speed = '120';
    }    
    // console.log('voiceId: ', voiceId);
    
    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            response = JSON.parse(xhr.responseText);
            // console.log("response: ", response);

            audioData[requestId+text] = response.body;

            // console.log('successfully loaded. text= '+text);
            // console.log(response.body);
            // console.log(audioData[requestId+text]);
        }
    };
    
    var requestObj = {
        "text": text,
        "voiceId": voiceId,
        "langCode": langCode,
        "speed": speed
    }
    // console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

let retryCounter;
function checkingDelayedPlayList() {
    // console.log('->checking delayed played list ('+retryCounter+')');  
    playAudioList();

    let isCompleted = true;
    for(let i=0; i<playList.length;i++) {
        if(playList[i].played == false) {
            isCompleted = false;
            break;
        }
    }
    
    if(isCompleted==true) {
        playList = [];
    } 
    else {
        playTm = setTimeout(function () {           
            retryCounter--;
    
            if(retryCounter>0) {
                checkingDelayedPlayList();
            }
        }, 1000);
    }    
}

function playAudioList() {
    // console.log('next = '+next+', playList: '+playList.length);
    
    for(let i=0; i<playList.length;i++) {
        // console.log('audio data--> ', audioData[requestId+playList[i].text])
        // console.log('playList: ', playList);

        if(next == true && playList[i].played == false && requestId == playList[i].requestId && audioData[requestId+playList[i].text]) {
            // console.log('[play] '+i+': '+requestId+', text: '+playList[i].text);
            playId = i;
            playAudioLine(audioData[requestId+playList[i].text]);            

            next = false;
            break;
        }
        else if(requestId != playList[i].requestId) {
            playList[i].played = true;
        }
    }
}

async function playAudioLine(audio_body){    
    var sound = "data:audio/ogg;base64,"+audio_body;
    
    var audio = document.querySelector('audio');
    audio.src = sound;
    
    // console.log('play audio');

    await playAudio(audio)
}

function delay(ms = 1000) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

// audio play
var audio = document.querySelector('audio');
audio.addEventListener("ended", function() {
    // console.log("playId: ", playId)

    if(playList[playId] != undefined) {
        console.log("played audio: ", playList[playId].text)

        delay(1000)

        next = true;
        playList[playId].played = true;
        audioData.remove([requestId+playList[playId].text]);

        playAudioList()
    }        
    else {
        playList = [];
        playId = 0;
    }
});

function playAudio(audio) {
    return new Promise(res=>{
        audio.play()
        audio.onended = res
    })
}

// Keep alive
let tm;
function ping() {
    console.log('->ping');
    webSocket.send('__ping__');
    tm = setTimeout(function () {
        console.log('reconnect...');    
        
        webSocket = connect(endpoint, 'reconnect');
    }, 5000);
}
function pong() {
    clearTimeout(tm);
}

function connect(endpoint, type) {
    const ws = new WebSocket(endpoint);

    // connection event
    ws.onopen = function () {
        console.log('connected...');
        isConnected = true;

        if(undelivered.size() && retry_count>0) {
            let keys = undelivered.getKeys();
            console.log('retry undelived messags!');            
            console.log('keys: ', keys);
            console.log('retry_count: ', retry_count);

            for(i in keys) {
                let message = undelivered.get(keys[i])
                console.log('message', message)
                if(!sendMessage(message)) break;
                else {
                    undelivered.remove(message.request_id)
                }
            }
            retry_count--;
        }
        else {
            retry_count = 3
        }

        if(type == 'initial')
            setInterval(ping, 40000);  // ping interval: 40 seconds
    };

    // message 
    ws.onmessage = function (event) {        
        if (event.data.substr(1,8) == "__pong__") {
            console.log('<-pong');
            pong();
            return;
        }
        else {
            response = JSON.parse(event.data)

            if(response.request_id) {
                if(!indexList.get(response.request_id+':receive')) { // the first received message
                    let current = new Date();
                    let elapsed = (current - sentTime.get(response.request_id))/1000;
                    // console.log('elapsed time: ', elapsed);
                }
                // console.log('response: ', response);

                if(response.status == 'completed') {
                    // console.log('completed!');
                    feedback.style.display = 'none';          
                    // console.log('received message: ', response.msg);
                    addReceivedMessage(response.request_id, response.msg);  

                    if(ttsMode=='enable') {                    
                        // console.log('Is already played? ', isPlayedTTS[response.request_id]);
                        if(isPlayedTTS[response.request_id] == undefined) {
                            requestId = response.request_id;
                            playList.push({
                                'played': false,
                                'requestId': requestId,
                                'text': response.msg
                            });
                            // console.log('new play list : '+response.msg+ '('+requestId+')')

                            lineText = "";      
                        
                            loadAudio(response.request_id, response.msg);
                                
                            next = true;
                            playAudioList();
                        }    
                        
                        retryCounter = 5;
                        checkingDelayedPlayList();
                    }
                }                
                
                else if(response.status == 'proceeding') {
                    // console.log('proceeding...');
                    feedback.style.display = 'none';
                    sentance.put(response.request_id, sentance.get(response.request_id)+response.msg); 
                    addReceivedMessage(response.request_id, response.msg);  

                    if(ttsMode=='enable') {
                        lineText += response.msg;
                        lineText = lineText.replace('\n','');
                        if(lineText.length>3 && (response.msg == '.' || response.msg == '?' || response.msg == '!')) {     
                            // console.log('lineText: ', lineText);
                            text = lineText
                            playList.push({
                                'played': false,
                                'requestId': requestId,
                                'text': text
                            });
                            // console.log('new play list : '+text+ '('+requestId+')')

                            lineText = "";
                
                            isPlayedTTS[response.request_id] = true;
                            loadAudio(response.request_id, text);
                        }
                        
                        requestId = response.request_id;
                        playAudioList();
                    } 
                }        

                else if(response.status == 'istyping') {
                    feedback.style.display = 'inline';
                    console.log('received message: ', response.msg);
                    feedback.innerHTML = '<i>'+response.msg+'</i>'; 
                    
                    sentance.put(response.request_id, "");
                }        
                else if(response.status == 'debug') {
                    feedback.style.display = 'none';
                    console.log('debug: ', response.msg);
                    // addNotifyMessage(response.msg);
                    addReceivedMessage(response.request_id, response.msg);  
                }          
                else if(response.status == 'error') {
                    feedback.style.display = 'none';
                    console.log('error: ', response.msg);

                    if(response.msg.indexOf('throttlingException') || response.msg.indexOf('Too many requests') || response.msg.indexOf('too many requests')) {
                        addNotifyMessage('허용된 요청수를 초과하였습니다. 추후 다시 재시도 해주세요.');  
                    }
                    else {
                        addNotifyMessage(response.msg);
                    }
                }   
            }
            else {
                console.log('system message: ', event.data);
            }
        }        
    };

    // disconnect
    ws.onclose = function () {
        console.log('disconnected...!');
        isConnected = false;

        ws.close();
        console.log('the session will be closed');
    };

    // error
    ws.onerror = function (error) {
        console.log(error);

        ws.close();
        console.log('the session will be closed');
    };

    return ws;
}

let callee = "AWS";
let index=0;

let userId = localStorage.getItem('userId'); // set userID if exists 
if(userId=="") {
    userId = uuidv4();
}
console.log('userId: ', userId);

let conversationType = localStorage.getItem('conv_type'); // set conv_type if exists 
if(conversationType=="") {
    conversationType = "normal";
}
console.log('conversationType: ', conversationType);

for (i=0;i<maxMsgItems;i++) {
    msglist.push(document.getElementById('msgLog'+i));

    // add listener        
    (function(index) {
        msglist[index].addEventListener("click", function() {
            if(msglist.length < maxMsgItems) i = index;
            else i = index + maxMsgItems;

            console.log('click! index: '+index);
        })
    })(i);
}

calleeName.textContent = "Chatbot";  
calleeId.textContent = "AWS";


if(langstate=='korean') {
    addNotifyMessage("Amazon Bedrock을 이용하여 채팅을 시작합니다.");
    addReceivedMessage(uuidv4(), "Amazon Bedrock을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다.")
}
else {
    addNotifyMessage("Start chat with Amazon Bedrock");             
    addReceivedMessage(uuidv4(), "Welcome to Amazon Bedrock. Use the conversational chatbot and summarize documents, TXT, PDF, and CSV. ")           
}

// get history
function getAllowTime() {    
    let allowableDays = 2; // two day's history
    
    let current = new Date();
    let allowable = new Date(current.getTime() - 24*60*60*1000*allowableDays);  
    let allowTime = getDate(allowable)+' '+getTime(current);
    console.log('Current Time: ', getDate(current)+' '+getTime(current));
    console.log('Allow Time: ', allowTime);
    
    return allowTime;
}
let allowTime = getAllowTime();
getHistory(userId, allowTime);

// Listeners
message.addEventListener('keyup', function(e){
    if (e.keyCode == 13) {
        onSend(e);
    }
});

// refresh button
refreshChatWindow.addEventListener('click', function(){
    console.log('go back user input menu');
    window.location.href = "index.html";
});

// depart button
depart.addEventListener('click', function(){
    console.log('depart icon');
    
    deleteItems(userId);    
});

sendBtn.addEventListener('click', onSend);
function onSend(e) {
    e.preventDefault();
    
    if(message.value != '') {
        console.log("msg: ", message.value);

        let current = new Date();
        let datastr = getDate(current);
        let timestr = getTime(current);
        let requestTime = datastr+' '+timestr

        let requestId = uuidv4();
        addSentMessage(requestId, timestr, message.value);
        
        if(conversationType=='qa-all') {
            type = "text",
            conv_type = 'qa',
            rag_type = 'all',
            function_type = 'rag'
        }
        else if(conversationType=='qa-kendra') {
            type = "text",
            conv_type = 'qa',
            rag_type = 'kendra',
            function_type = 'rag'
        }
        else if(conversationType=='qa-opensearch') {
            type = "text",
            conv_type = 'qa',
            rag_type = 'opensearch',
            function_type = 'rag'
        }
        else if(conversationType=='qa-faiss') {
            type = "text",
            conv_type = 'qa',
            rag_type = 'faiss',
            function_type = 'rag'
        }
        else if(conversationType=='dual-search') {
            type = "text",
            conv_type = 'qa',
            rag_type = 'all',
            function_type = 'dual-search'
        }
        else if(conversationType=='code-generation-python') {
            type = "code",
            conv_type = 'qa',
            rag_type = 'opensearch',
            function_type = 'code-generation-python'
        }
        else if(conversationType=='code-generation-nodejs') {
            type = "code",
            conv_type = 'qa',
            rag_type = 'opensearch',
            function_type = 'code-generation-nodejs'
        }
        else {
            type = "text",
            conv_type = conversationType,
            rag_type = ''
            function_type = ''
        }
        
        sendMessage({
            "user_id": userId,
            "request_id": requestId,
            "request_time": requestTime,        
            "type": type,
            "body": message.value,
            "conv_type": conv_type,
            "rag_type": rag_type,
            "function_type": function_type
        })
        
        sentTime.put(requestId, current);
    }
    message.value = "";

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

function uuidv4() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

(function() {
    window.addEventListener("focus", function() {
//        console.log("Back to front");

        // if(msgHistory.get(callee))
        //    updateCallLogToDisplayed();
    })
})();

function getDate(current) {    
    return current.toISOString().slice(0,10);
}

function getTime(current) {
    let time_map = [current.getHours(), current.getMinutes(), current.getSeconds()].map((a)=>(a < 10 ? '0' + a : a));
    return time_map.join(':');
}

function addSentMessage(requestId, timestr, text) {
    idx = index;
    if(!indexList.get(requestId+':send')) {
        indexList.put(requestId+':send', index);           
        index++;  
    }
    else {
        idx = indexList.get(requestId+':send');
        console.log("reused index="+index+', id='+requestId+':send');        
    }
    console.log("idx:", idx);   

    var length = text.length;    
    console.log('length: ', length);
    if(length < 10) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender20 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;   
    }
    else if(length < 13) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender25 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;   
    }
    else if(length < 17) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender30 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }  
    else if(length < 21) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender35 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }
    else if(length < 26) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender40 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }
    else if(length < 35) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender50 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }
    else if(length < 80) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender60 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }  
    else if(length < 145) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender70 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }  
    else {
        msglist[idx].innerHTML = 
            `<div class="chat-sender80 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }     

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}       

function addSentMessageForSummary(requestId, timestr, text) {  
    console.log("sent message: "+text);

    idx = index;
    if(!indexList.get(requestId+':send')) {
        indexList.put(requestId+':send', index);       
        index++;      
    }
    else {
        idx = indexList.get(requestId+':send');
        console.log("reused index="+index+', id='+requestId+':send');        
    }
    console.log("index:", index);   

    let length = text.length;
    if(length < 100) {
        msglist[idx].innerHTML = 
            `<div class="chat-sender60 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;   
    }
    else {
        msglist[idx].innerHTML = 
            `<div class="chat-sender80 chat-sender--right"><h1>${timestr}</h1>${text}&nbsp;<h2 id="status${idx}"></h2></div>`;
    }   

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}  

function addReceivedMessage(requestId, msg) {
    // console.log("add received message: "+msg);
    sender = "Chatbot"
    
    idx = index;
    if(!indexList.get(requestId+':receive')) {
        indexList.put(requestId+':receive', index);         
        index++;    
    }
    else {
        idx = indexList.get(requestId+':receive');
        // console.log("reused index="+index+', id='+requestId+':receive');        
    }
    // console.log("index:", index);   

    msg = msg.replaceAll("\n", "<br/>");

    let length = msg.length;
    // console.log("length: ", length);

    if(length < 10) {
        msglist[idx].innerHTML = `<div class="chat-receiver20 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 13) {
        msglist[idx].innerHTML = `<div class="chat-receiver25 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 17) {
        msglist[idx].innerHTML = `<div class="chat-receiver30 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 21) {
        msglist[idx].innerHTML = `<div class="chat-receiver35 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 25) {
        msglist[idx].innerHTML = `<div class="chat-receiver40 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 35) {
        msglist[idx].innerHTML = `<div class="chat-receiver50 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 80) {
        msglist[idx].innerHTML = `<div class="chat-receiver60 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else if(length < 145) {
        msglist[idx].innerHTML = `<div class="chat-receiver70 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
    else {
        msglist[idx].innerHTML = `<div class="chat-receiver80 chat-receiver--left"><h1>${sender}</h1>${msg}&nbsp;</div>`;  
    }
     
    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

function addNotifyMessage(msg) {
    console.log("index:", index);   

    msglist[index].innerHTML =  
        `<div class="notification-text">${msg}</div>`;     

    index++;

    chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
}

refreshChatWindow.addEventListener('click', function(){
    console.log('update chat window');
    // updateChatWindow(callee);
});

attachFile.addEventListener('click', function(){
    console.log('click: attachFile');

    let input = $(document.createElement('input')); 
    input.attr("type", "file");
    input.trigger('click');    
    
    $(document).ready(function() {
        input.change(function(evt) {
            var input = this;
            var url_file = $(this).val();
            var ext = url_file.substring(url_file.lastIndexOf('.') + 1).toLowerCase();
            //var filename = url_file.substring(url_file.lastIndexOf('\\') + 1).toLowerCase();
            var filename = url_file.substring(url_file.lastIndexOf('\\') + 1);

            console.log('url: ' + url_file);
            console.log('filename: ' + filename);
            console.log('ext: ' + ext);

            if(ext == 'pdf') {
                contentType = 'application/pdf'           
            }
            else if(ext == 'txt') {
                contentType = 'text/plain'
            }
            else if(ext == 'csv') {
                contentType = 'text/csv'
            }
            else if(ext == 'ppt') {
                contentType = 'application/vnd.ms-powerpoint'
            }
            else if(ext == 'pptx') {
                contentType = 'application/vnd.ms-powerpoint'
            }
            else if(ext == 'doc' || ext == 'docx') {
                contentType = 'application/msword'
            }
            else if(ext == 'xls') {
                contentType = 'application/vnd.ms-excel'
            }
            else if(ext == 'py') {
                contentType = 'application/x-python-code'
            }
            else if(ext == 'js') {
                contentType = 'application/javascript'
            }
            else if(ext == 'md') {
                contentType = 'text/markdown'
            }
            else if(ext == 'png') {
                contentType = 'image/png'
            }
            else if(ext == 'jpeg' || ext == 'jpg') {
                contentType = 'image/jpeg'
            }
            console.log('contentType: ', contentType)

            let current = new Date();
            let datastr = getDate(current);
            let timestr = getTime(current);
            let requestTime = datastr+' '+timestr
            let requestId = uuidv4();

            let command = message.value;
            if(ext == 'png' || ext == 'jpeg' || ext == 'jpg') {
                addSentMessageForSummary(requestId, timestr, message.value+"<br>"+"uploading the selected file in order to summarize...");

                message.value = "";
            }
            else {
                addSentMessageForSummary(requestId, timestr, "uploading the selected file in order to summarize...");
            }

            const uri = "upload";
            const xhr = new XMLHttpRequest();
        
            xhr.open("POST", uri, true);
            xhr.onreadystatechange = () => {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    response = JSON.parse(xhr.responseText);
                    console.log("response: " + JSON.stringify(response));
                                        
                    // upload the file
                    const body = JSON.parse(response.body);
                    console.log('body: ', body);

                    const uploadURL = body.UploadURL;                    
                    console.log("UploadURL: ", uploadURL);

                    var xmlHttp = new XMLHttpRequest();
                    xmlHttp.open("PUT", uploadURL, true);       

                    //let formData = new FormData();
                    //formData.append("attachFile" , input.files[0]);
                    //console.log('uploading file info: ', formData.get("attachFile"));

                    const blob = new Blob([input.files[0]], { type: contentType });

                    xmlHttp.onreadystatechange = function() {
                        if (xmlHttp.readyState == XMLHttpRequest.DONE && xmlHttp.status == 200 ) {
                            console.log(xmlHttp.responseText);

                            function_type = 'upload'
                            if(conversationType=='qa-all') {
                                conv_type = 'qa',
                                rag_type = 'all'
                            }
                            else if(conversationType=='qa-kendra') {
                                conv_type = 'qa',
                                rag_type = 'kendra'
                            }
                            else if(conversationType=='qa-opensearch') {
                                conv_type = 'qa',
                                rag_type = 'opensearch'
                            }
                            else if(conversationType=='qa-faiss') {
                                conv_type = 'qa',
                                rag_type = 'faiss'
                            }
                            else {
                                conv_type = conversationType,
                                rag_type = ''
                            }

                            // summary for the upload file                            
                            sendMessage({
                                "user_id": userId,
                                "request_id": requestId,
                                "request_time": requestTime,
                                "type": "document",
                                "body": filename,
                                "command": command,
                                "conv_type": conv_type,
                                "rag_type": rag_type,
                                "function_type": function_type
                            })
                        }
                        else if(xmlHttp.readyState == XMLHttpRequest.DONE && xmlHttp.status != 200) {
                            console.log('status' + xmlHttp.status);
                            alert("Try again! The request was failed.");
                        }
                    };
        
                    xmlHttp.send(blob); 
                    // xmlHttp.send(formData); 
                    console.log(xmlHttp.responseText);
                }
            };
        
            var requestObj = {
                "filename": filename,
                "contentType": contentType,
            }
            console.log("request: " + JSON.stringify(requestObj));
        
            var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});
        
            xhr.send(blob);       
        });
    });
       
    return false;
});

function getHistory(userId, allowTime) {
    const uri = "history";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            let history = JSON.parse(response['msg']);
            console.log("history: " + JSON.stringify(history));
                        
            for(let i=0; i<history.length; i++) {
                if(history[i].type=='text') {                
                    // let timestr = history[i].request_time.substring(11, 19);
                    let requestId = history[i].request_id;
                    console.log("requestId: ", requestId);
                    let timestr = history[i].request_time;
                    console.log("timestr: ", timestr);
                    let body = history[i].body;
                    console.log("question: ", body);
                    let msg = history[i].msg;
                    console.log("answer: ", msg);
                    addSentMessage(requestId, timestr, body)
                    addReceivedMessage(requestId, msg);                            
                }                 
            }         
            if(history.length>=1) {
                if(langstate=='korean') {
                    addNotifyMessage("대화를 다시 시작하였습니다.");
                }
                else {
                    addNotifyMessage("Welcome back to the conversation");                               
                }
                chatPanel.scrollTop = chatPanel.scrollHeight;  // scroll needs to move bottom
            }
        }
    };
    
    var requestObj = {
        "userId": userId,
        "allowTime": allowTime
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}

function deleteItems(userId) {
    const uri = "delete";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            console.log("response: " + JSON.stringify(response));

            window.location.href = "index.html";
        }
    };
    
    var requestObj = {
        "userId": userId
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);            
}
