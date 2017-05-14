initialization();


function handleFileSelect(files) {
    $.ajax({
        type: "POST",
        url: "/load",
        async: true,
        data: {file:files[0].name },
        success: load_receive
    });    
}

function handleOldSelect(files) {
    $.ajax({
        type: "POST",
        url: "/load_old",
        async: true,
        data: {file:files[0].name },
        success: load_old_receive
    });
}

function load_receive(response) {
    if(response.flag){
        display_num_labeled(response);
        $("button").removeAttr('disabled');
        if (!response.hasLabel){
            $("#auto").attr('disabled',true);
        }
        train_send()
    }
    else{
        window.alert("Load file failed!! Must put file under workspace/data");
    }
}

function load_old_receive(response) {
    if(response.flag){
        train_send()
    }
    else{
        window.alert("Load file failed!! Must put file under workspace/coded");
    }
}

function export_send(){
    $.ajax({
        type: "POST",
        url: "/export",
        async: true,
        data: {},
        success: export_receive
    });
}

function export_receive(response) {
    if(response.flag){
        window.alert("Export succeeded!!");
    }
    else{
        window.alert("Export failed!!");
    }
}

function plot_send() {
    $.ajax({
        type: "POST",
        url: "/plot",
        async: true,
        data: {},
        success: plot_receive
    });
}

function plot_receive(response) {

    $("#myImage").attr("src","/static/image/"+response.path);
}

function initialization(){
    // global variables
    current_node=document.getElementById("elasticinput");    
    changed=false;
    learn_result={};
    can={};
    $("#myImage").removeAttr("src")
}


function display_num_labeled(response){    
    document.getElementById("num_labeled").innerText="Documents Coded: "+response.pos.toString()+"/"+response.done.toString()+" ("+response.total.toString()+")";
}


function show_send(what,which) {
    $("ol li").css("color","white");
    $(what).css("color","yellow");
    current_node=what;
    document.getElementById("send_label").removeAttribute('disabled');
    var tmp={};
    if (which=="search"){
        tmp=search_result;        
    }
    else if (which=="learn"){
        tmp=can;
    }
    else if (which=="train"){
        tmp=train_result;
    }

    document.getElementById("which_part").value=which;
    document.getElementById("display").labeling.value=tmp[what.value].code;
    document.getElementById("displaydoc_id").value = tmp[what.value].id;
    $("#truelabel").html("True Label: "+tmp[what.value].label);
    // document.getElementById("displaydoc").innerHTML = tmp[what.value]._source.selftext;
    $("#displaydoc").html("<h3><a href=\""+tmp[what.value]["PDF Link"]+"\" target=\"blank\">"+tmp[what.value]["Document Title"]+"</a></h3>"+tmp[what.value]["Abstract"]);
}


function labeling_send(what){
    $.ajax({
        type: "POST",
        url: "/labeling",
        async: true,
        data: { id: document.getElementById("displaydoc_id").value, label: what.labeling.value },
        success: labeling_receive
    });
}

function labeling_receive(response){
    
    if(response.flag){
        display_num_labeled(response);
    }  
    nextnode=current_node.nextSibling;
    prevnode=current_node.previousSibling;
    current_node.remove();
    if(nextnode){
        show_send(nextnode,document.getElementById("which_part").value);
    }
    else if(prevnode){
        show_send(prevnode,document.getElementById("which_part").value);
    }
    else{
        document.getElementById("displaydoc_id").value = "none";
        $("#displaydoc").html("Done! Hit Next Button for next batch.");
        document.getElementById("send_label").setAttribute("disabled","disabled");
    }  
}

function auto_review(){
    var ids = {}
    for (var i = 0; i < can.length; ++i){
        ids[i]=can[i].id
    }

    $.ajax({
        type: "POST",
        url: "/auto",
        async: true,
        data: ids,
        success: auto_receive
    });
}

function auto_receive(response){
    if(response.flag){
        display_num_labeled(response);
    }
    var olnode=document.getElementById("learn_result");
    while (olnode.firstChild) {
        olnode.removeChild(olnode.firstChild);
    }
    document.getElementById("displaydoc_id").value = "none";
    $("#displaydoc").html("Done! Hit Next Button for next batch.");
    document.getElementById("send_label").setAttribute("disabled","disabled");
}

function train_send(){
    changed=false;
    $.ajax({
        type: "POST",
        url: "/train",
        async: true,
        data: { },
        success: train_receive
    }); 
}

function train_receive(response){
    learn_result=response;
    view_selection(document.getElementById("view_options"));
    $("#oldFile").removeAttr('disabled');
    if(document.getElementById("auto_plot").checked){
        plot_send()
    }
}

function view_selection(what){
    if(changed){
        train_send();        
    }
    else{
        switch(parseInt(what.selectedIndex)){
            case 0:
                view_random();
                break;
            case 1:
                view_certain();
                break;
            case 2:
                view_reuse();
        }
    }    
}

function view_random(){
    can = learn_result.random;
    var olnode=document.getElementById("learn_result");
    while (olnode.firstChild) {
        olnode.removeChild(olnode.firstChild);
    }

    for (var i = 0; i < can.length; ++i){

        var newli=document.createElement("li");
        var node=document.createTextNode( can[i]["Document Title"]);
        newli.appendChild(node);
        newli.setAttribute("value",i);
        newli.setAttribute("onclick","show_send(this,\"learn\")");
        olnode.appendChild(newli);
    }
    show_send(olnode.firstChild,"learn");
}

function view_certain(){
    var olnode=document.getElementById("learn_result");
    while (olnode.firstChild) {
        olnode.removeChild(olnode.firstChild);
    }
    if ("certain" in learn_result){
        can = learn_result.certain;      

        for (var i = 0; i < can.length; ++i){

            var newli=document.createElement("li");
            var node=document.createTextNode( can[i]["Document Title"]+" ("+can[i]["prob"].toString()+")");
            newli.appendChild(node);
            newli.setAttribute("value",i);
            newli.setAttribute("onclick","show_send(this,\"learn\")");
            olnode.appendChild(newli);
        }
        show_send(olnode.firstChild,"learn");
    }
}

function view_reuse() {
    var olnode = document.getElementById("learn_result");
    while (olnode.firstChild) {
        olnode.removeChild(olnode.firstChild);
    }
    if ("certain" in learn_result) {
        can = learn_result.reuse;
        
        for (var i = 0; i < can.length; ++i) {

            var newli = document.createElement("li");
            var node = document.createTextNode(can[i]["Document Title"] + " (" + can[i]["prob"].toString() + ")");
            newli.appendChild(node);
            newli.setAttribute("value",i);
            newli.setAttribute("onclick", "show_send(this,\"learn\")");
            olnode.appendChild(newli);
        }
        show_send(olnode.firstChild, "learn");
    }
}

function view_support(){
    if (voc){

        var olnode=document.getElementById("learn_result");
        while (olnode.firstChild) {
            olnode.removeChild(olnode.firstChild);
        }

        var ind = new Array(support.length);
        for (var i = 0; i < support.length; ++i){
            ind[i]=i;
        }
        ind.sort(function (a,b) {return Math.abs(dual_coef[b])-Math.abs(dual_coef[a])});
        for (var i = 0; i < support.length; ++i){

            var newli=document.createElement("li");
            var node=document.createTextNode( train_result[support[ind[i]]]._source.title+" ("+dual_coef[ind[i]].toString()+")");
            newli.appendChild(node);
            newli.setAttribute("value",support[ind[i]]);
            newli.setAttribute("onclick","show_send(this,\"train\")");
            olnode.appendChild(newli);
        }
        show_send(olnode.firstChild,"train");
    }
}

function view_coef(){
    if (voc){
        var display_limit = 30;
        var olnode=document.getElementById("learn_result");
        while (olnode.firstChild) {
            olnode.removeChild(olnode.firstChild);
        }

        var ind = [];
        for (var i = 0; i < voc.length; ++i){
            ind.push(i);
        }
        ind.sort(function (a,b) {return Math.abs(coef[b])-Math.abs(coef[a])});
        for (var i = 0; i < Math.min(coef.length,display_limit); ++i){
            
            var newli=document.createElement("li");
            var node=document.createTextNode( voc[ind[i]]+" ("+coef[ind[i]].toString()+")");
            newli.appendChild(node);
            newli.setAttribute("value",ind[i]);
            newli.setAttribute("onclick","feature_selection(this,\"mask\")");
            olnode.appendChild(newli);
        }

    }
}

function edit_coef(){
    if (voc){
        var olnode=document.getElementById("learn_result");
        while (olnode.firstChild) {
            olnode.removeChild(olnode.firstChild);
        }

        for (var i = 0; i < voc.length; ++i){
            if(my_mask.indexOf(i)<0){
                continue;
            }
            var newli=document.createElement("li");
            var node=document.createTextNode( voc[i]);
            newli.appendChild(node);
            newli.setAttribute("value",i);
            newli.setAttribute("onclick","feature_selection(this,\"unmask\")");
            olnode.appendChild(newli);
        }
    }
}

function restart_send(){
    if (confirm("You will loose all your effort so far, are you sure?") == true) {
        $.ajax({
            type: "POST",
            url: "/restart",
            async: true,
            data: {  },
            success: load_receive
        });
        $("#oldFile").attr('disabled',true);
        initialization();
    }    
}

function check_response(response){
    if(response=="done"){
        window.alert("Done!");
        return true;
    }
}

function highlight(text, which){
    if (which=="search"){
        var keywords=stemmer(search_key.toLowerCase()).split(' ');
        var exp=/(\w+)/g;
        function searchhighlight(match){
            if(keywords.indexOf(stemmer(match.toLowerCase()))>-1){
                return "<span style='background-color: green'>"+match+"</span>";
            }
            else{
                return match;
            }
        }
        return text.replace(exp,searchhighlight);
    }
    else{
        var ind = [];
        for (var i = 0; i < voc.length; ++i){
            ind.push(i);
        }
        ind.sort(function (a,b) {return Math.abs(coef[b])-Math.abs(coef[a])});
        var redones=[];
        var greenones=[];
        for (var i=0; i< Math.min(display_limit, voc.length); ++i){
            if (coef[ind[i]]>0){
                greenones.push(voc[ind[i]]);
            }
            else if(coef[ind[i]]<0){
                redones.push(voc[ind[i]]);
            }
        }
        var exp=/(\w+)/g;
        function redorgreen(match){
            if(greenones.indexOf(stemmer(match.toLowerCase()))>-1){
                return "<span class='tooltip' style='background-color: green'>"+match+"<span class='tooltiptext'>"+coef[voc.indexOf(stemmer(match.toLowerCase()))]+"</span></span>";
            }
            else if(redones.indexOf(stemmer(match.toLowerCase()))>-1){
                return "<span class='tooltip' style='background-color: red'>"+match+"<span class='tooltiptext'>"+coef[voc.indexOf(stemmer(match.toLowerCase()))]+"</span></span>";
            }
            else{
                return match;
            }
        }
        
        return text.replace(exp,redorgreen);
    }    
}



