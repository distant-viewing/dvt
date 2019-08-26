"use strict";

var oReq = new XMLHttpRequest();

function padZeros(number, pad) {
  return ("0".repeat(pad) + number).toString().substr(-pad, pad)
}

function createFrameData(jsonObj) {
  var container = document.getElementById("container");
  var frame_info = document.getElementById("frame-default");
  var fps = jsonObj['meta'][0]['fps'];

  for(var i = 0; i < jsonObj.meta.length; i++) {
    let cln = frame_info.cloneNode(true);
    cln.style = "";

    let lobj = jsonObj['length'][i];

    cln.children[0].children[0].src =
      'img/display/frame-' + padZeros(i, 6) + ".png";
    cln.children[0].children[0].id = 'img' + (i).toString();

    let ulist = cln.children[1].children[0].children;
    ulist[0].children[1].textContent = lobj['num_faces'].toString();
    ulist[1].children[1].textContent = lobj['num_people'].toString();
    ulist[2].children[1].textContent = lobj['objects'].toString();
    ulist[3].children[1].textContent = lobj['shot_length'].toString();
    ulist[4].children[1].textContent = lobj['people'].toString();

    container.appendChild(cln);
  }
}

oReq.onreadystatechange = function () {
  var DONE = this.DONE || 4;
  if (this.readyState === DONE){
    createFrameData(JSON.parse(this.responseText));
    document.getElementById("help").style = "display: none;";

  }
};

oReq.open("GET", "data.json");
oReq.send(null);
