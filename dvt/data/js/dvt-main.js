"use strict";

var oReq = new XMLHttpRequest();

function createVideoData(jsonObj) {
  var container = document.getElementById("container");
  var image_template = document.getElementById("template");

  for(var i = 0; i < jsonObj.length; i++) {
    let cln = image_template.cloneNode(true);
    cln.style = "";

    let link_elem = cln.children[0].children[0];
    link_elem.href = jsonObj[i]['video_name'];
    link_elem.children[0].src = jsonObj[i]['thumb_path'];
    link_elem.children[1].children[0].textContent =
      jsonObj[i]['video_name_long'];

    container.appendChild(cln);
  }

}

oReq.onreadystatechange = function () {
  var DONE = this.DONE || 4;
  if (this.readyState === DONE){
      createVideoData(JSON.parse(this.responseText));
      document.getElementById("help").style = "display: none;";
  }
};

oReq.open("GET", "toc.json");
oReq.send(null);
