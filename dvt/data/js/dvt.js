/**
 * @fileoverview Functions for Distant Viewing Toolkit's visualization engine.
 * @author tbarnold@protonmail.ch (Taylor Arnold)
 */

/* jshint esversion: 6 */
/* globals $:false */
/* exported runMain, runImage, runVideo */

/**
 * Convert integer to string with (minimum) fixed width of zeros padded on
 * the left.
 *
 * @param {number} number A numeric value to pad.
 * @param {number} pad Integer number of places to pad.
 * @return {string} The padded integer as a string.
 */
function padZeros(number, pad) {
    return ("0".repeat(pad) + number).toString().substr(-pad, pad);
}

/**
 * Convert seconds to HH:MM:SS format.
 *
 * @param {number} total_seconds Integer number of seconds to display.
 * @return {string} The formated time as a straing.
 */
function formatTime(total_seconds) {

    var total_minutes = Math.floor(total_seconds / 60);
    var seconds = total_seconds % 60;
    var hours = Math.floor(total_minutes / 60);
    var minutes = total_minutes % 60;

    var time_start = padZeros(hours, 2) + ":" + padZeros(minutes, 2) + ":" +
        padZeros(seconds, 2);

    return time_start;
}

/**
 * Functional to create click event on an element that calls the modal element.
 *
 * @param {string} desc Description for the modal.
 * @param {style} src Path to the image in the modal; when not set, uses the
 *    source of the calling element.
 */
function setImageClick(desc, src) {
    return function() {
        var hoverlay = document.getElementById("overlay-head");
        hoverlay.children[0].textContent = desc;
        var overlay = document.getElementById("overlay");
        overlay.children[0].src =
            (typeof src !== 'undefined') ? src : this.src;
        overlay.onclick = function() {
            $('#myModal').modal('hide');
        };
        $('#myModal').modal('show');
    };
}

/**
 * Setup an image with desired thumbnail and click event.
 *
 * @param {object} item Image object to modify.
 * @param {string} img Path to set the image's source to.
 * @param {string} desc Description for the modal.
 * @param {style} item Additional style to apply to the image, if desired.
 */
function setImage(item, img, desc, style) {
    item.src = 'img/' + img;
    item.onclick = setImageClick(desc);
    if (typeof style !== 'undefined') {
        item.style = style;
    }
}

/**
 * Format JSON data from image annotations and create web elements to
 * visualize the output in the browser.
 *
 * @param {object} jsonObj Data extracted from the Distant Viewing Toolkit.
 */
function createImageFrameData(jsonObj) {
    var container = document.getElementById("container");
    var frame_info = document.getElementById("frame-default");

    jsonObj.length.forEach(function(elem, index) {
        // clone the template object
        let cln = frame_info.cloneNode(true);
        cln.style = "";

        // set the annotated thumbnail image
        setImage(
            cln.children[0].children[0],
            "thumb" + '/frame-' + padZeros(index, 6) + ".png",
            "Original Image",
            "height: 250px;"
        );

        // set the original thumbnail image
        setImage(
            cln.children[0].children[1],
            "display" + '/frame-' + padZeros(index, 6) + ".png",
            "Detected objects (blue) and faces (orange)",
            "height: 250px;"
        );

        // fill in metadata
        let ulist = cln.children[1].children[0].children;
        ulist[0].children[1].textContent = elem.num_faces.toString();
        ulist[1].children[1].textContent = elem.num_people.toString();
        ulist[2].children[1].textContent = elem.objects.toString();
        ulist[3].children[1].textContent = elem.shot_length.toString();
        ulist[4].children[1].textContent = elem.people.toString();

        // add to the page
        container.appendChild(cln);
    });
}

/**
 * Format JSON data from video annotations and create web elements to
 * visualize the output in the browser.
 *
 * @param {object} jsonObj Data extracted from the Distant Viewing Toolkit.
 */
function createVideoFrameData(jsonObj) {
    var container = document.getElementById("container");
    var frame_info = document.getElementById("frame-default");
    var fps = jsonObj.meta[0].fps;

    jsonObj.length.forEach(function(elem, index) {
        // clone the template object
        let cln = frame_info.cloneNode(true);
        cln.style = "";

        // compute frame, start time and stop time
        let frame_num = jsonObj.cut[index].mpoint;
        let time_start = Math.floor(jsonObj.cut[index].frame_start / fps);
        let time_end = Math.floor(jsonObj.cut[index].frame_end / fps);
        let sound_flag = ("power" in jsonObj);

        // set the annotated thumbnail image
        setImage(
            cln.children[0].children[0],
            "display" + '/frame-' + padZeros(frame_num, 6) + ".png",
            "Detected objects (blue) and faces (orange)",
            sound_flag ? "" : "height: 250px;"
        );

        setImage(
            cln.children[0].children[1],
            "flow" + '/frame-' + padZeros(frame_num, 6) + ".png",
            "Optical flow",
            sound_flag ? "" : "height: 250px;"
        );

        // determine if there was sound data included in the input
        if ("power" in jsonObj) {
            // resize the container for more images
            cln.children[0].style = "width:888px";
            cln.children[1].style = "font-size: x-small;";

            // add the tone
            setImage(
                cln.children[0].children[2],
                "tone" + '/frame-' + padZeros(index, 6) + ".png",
                "Audio tone",
                ""
            );

            // add the spectrogram
            setImage(
                cln.children[0].children[3],
                "spec" + '/frame-' + padZeros(index, 6) + ".png",
                "Spectrogram",
                ""
            );
        }

        // include metadata
        let ulist = cln.children[1].children[0].children;
        ulist[0].children[1].textContent = formatTime(time_start);
        ulist[1].children[1].textContent = formatTime(time_end);
        ulist[2].children[1].textContent = elem.num_faces.toString();
        ulist[3].children[1].textContent = elem.num_people.toString();
        ulist[4].children[1].textContent = elem.objects.toString();
        ulist[5].children[1].textContent = elem.shot_length.toString();
        ulist[6].children[1].textContent = elem.people.toString();

        // button to create the original frame
        cln.children[1].children[1].children[0].onclick = setImageClick(
            "Original frame",
            'img/frames/frame-' + padZeros(frame_num, 6) + ".png"
        );

        container.appendChild(cln);
    });
}

/**
 * Format JSON data to create table of contents page.
 *
 * @param {object} jsonObj Data extracted from the Distant Viewing Toolkit.
 */
function createMetaData(jsonObj) {
    var container = document.getElementById("container");
    var image_template = document.getElementById("template");

    jsonObj.forEach(function(elem) {
        var cln = image_template.cloneNode(true);
        cln.style = "";

        var link_elem = cln.children[0].children[0];
        link_elem.href = elem.video_name;
        link_elem.children[0].src = elem.thumb_path;
        link_elem.children[1].children[0].textContent = elem.video_name_long;

        container.appendChild(cln);
    });
}

/**
 * Read JSON file and execute a function.
 *
 * @param {str} path Relative path to a JSON data file.
 * @param {function} createElements Function to call on the JSON data.
 */
function run(path, createElements) {
    var oReq = new XMLHttpRequest();

    oReq.onreadystatechange = function() {
        var DONE = this.DONE || 4;
        if (this.readyState === DONE) {
            createElements(JSON.parse(this.responseText));
            document.getElementById("help").style = "display: none;";
        }
    };

    oReq.open("GET", path);
    oReq.send(null);
}

/* Functions to call on loading the respective pages to build dynamic pages. */
function runMain() {
    run("toc.json", createMetaData);
}

function runImage() {
    run("data/viz-data.json", createImageFrameData);
}

function runVideo() {
    run("data/viz-data.json", createVideoFrameData);
}
