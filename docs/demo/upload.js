'use strict';
let image = new Image();
let dw, dh;
let objs = [];
let objsConvas = [];
let colors = ['red', 'blue'];
let IMG_SIZE = 224;

// set variables for track mouse moves:
let canvasx;
let canvasy;
let last_mousex = 0;
let last_mousey = 0;
let mousex = 0;
let mousey = 0;
let mousedown = false;


image.onload = function() {
	// clear the object selection page
	$('generate_btn').el.style.display = "none";
	objs = [];
	clearConvas('canvas');
	clearConvas('canvas1');
	clearConvas('canvas2');
	objsConvas = [$('canvas1').el, $('canvas2').el];
	objsConvas.forEach(function(el){
		 el.parentElement.style.display = "none";
	})

	// go to the object selection page
	next_page();
	$('canvas').el.parentElement.style.display = "block";

	let ratio = IMG_SIZE / image.width; // Math.min(image.width, image.height);
	dw = image.width * ratio;
	dh = image.height * ratio;

	// reset the size of the main canvas and refresh the image
	$('canvas').el.width = dw;
	$('canvas').el.height = dh;
	refreshConvas(image, objs, objsConvas);

	canvasx = $('canvas').el.getBoundingClientRect().left;
	canvasy = $('canvas').el.getBoundingClientRect().top;
}

function clearConvas(target='canvas') {
	let ctx = $(target).el.getContext('2d');
	// reset the size of the
	$(target).el.width = dw;
	$(target).el.height = dh;
	ctx.clearRect(0,0,$(target).el.width,$(target).el.height); //clear canvas	
}

function refreshConvas(image, objs, objsConvas, target='canvas', obj_size=IMG_SIZE) {
	let ctx = $(target).el.getContext('2d');
	// reset the size of the
	$(target).el.width = dw;
	$(target).el.height = dh;
	let ratio = image.width / dw;
	ctx.clearRect(0,0,$(target).el.width,$(target).el.height); //clear canvas
	ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, dw, dh);
	objs.forEach(function(bbox, index) {
		let x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3];
		let octx = objsConvas[index].getContext('2d')
		objsConvas[index].width = obj_size;
		objsConvas[index].height = obj_size;
		objsConvas[index].parentElement.style.display = "block";
		objsConvas[index].parentElement.parentElement.style.display = "block";
		octx.clearRect(0,0,objsConvas[index].width,objsConvas[index].height); //clear canvas
		octx.drawImage(image, x * ratio, y * ratio, w * ratio, h * ratio, 0, 0, objsConvas[index].width, objsConvas[index].height);
		
		ctx.beginPath();
		ctx.rect(x,y,w,h);
		if (obj_size != IMG_SIZE) { // if it is visualisation of the final page
			ctx.strokeStyle = 'green';
		} else {
			ctx.strokeStyle = colors[index];
		}
		ctx.lineWidth = 3;
		ctx.stroke();
	});
	if (obj_size != IMG_SIZE) { // if it is visualisation of the final page
		objsConvas[0].parentElement.style.top = dh + 10;
		objsConvas[1].parentElement.style.top = dh + 10;
		objsConvas[1].parentElement.nextElementSibling.style.top = dh + 10 + 110;
	}
}
