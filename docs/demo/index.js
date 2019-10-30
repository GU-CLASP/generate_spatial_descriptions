/**
 * This file is modified heavily from a source code of the tutorial on tensorflow.js
 * The original license was as following:
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

//import * as tf from '@tensorflow/tfjs';

import {IX2WORD} from './ix2word.js';

const MOBILENET_MODEL_PATH = './models/mobilenet/model.json';

const LANGUAGE_MODEL_PATH = './models/language_model/model.json';
const LANGUAGE_MODEL_TOP_PATH = './models/language_model_top/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet, lm, lm_top;

const mobilenetDemo = async () => {
  status('Loading model...');
  const startTime1 = performance.now();
  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  lm = await tf.loadLayersModel(LANGUAGE_MODEL_PATH);
  lm_top = await tf.loadLayersModel(LANGUAGE_MODEL_TOP_PATH);
  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  lm.predict([
    tf.zeros([1, 49, 1280]),
    tf.zeros([1, 11]),
    tf.zeros([1, 2, 1280]),
    tf.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]),
  ]);

  lm_top.predict([
    tf.zeros([1, 17, 200]),
  ]).dispose();

  const totalTime1 = performance.now() - startTime1;

  status(`Model loaded in ${Math.floor(totalTime1)} ms`);
  document.getElementById('goto_upload_btn').style.display = "inline-block";
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElements, sf) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const vfg = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElements[0]).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched).reshape([1, 49, 1280]);
  });

  let startTime3;
  const vfs = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img1 = tf.browser.fromPixels(imgElements[1]).toFloat();
    const img2 = tf.browser.fromPixels(imgElements[2]).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized1 = img1.sub(offset).div(offset);
    const normalized2 = img2.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = tf.concat([
      normalized1.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]),
      normalized2.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]),
    ], 0);
    //const batched1 = normalized1.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3])
    //const batched2 = normalized2.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3])

    startTime3 = performance.now();
    // Make a prediction through mobilenet.
    let out = mobilenet.predict(batched);
    out = tf.mean(out.reshape([2, 49, 1280]), 1);
    out = out.reshape([1, 2, 1280]);
    //let out1 = mobilenet.predict(batched1);
    //let out2 = mobilenet.predict(batched2);
    //return [out1, out2];
    return out;
  });

  let startTime4;
  const att_results = tf.tidy(() => {
    // beam search
    function search(callback, k=1, sequence_max_len=16) {
      // (log(1), initialize_of_zeros)
      let k_beam = [[0, [1].concat((new Array(sequence_max_len)).fill(0))]]

      // l : point on target sentence to predict
      for (let l=0; l < sequence_max_len; l++) {
        let all_k_beams = []
        k_beam.forEach(function(el) {
          const prob = el[0], sent_predict = el[1];
          // callback predicts the sequences with given states:
          const predicted_t = callback(sent_predict);
          const predicted = predicted_t.arraySync()[0];

          // arg top k!
          const possible_k = predicted[l].map((x,i) => [x,i]).sort((a,b) => a[0] < b[0] ? -1 : 1).slice(-k).map((x,i) => x[1]).reverse();

          // add to all possible candidates for k-beams
          possible_k.forEach(function(next_wid) {
            let logprob = 0;
            for (let i=0; i<l ; i++) {
              logprob += Math.log(predicted[i][sent_predict[i+1]]);
            }
            logprob += Math.log(predicted[l][next_wid]);

            let sent = sent_predict.slice(0, l+1).concat([next_wid]).concat((new Array(sequence_max_len-l-1)).fill(0));
            all_k_beams.push([logprob,sent]);
          })
        })

        // top k
        k_beam = all_k_beams.sort().slice(-k).reverse()
      }
      return k_beam;
    };

    const model_predict_fn = function (sent, returnBeta=false) {
      const outs = lm.predict([
        vfg, //tf.zeros([1, 7 * 7, 1280]),
        tf.tensor([sf]), //tf.zeros([1, 11]),
        vfs, //tf.zeros([1, 2, 1280]),
        tf.tensor([sent]),
      ]);

      if (returnBeta) {
        const beta = tf.mul(outs[0], tf.norm(outs[1], 2, 3));
        return tf.div(beta, tf.sum(beta, 2).expandDims(2));
      } else {
        const c = tf.sum(outs[0].reshape([1, 17, 4, 1]).mul(outs[1]), 2);
        const p = lm_top.predict(c);
        return p;
      }

    };
    let results = [];
    const out = search(model_predict_fn, 5);
    out.forEach(function(item, index) {
      results.push({
        'logprob': item[0],
        'seq': [],
      });
      const beta = model_predict_fn(item[1], true).arraySync()[0];
      for (let i=1; i<item[1].length; i++) {
        if ((item[1][i] == 0) || (item[1][i] == 1)) {
          break;
        } else {
          results[index]['seq'].push([IX2WORD[item[1][i]]].concat(beta[i-1]));
        }
      }
    });

    // remove repetitions
    let _results = [];
    let _keys = [];    
    results.forEach((item) => {
      let k = item['seq'].map((x) => x[0]).join(" ").trim();
      if (_keys.includes(k)) {
        return
      } else {
        _results.push(item);
        _keys.push(k)
      }
    });
    
    startTime4 = performance.now();
    return _results;
  });

  // Convert logits to probabilities and class names.
  //const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  const totalTime3 = performance.now() - startTime3;
  const totalTime4 = performance.now() - startTime4;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
         `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Show the classes in the DOM.
  //showResults(imgElement, classes);
  //console.log(att_results);
  showResults(att_results);
}

function generate_description(e) {
  $("predictions").el.innerHTML = "Processing ..."
  // VisKE: geometric relation between two bounding boxes:
  // spatial features
  // based on VisKE
  let sbj_bbx = [objs[0][0] / dw, objs[0][1] / dh, objs[0][2] / dw, objs[0][3] / dh];
  let obj_bbx = [objs[1][0] / dw, objs[1][1] / dh, objs[1][2] / dw, objs[1][3] / dh];
  let a1, a2, w, h, overlap_a, sf1;
  // area of each bbox:
  a1 = sbj_bbx[2] * sbj_bbx[3]
  a2 = obj_bbx[2] * obj_bbx[3]
  // overlap width:
  if (obj_bbx[0] <= sbj_bbx[0] && sbj_bbx[0] <= obj_bbx[0]+obj_bbx[2] && obj_bbx[0]+obj_bbx[2] <= sbj_bbx[0] + sbj_bbx[2]) {
    // overlap
    w = (obj_bbx[0]+obj_bbx[2]) - (sbj_bbx[0])
  } else {
    if (obj_bbx[0] <= sbj_bbx[0] &&
        sbj_bbx[0] <= sbj_bbx[0] + sbj_bbx[2] &&
        sbj_bbx[0] + sbj_bbx[2] <= obj_bbx[0]+obj_bbx[2]) {
      // obj contains sbj
      w = sbj_bbx[2]
    } else {
      if (sbj_bbx[0] <= obj_bbx[0] &&
          obj_bbx[0] <= sbj_bbx[0] + sbj_bbx[2] &&
          sbj_bbx[0] + sbj_bbx[2] <= obj_bbx[0]+obj_bbx[2]) {
        // overlaps
        w = (sbj_bbx[0]+sbj_bbx[2]) - (obj_bbx[0])
      } else {
        if (sbj_bbx[0] <= obj_bbx[0] &&
            obj_bbx[0] <= obj_bbx[0]+obj_bbx[2] &&
            obj_bbx[0]+obj_bbx[2] <= sbj_bbx[0] + sbj_bbx[2]) {
          // subj contains obj
          w = obj_bbx[2]
        } else {
          w = 0
        }
      }
    }
  }

  // overlap height:
  if (obj_bbx[1] <= sbj_bbx[1] &&
      sbj_bbx[1] <= obj_bbx[1]+obj_bbx[3] &&
      obj_bbx[1]+obj_bbx[3] <= sbj_bbx[1] + sbj_bbx[3]) {
    // overlap
    h = (obj_bbx[1]+obj_bbx[3]) - (sbj_bbx[1])
  } else {
    if (obj_bbx[1] <= sbj_bbx[1] &&
        sbj_bbx[1] <= sbj_bbx[1] + sbj_bbx[3] &&
        sbj_bbx[1] + sbj_bbx[3] <= obj_bbx[1]+obj_bbx[3]) {
      // obj contains sbj
      h = sbj_bbx[3]
    } else {
      if (sbj_bbx[1] <= obj_bbx[1] &&
          obj_bbx[1] <= sbj_bbx[1] + sbj_bbx[3] &&
          sbj_bbx[1] + sbj_bbx[3]<= obj_bbx[1]+obj_bbx[3]) {
        // overlaps
        h = (sbj_bbx[1]+sbj_bbx[3]) - (obj_bbx[1])
      } else {
        if (sbj_bbx[1] <= obj_bbx[1] &&
            obj_bbx[1] <= obj_bbx[1]+obj_bbx[3] &&
            obj_bbx[1]+obj_bbx[3] <= sbj_bbx[1] + sbj_bbx[3]) {
          // subj contains obj
          h = obj_bbx[3]
        } else {
          h = 0
        }
      }
    }
  }

  // overlap area
  overlap_a = w * h

  sf1 = [
    obj_bbx[0] - sbj_bbx[0] + (obj_bbx[2] - sbj_bbx[2])/2, // dx
    obj_bbx[1] - sbj_bbx[1] + (obj_bbx[3] - sbj_bbx[3])/2, // dy
    ((a1+a2) == 0) ? 0 : overlap_a/(a1+a2), // ov
    (a1 == 0)      ? 0 : overlap_a/a1, // ov1
    (a2 == 0)      ? 0 : overlap_a/a2, // ov2
    sbj_bbx[3], // h1
    sbj_bbx[2], // w1
    obj_bbx[3], // h2
    obj_bbx[2], // w2
    a1, // a1
    a2, // a2
  ]

  // Fill the image & call predict.
  let cimg = document.getElementById('canvas0');
  let obj1 = document.getElementById('canvas1');
  let obj2 = document.getElementById('canvas2');

  let img0 = document.createElement('img');
  img0.src = cimg.toDataURL("image/jpeg");
  img0.width = IMAGE_SIZE;
  img0.height = IMAGE_SIZE;
  img0.onload = function() {
    let img1 = document.createElement('img');
    img1.src = obj1.toDataURL("image/jpeg");
    img1.width = IMAGE_SIZE;
    img1.height = IMAGE_SIZE;
    img1.onload = function() {
      let img2 = document.createElement('img');
      img2.src = obj1.toDataURL("image/jpeg");
      img2.width = IMAGE_SIZE;
      img2.height = IMAGE_SIZE;
      img2.onload = function() {
        predict([img0, img1, img2], sf1);
      };
    };
  };
};

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
const trigger = document.getElementById('generate_btn');
trigger.addEventListener('click', generate_description);

function showResults(att_results) {
  const color_list = ["grey", "green", "red", "blue"];
  $('predictions').el.innerHTML = "";
  att_results.forEach((item, i) => {
    if (i > 2) { // skip results more than 3
      return
    }
    
    const seq = item['seq'];
    //console.log('logprob', logprob)
    let el = document.createElement('div');
    el.innerHTML = seq.map((x) => x[0]).join(" ");
    el.style["font-size"] = "10pt";
    el.style["text-align"] = "center";
    el.style.margine = "10px 0 0 0";
    $('predictions').el.appendChild(el);
    el = document.createElement('div');
    el.innerHTML = "logprob: " + item['logprob'].toFixed(3);
    $('predictions').el.appendChild(el);
    seq.forEach((w) => {
      let el = document.createElement('div');
      let el_text = document.createElement('div');
      let el_pbar = document.createElement('div');

      el_text.innerHTML = w[0];
      el_text.style.width = 80;
      el_text.style.height = 20;
      el_text.style.display = "inline-block";
      el_text.style.background = "white";
      //el_text.style["vertical-align"] = "middle";
      el_text.style["font-size"] = "10pt";
      el_text.style["vertical-align"] = "top";
      el_text.style["line-height"] = "20px";

      el_pbar.style.width = 219;
      el_pbar.style.height = 15;
      el_pbar.style.display = "inline-block";
      el_pbar.style.overflow = "hidden";
      el_pbar.style["border-radius"] = "18px";
      el_pbar.style["margine"] = "2px 0 0";
      
      for (let j = 1; j < 5; j++){
        let el_pbarx = document.createElement('div');
        el_pbarx.innerHTML = parseInt(w[j] * 100) + "%";
        el_pbarx.style.width = parseInt(w[j] * 220);
        el_pbarx.style.height = 20;
        el_pbarx.style.background = color_list[j-1];
        el_pbarx.style.color = "white";
        el_pbarx.style.display = "inline-block";
        el_pbarx.style["text-align"] = "center";
        el_pbarx.style["overflow"] = "hidden";
        el_pbarx.style["vertical-align"] = "middle";
        el_pbar.appendChild(el_pbarx);
      }

      //console.log('w', w[0], w[1])
      el.appendChild(el_text)
      el.appendChild(el_pbar)
      $('predictions').el.appendChild(el)
    });
  });
  document.getElementById('goto_about_btn').style.display = "inline-block";
}

mobilenetDemo();
