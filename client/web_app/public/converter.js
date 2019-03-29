var ndarray = require('ndarray')
var onnx = require('onnxjs')
const fs = require('fs')
var csv = require('csv')
var Papa = require('papaparse')
var MicroModal = require('micromodal');
let parsed_csv
const y_pred = []

MicroModal.init();

// ONNX button related events
const realOnnxBtn = document.getElementById('real-onnx')
const customOnnxBtn = document.getElementById('custom-onnx-btn')
const scoreBtn = document.getElementById('score-btn')


customOnnxBtn.addEventListener('click', function() {
  realOnnxBtn.click();
});

realOnnxBtn.addEventListener('change', function() {
	if (realOnnxBtn.value) {
		customOnnxBtn.innerHTML = 'ONNX Model uploaded'
		customOnnxBtn.className = "btn btn-lg btn-success btn-block text-uppercase";
    customOnnxBtn.setAttribute("disabled", "");
    const selectedOnnxModal = document.getElementById('real-onnx').files[0];
    console.log(selectedOnnxModal)
	}
})


// CSV button related events
const realCsvBtn = document.getElementById('real-csv')
const customCsvBtn = document.getElementById('custom-csv-btn')


customCsvBtn.addEventListener('click', function() {
  realCsvBtn.click();
});

realCsvBtn.addEventListener('change', function() {
	if (realCsvBtn.value) {
		customCsvBtn.innerHTML = 'CSV file uploaded'
    customCsvBtn.className = "btn btn-lg btn-success btn-block text-uppercase";
    customCsvBtn.setAttribute("disabled", "");
    const selectedFile = document.getElementById('real-csv').files[0];
    Papa.parse('./x_test', config = {
      skipEmptyLines: true,
      // header:true,
      dynamicTyping: true,
      complete: function(results, file) {
      parsed_csv = results;
      final_csv = [].concat.apply([], parsed_csv.data);
      // console.log(final_csv)
        }
    })
	}
})

var x_test_copy = []
scoreBtn.addEventListener('click', async function() {
  test_data = ndarray(new Float32Array(final_csv), [61, 5])
  x_test = test_data.hi(61,4)
  for (var i = 0; i < x_test.shape[0]; i++) {
    for (var j = 0; j < x_test.shape[1]; j++) {
      x_test_copy.push(x_test.get(i,j))
    }
  }
  // create a session
  const myOnnxSession = new onnx.InferenceSession();

    await myOnnxSession.loadModel("./onnx_model");
    const inputTensor = new onnx.Tensor(x_test_copy, 'float32', x_test.shape);


    for (var i = 0; i < 60; i++) {
      var collect_array = []
      for (var j = 0; j < 4; j++) {
        collect_array.push(inputTensor.get([i,j]))
        if(collect_array.length == 4){
          const collectArrayTensor = new onnx.Tensor(collect_array, 'float32', [1, 4]);
          const outputMap = await myOnnxSession.run([collectArrayTensor]);
          const outputData = outputMap.values().next().value.data;
          y_pred.push(outputData)
        }
      }
    }

    for(var i = 0; i < 60; i++){
      if(y_pred[i] < 0.5){
        y_pred[i] = 0;
      }
      else{
        y_pred[i] = 1;
      }
    }

    var y_test_copy = []

    y_test = test_data.lo(0,4)
    for (var i = 0; i < 60; i++) {
        y_test_copy.push(y_test.get(i,0))
    }

    var count = 0;
    for (var i = 0; i < 60; i++) {
      if(y_test_copy[i] == y_pred[i]){
        count += 1
       }
      }
      console.log(count)
    score = Math.round(count/y_test.shape[0] * 100)
    setTimeout(function(){
      document.getElementById('score-value').innerHTML = score;
      document.getElementById('loader-icon').style.display = "none";
      document.getElementById('score-value').innerHTML = score + "%";
      scoreBtn.innerHTML = 'Refresh to rerun the test'
      scoreBtn.setAttribute("disabled", "");
    }, 1000)
    
    console.log(y_test_copy, '--y_test')
    console.log(score, '--score')

});


// TODO
// 1) Loading test_data -- x_test + y_test
// 2) Preprocessing test_data
// 3) Run the model on test_data
// 4) Display score as pop-up
// 5) Disable and color change of buttons
// 6) Help button - Info on model and test_data format. Also links to download sample datasets
// 7) Option to load in-browser prebuilt model and test_data
