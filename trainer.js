let mnist = require('mnist');
let fs = require('fs');
let save = fs.createWriteStream('result.csv', {
    flags: 'a'
})

let set = mnist.set(60000, 10000);

let trainingSet = set.training;
let testSet = set.test;

const INPUTS = 28*28;
const OUTPUTS = 10;
const EPOCHES = 300;
const BATCH_SIZE = 500;

let network = {
    biases: [],
    weights: [],
    weightGradient: [],
    biasesGradient: [],
}

for (let i = 0; i < OUTPUTS; i++) {
    network.biases[i] = Math.random();
    network.weights[i] = [];

    for (let j = 0; j < INPUTS; j++) {
        network.weights[i][j] = Math.random();
    }
}

let numberOfBatches = trainingSet.length / BATCH_SIZE;
console.log("Training stared");
for (let i = 0; i < EPOCHES; i++) {
    let batchPart = i % numberOfBatches;
    let batch = trainingSet.slice(batchPart * BATCH_SIZE, (batchPart + 1) * BATCH_SIZE)
    trainingStep(batch, 0.5, network);
    let accuracy = test(testSet, network)
    save.write((i + 1) + ',' + accuracy + '\n');
    console.log("Epoche: " + (i + 1) + " Accuracy:" + accuracy)
}

save.close()

function test(testSet, network) {
    let success = 0;
    for (let i = 0; i < testSet.length; i++) {
        let result = forward(testSet[i].input, network);
        let biggestValue = Math.max(...result);
        let expected = testSet[i].output;
        if (expected.indexOf(1) === result.indexOf(biggestValue)) {
            success++;
        }
    }
   return success/testSet.length;
}


function trainingStep(set, learningRate, network) {
    for (let i = 0; i < OUTPUTS; i++) {
        network.biasesGradient[i] = 0;
        network.weightGradient[i] = [];
        for (let j = 0; j < INPUTS; j++) {
            network.weightGradient[i][j] = 0;
        }
    }
    for (let i = 0; i < set.length; i++) {
        backpropagation(set[i].input, set[i].output.indexOf(1), network)
    }

    for (let i = 0; i < OUTPUTS; i++) {
        network.biases[i] -= learningRate * network.biasesGradient[i] / set.length;

        for (let j = 0; j < INPUTS; j++) {
            network.weights[i][j] -= learningRate * network.weightGradient[i][j] / set.length;
        }
    }
}

function backpropagation(image, label, network) {
    let result = forward(image, network);
    for (let i = 0; i < OUTPUTS; i++) {
        let biasesGradientPart = (i === label) ? result[i] - 1 : result[i]

        for (let j = 0; j < INPUTS; j++) {
            network.weightGradient[i][j] += biasesGradientPart * image[j];
        }
        network.biasesGradient[i] += biasesGradientPart * 1;
    }
}

function forward(image, network) {
    result = [];
    for (let i = 0; i < OUTPUTS; i++) {
        result[i] = network.biases[i];
        for (let j = 0; j < INPUTS; j++) {
            result[i] += network.weights[i][j]*image[j];
        }
    }
    return softmax(result)
}

function softmax(array) {
    return array.map(function(value,index) {
        return Math.exp(value) /
            array.map(
                function(y /*value*/){
                    return Math.exp(y)
                }
            ).reduce(
                function(a,b) {
                    return a+b
                }
            )
    })
}
