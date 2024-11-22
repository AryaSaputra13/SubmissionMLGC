const tf = require('@tensorflow/tfjs-node');

async function predictClassification(model, image) {
  const tensor = tf.node
    .decodeImage(image)
    .resizeNearestNeighbor([224, 224])
    .expandDims()
    .toFloat()

    const channels = tensor.shape[3];
    if (channels !== 3) {
        throw new Error('Non-RGB');
    }

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const label = confidenceScore > 50 ? 'Cancer' : 'Non-cancer';
    const suggestion = label === 'Cancer'
    ? 'Segera periksa ke dokter!'
    : 'Penyakit kanker tidak terdeteksi.';


    return { label, suggestion };
}

module.exports = predictClassification;