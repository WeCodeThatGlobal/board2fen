import * as tf from '@tensorflow/tfjs';

export async function loadModels(basePath = './nn/') {
    const [detectionModel, classificationModel, mobilenetFeatureVector] = await Promise.all([
        tf.loadGraphModel(`${basePath}detection_model/model.json`),
        tf.loadLayersModel(`${basePath}classification_model/model.json`),
        tf.loadGraphModel(`${basePath}mobilenet/model.json`),
    ]);

    tf.tidy(() => {
        detectionModel.execute(tf.zeros(detectionModel.inputs[0].shape));
        mobilenetFeatureVector.predict(tf.zeros([1, 224, 224, 3]));
    });

    return { detectionModel, classificationModel, mobilenetFeatureVector };
}