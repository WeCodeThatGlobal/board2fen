import * as tf from '@tensorflow/tfjs';

export async function detectChessboard(img, detectionModel) {
    const modelSize = 640;
    const tfImg = tf.browser.fromPixels(img);
    const [h, w] = tfImg.shape;
    const maxSize = Math.max(h, w);

    const padded = tfImg.pad([[0, maxSize - h], [0, maxSize - w], [0, 0]]);
    const input = tf.image.resizeBilinear(padded, [modelSize, modelSize]).div(255).expandDims(0);
    const res = detectionModel.execute(input).transpose([0, 2, 1]);
    const scores = res.slice([0, 0, 4], [-1, -1, 1]).squeeze([0]).max(1);

    const wT = res.slice([0, 0, 2], [-1, -1, 1]);
    const hT = res.slice([0, 0, 3], [-1, -1, 1]);
    const x1 = tf.sub(res.slice([0, 0, 0], [-1, -1, 1]), wT.div(2));
    const y1 = tf.sub(res.slice([0, 0, 1], [-1, -1, 1]), hT.div(2));
    const boxes = tf.concat([y1, x1, tf.add(y1, hT), tf.add(x1, wT)], 2).squeeze();

    const selected = await tf.image.nonMaxSuppressionAsync(boxes, scores, 1, 0.45, 0.2);
    const bestBox = boxes.gather(selected, 0).dataSync();

    const scaleX = maxSize / w;
    const scaleY = maxSize / h;

    return {
        x1: (bestBox[1] / modelSize) * maxSize / scaleX,
        y1: (bestBox[0] / modelSize) * maxSize / scaleY,
        x2: (bestBox[3] / modelSize) * maxSize / scaleX,
        y2: (bestBox[2] / modelSize) * maxSize / scaleY,
    };
}