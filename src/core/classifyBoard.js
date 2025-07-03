import * as tf from '@tensorflow/tfjs';
import { parseFen, getPieceFromIndex } from './utils';

export function classifyBoard(boardCanvas, models) {
    const features = extractTileFeatures(boardCanvas, models.mobilenetFeatureVector);
    const pieceArray = classifyTiles(features, models.classificationModel);
    return parseFen(pieceArray);
}

function extractTileFeatures(canvas, mobilenet) {
    const tileCanvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const tileSize = canvas.width / 8;
    const features = [];

    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
        tileCanvas.width = tileSize;
        tileCanvas.height = tileSize;

        const x = col * tileSize;
        const y = row * tileSize;
        const tileData = ctx.getImageData(x, y, tileSize, tileSize);
        tileCanvas.getContext('2d').putImageData(tileData, 0, 0);

        const tensor = tf.browser.fromPixels(tileCanvas)
            .resizeBilinear([224, 224])
            .div(255)
            .expandDims();

        const feature = mobilenet.predict(tensor).squeeze();
        features.push(feature);
        }
    }

    return features;
}

function classifyTiles(features, model) {
  return features.map((feature) => {
    const prediction = model.predict(feature.expandDims()).squeeze().arraySync();
    const maxIdx = prediction.indexOf(Math.max(...prediction));
    return getPieceFromIndex(maxIdx);
  });
}