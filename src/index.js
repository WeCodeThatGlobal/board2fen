import * as tf from '@tensorflow/tfjs';
import { loadImage } from './core/utils';
import { loadModels } from './core/modelLoader';
import { detectChessboard } from './core/detectChessboard';
import { cropImageToBox } from './core/cropImage';
import { classifyBoard } from './core/classifyBoard';

let models = null;

/**
 * @param {HTMLImageElement | HTMLCanvasElement | Blob | string} input
 * @returns {Promise<string>}
 */
export async function board2fen(input) {
  const image = await loadImage(input);

  if (!models) {
    models = await loadModels('/nn/');
  }

  const box = await detectChessboard(image, models.detectionModel);

  if (!box) throw new Error('No chessboard detected');

  const cropped = cropImageToBox(image, box);

  const fen = classifyBoard(cropped, models);

  return fen;
}
