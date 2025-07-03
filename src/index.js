/**
 * Stub function to convert chessboard image to FEN.
 *
 * @param {HTMLImageElement | HTMLCanvasElement | Blob | string} input
 * @returns {Promise<string>} A hardcoded FEN string
 */
export async function board2fen(input) {
    await new Promise((res) => setTimeout(res, 300));
  
    if (typeof input === 'string') {
      console.log('Input is base64 string');
    } else if (input instanceof Blob) {
      console.log('Input is Blob');
    } else if (input instanceof HTMLImageElement) {
      console.log('Input is HTMLImageElement');
    } else if (input instanceof HTMLCanvasElement) {
      console.log('Input is HTMLCanvasElement');
    } else {
      throw new Error('Unsupported input type');
    }
  
    return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  }