export async function loadImage(input) {
    if (typeof input === 'string') {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.src = input;
      await img.decode();
      return img;
    }
  
    if (input instanceof Blob) {
      const img = new Image();
      img.src = URL.createObjectURL(input);
      await img.decode();
      return img;
    }
  
    if (input instanceof HTMLImageElement || input instanceof HTMLCanvasElement) {
      return input;
    }
  
    throw new Error('Unsupported input type');
}



export function getPieceFromIndex(index) {
    const lookup = {
      0: 'p', 1: 'r', 2: 'n', 3: 'b', 4: 'q', 5: 'k',
      6: 'P', 7: 'R', 8: 'N', 9: 'B', 10: 'Q', 11: 'K',
      12: 's',
    };
    return lookup[index] || 's';
}
  
export function parseFen(arr) {
    const rows = [];
    for (let i = 0; i < 8; i++) {
      const row = [];
      for (let j = 0; j < 8; j++) {
        const piece = arr[i * 8 + j];
        if (piece === 's') {
          if (typeof row[row.length - 1] === 'number') {
            row[row.length - 1]++;
          } else {
            row.push(1);
          }
        } else {
          row.push(piece);
        }
      }
      rows.push(row.join(''));
    }
    return rows.join('/');
}