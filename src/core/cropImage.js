export function cropImageToBox(img, box) {
    const canvas = document.createElement('canvas');
    const width = Math.round(box.x2 - box.x1);
    const height = Math.round(box.y2 - box.y1);
    canvas.width = width;
    canvas.height = height;
  
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, box.x1, box.y1, width, height, 0, 0, width, height);
    return canvas;
}