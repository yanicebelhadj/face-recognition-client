/**
 * Fabrication de vignettes.
 *
 * Sert au repli `localStorage` : une photo d'origine pèse souvent plusieurs
 * mégaoctets, largement au-dessus du quota (~5 Mo pour tout le domaine). Une
 * vignette JPEG de 160 px tient dans quelques kilooctets, ce qui permet de
 * conserver 25 profils sans approcher la limite.
 */
const THUMBNAIL_SIZE = 160;
const THUMBNAIL_QUALITY = 0.72;

/**
 * Réduit une image en carré, recadrée au centre, et la renvoie en data URL.
 * Renvoie `null` si le navigateur ne sait pas décoder le fichier.
 *
 * @param {Blob} blob
 * @returns {Promise<string|null>}
 */
export async function createThumbnail(blob, size = THUMBNAIL_SIZE) {
  const url = URL.createObjectURL(blob);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error("Image illisible."));
      element.src = url;
    });

    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    // Recadrage centré : on prend le plus grand carré de l'image d'origine.
    const side = Math.min(image.naturalWidth, image.naturalHeight);
    const offsetX = (image.naturalWidth - side) / 2;
    const offsetY = (image.naturalHeight - side) / 2;
    canvas.getContext("2d").drawImage(image, offsetX, offsetY, side, side, 0, 0, size, size);

    return canvas.toDataURL("image/jpeg", THUMBNAIL_QUALITY);
  } catch {
    return null;
  } finally {
    URL.revokeObjectURL(url);
  }
}
